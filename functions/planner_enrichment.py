"""Post-generation enrichment for planner content.

After the LLM generates a plan, this module attaches REAL data from Google APIs:
- YouTube Data API v3 -> tutorial/workout/how-to videos (task.video)
- Google Places API (New, with legacy fallback) -> real venues (task.place)

Two stages:
  A) One cheap LLM call produces "directives": which tasks deserve a video or a
     place, and what to search for (queries deduped/reused across repeated tasks).
  B) Deterministic execution: dedupe queries, call Google APIs in parallel with
     hard caps and a wall-clock budget, attach results to tasks.

Design rules:
- Enrichment must NEVER fail generation: every failure path returns the content
  unchanged (or partially enriched).
- No key-bearing URLs are ever stored (plan docs get copied to public feeds).
- Quota guardrails: per-plan query caps, top-1 results, warm-instance TTL cache,
  and a YouTube quotaExceeded circuit breaker.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from cachetools import TTLCache

from generate_planner_content import (
    GeneratePlannerRequest,
    PlannerContent,
    ProgressCallback,
    TaskPlace,
    TaskVideo,
    get_openai_client,
)
from google_api_key import resolve_google_api_key


@dataclass
class EnrichmentConfig:
    directive_model: str = "gpt-5.4-mini"   # = ChatWrapperConfig.extraction_model
    max_video_queries: int = 8              # unique YouTube searches per plan
    max_place_queries: int = 10             # unique Places searches per plan
    max_video_attachments: int = 60         # tasks that may carry a video
    max_place_attachments: int = 40
    http_timeout_s: float = 6.0
    total_budget_s: float = 25.0            # hard wall-clock cap for Stage B


# Warm Cloud Run instances reuse search results across plans.
_search_cache: TTLCache = TTLCache(maxsize=512, ttl=6 * 3600)
_cache_lock = threading.Lock()

# YouTube quota circuit breaker: once we see quotaExceeded, skip video searches
# until (roughly) the daily quota reset. Module-level = per warm instance.
_youtube_disabled_until: float = 0.0


# ---------------------------------------------------------------------------
# Stage A — directive generation
# ---------------------------------------------------------------------------

_DIRECTIVE_SCHEMA: Dict[str, Any] = {
    "name": "enrichment_directives",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "directives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Positional task key, e.g. d3t1 (day 3, task index 1)",
                        },
                        "kind": {"type": "string", "enum": ["video", "place"]},
                        "query": {
                            "type": "string",
                            "description": "Search query. REUSE the exact same query for repeated tasks.",
                        },
                        "area": {
                            "type": ["string", "null"],
                            "description": "City/region for place searches (destination for travel plans). Null for videos.",
                        },
                    },
                    "required": ["task", "kind", "query", "area"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["directives"],
        "additionalProperties": False,
    },
}


def _build_compact_plan(content: PlannerContent, skip_existing: bool) -> List[Dict[str, str]]:
    """Flatten tasks to positional keys (LLM task ids are not guaranteed unique)."""
    compact: List[Dict[str, str]] = []
    for day in content.days:
        for idx, task in enumerate(day.tasks):
            if skip_existing and (task.video is not None or task.place is not None):
                continue
            entry = {"key": f"d{day.dayNumber}t{idx}", "text": (task.text or "")[:100]}
            if task.note:
                entry["note"] = task.note[:60]
            compact.append(entry)
    return compact


def _directive_system_prompt(config: EnrichmentConfig) -> str:
    return (
        "You select which tasks in a generated day-by-day plan should be enriched with real data.\n"
        "Two kinds of enrichment:\n"
        '- "video": a YouTube search for tasks where WATCHING a demonstration clearly helps '
        "(exercise form, workout follow-along, recipe, technique, tutorial, guided meditation).\n"
        '- "place": a Google Places search for tasks that happen at a REAL physical venue '
        "(visit a temple, eat at a local restaurant, train at a gym, museum, market, park).\n"
        "Rules:\n"
        "- Be conservative: only tag tasks where enrichment clearly adds value. Most tasks need NOTHING.\n"
        "- REUSE the exact same query string for repeated tasks (e.g. the same weekly workout) — "
        "one query can serve many tasks.\n"
        f"- At most {config.max_video_queries} UNIQUE video queries and "
        f"{config.max_place_queries} UNIQUE place queries across the whole plan.\n"
        "- For place directives, 'area' MUST be the city/region the task happens in — for travel plans "
        "infer the destination from the plan name, description and task texts. If you cannot infer any "
        "location, do not emit a place directive.\n"
        "- Never invent venue names as queries; query by what the user needs "
        '(e.g. "specialty coffee cafe", "muay thai gym") plus the area.\n'
        "- Write queries in the plan's language.\n"
        "- Video queries should be specific enough to land a good tutorial "
        '(e.g. "beginner bodyweight squat form" not "exercise").'
    )


def _generate_directives(
    content: PlannerContent,
    req: GeneratePlannerRequest,
    config: EnrichmentConfig,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    compact = _build_compact_plan(content, skip_existing)
    if not compact:
        return []

    user_message = json.dumps(
        {
            "planName": content.planName,
            "category": req.category,
            "language": req.language,
            "detailPrompt": (req.detailPrompt or "")[:500],
            "tasks": compact,
        },
        ensure_ascii=False,
    )

    response = get_openai_client().chat.completions.create(
        model=config.directive_model,
        messages=[
            {"role": "system", "content": _directive_system_prompt(config)},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_schema", "json_schema": _DIRECTIVE_SCHEMA},
    )
    if not response.choices or not response.choices[0].message.content:
        return []

    data = json.loads(response.choices[0].message.content)
    directives = data.get("directives") or []

    # Enforce caps in code regardless of what the prompt asked for.
    valid_keys = {entry["key"] for entry in compact}
    seen_queries: Dict[str, set] = {"video": set(), "place": set()}
    attach_counts = {"video": 0, "place": 0}
    max_queries = {"video": config.max_video_queries, "place": config.max_place_queries}
    max_attach = {"video": config.max_video_attachments, "place": config.max_place_attachments}

    kept: List[Dict[str, Any]] = []
    for d in directives:
        kind = d.get("kind")
        query = (d.get("query") or "").strip()
        key = (d.get("task") or "").strip()
        if kind not in ("video", "place") or not query or key not in valid_keys:
            continue
        norm = query.lower()
        if norm not in seen_queries[kind] and len(seen_queries[kind]) >= max_queries[kind]:
            continue  # over the unique-query budget
        if attach_counts[kind] >= max_attach[kind]:
            continue
        seen_queries[kind].add(norm)
        attach_counts[kind] += 1
        kept.append({"task": key, "kind": kind, "query": query, "area": d.get("area")})
    return kept


# ---------------------------------------------------------------------------
# Stage B — execution against Google APIs
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(r"&#(\d+);")


def _decode_entities(s: str) -> str:
    """YouTube returns titles with HTML entities (&amp;, &#39;, ...)."""
    out = (
        str(s or "")
        .replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return _ENTITY_RE.sub(lambda m: chr(int(m.group(1))), out).strip()


def _search_youtube(query: str, lang: str, key: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Top-1 YouTube search (port of EVOforluanching/src/services/evoYouTube.js)."""
    global _youtube_disabled_until
    if time.time() < _youtube_disabled_until:
        return None
    try:
        res = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "type": "video",
                "q": query,
                "maxResults": 1,
                "safeSearch": "moderate",
                "relevanceLanguage": "th" if lang == "th" else "en",
                "key": key,
            },
            timeout=timeout,
        )
        if res.status_code == 403 and "quotaExceeded" in res.text:
            # Daily quota is gone — stop burning calls until the next quota
            # window (resets midnight PT; 12h is a safe under-estimate).
            _youtube_disabled_until = time.time() + 12 * 3600
            print("Enrichment: YouTube daily quota exceeded — video enrichment disabled on this instance")
            return None
        res.raise_for_status()
        items = res.json().get("items") or []
        for it in items:
            video_id = (it.get("id") or {}).get("videoId")
            if not video_id:
                continue
            sn = it.get("snippet") or {}
            thumbs = sn.get("thumbnails") or {}
            thumb = (thumbs.get("medium") or thumbs.get("high") or thumbs.get("default") or {}).get("url")
            return {
                "videoId": video_id,
                "title": _decode_entities(sn.get("title")),
                "channel": _decode_entities(sn.get("channelTitle")) or None,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": thumb,
            }
        return None
    except Exception as exc:
        print(f"Enrichment: YouTube search failed for {query!r}: {exc}")
        return None


def _maps_url_for_place(place_id: Optional[str], name: str, lat: Optional[float], lng: Optional[float]) -> str:
    """Keyless Google Maps URL, safe to persist (mirrors evoPlaces.js mapsUrlForPlace)."""
    if place_id:
        from urllib.parse import quote
        return (
            "https://www.google.com/maps/search/?api=1"
            f"&query={quote(name or 'place')}&query_place_id={place_id}"
        )
    if lat is not None and lng is not None:
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
    from urllib.parse import quote
    return f"https://www.google.com/maps/search/?api=1&query={quote(name or '')}"


def _search_place(query: str, area: Optional[str], key: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Top-1 Places (New) text search with legacy fallback (port of evoPlaces.js)."""
    text_query = f"{query} {area}".strip() if area else query
    try:
        res = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": (
                    "places.displayName,places.formattedAddress,places.location,places.id,places.rating"
                ),
            },
            json={"textQuery": text_query, "maxResultCount": 1},
            timeout=timeout,
        )
        if res.ok:
            places = res.json().get("places") or []
            if places:
                p = places[0]
                loc = p.get("location") or {}
                place_id = (p.get("id") or "").replace("places/", "") or None
                name = (p.get("displayName") or {}).get("text") or text_query
                lat, lng = loc.get("latitude"), loc.get("longitude")
                return {
                    "name": name,
                    "address": p.get("formattedAddress"),
                    "lat": lat,
                    "lng": lng,
                    "rating": p.get("rating"),
                    "placeId": place_id,
                    "mapsUrl": _maps_url_for_place(place_id, name, lat, lng),
                }
            return None
        # New API unavailable (not enabled / no billing) -> legacy text search.
        res = requests.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": text_query, "key": key},
            timeout=timeout,
        )
        res.raise_for_status()
        results = res.json().get("results") or []
        if not results:
            return None
        p = results[0]
        loc = (p.get("geometry") or {}).get("location") or {}
        place_id = p.get("place_id")
        name = p.get("name") or text_query
        lat, lng = loc.get("lat"), loc.get("lng")
        return {
            "name": name,
            "address": p.get("formatted_address"),
            "lat": lat,
            "lng": lng,
            "rating": p.get("rating"),
            "placeId": place_id,
            "mapsUrl": _maps_url_for_place(place_id, name, lat, lng),
        }
    except Exception as exc:
        print(f"Enrichment: Places search failed for {text_query!r}: {exc}")
        return None


def _execute_searches(
    directives: List[Dict[str, Any]],
    lang: str,
    key: str,
    config: EnrichmentConfig,
) -> Dict[tuple, Optional[Dict[str, Any]]]:
    """Run deduped searches in parallel. Returns {(kind, normalized_query): result}."""
    unique: Dict[tuple, Dict[str, Any]] = {}
    for d in directives:
        cache_key = (
            "yt" if d["kind"] == "video" else "pl",
            f"{d['query'].lower()}|{(d.get('area') or '').lower()}",
        )
        unique.setdefault(cache_key, d)

    results: Dict[tuple, Optional[Dict[str, Any]]] = {}
    to_fetch: Dict[tuple, Dict[str, Any]] = {}
    with _cache_lock:
        for cache_key, d in unique.items():
            if cache_key in _search_cache:
                results[cache_key] = _search_cache[cache_key]
            else:
                to_fetch[cache_key] = d

    if not to_fetch:
        return results

    deadline = time.monotonic() + config.total_budget_s
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for cache_key, d in to_fetch.items():
            if d["kind"] == "video":
                fut = pool.submit(_search_youtube, d["query"], lang, key, config.http_timeout_s)
            else:
                fut = pool.submit(_search_place, d["query"], d.get("area"), key, config.http_timeout_s)
            futures[fut] = cache_key
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=max(config.total_budget_s, 1.0)):
                cache_key = futures[fut]
                try:
                    result = fut.result(timeout=0)
                except Exception:
                    result = None
                results[cache_key] = result
                with _cache_lock:
                    _search_cache[cache_key] = result
                if time.monotonic() > deadline:
                    break  # stragglers are abandoned; enrichment stays partial
        except concurrent.futures.TimeoutError:
            # Budget elapsed with futures still pending — keep what completed.
            print(f"Enrichment: search budget ({config.total_budget_s}s) elapsed, "
                  f"{len(results)}/{len(unique)} searches completed")

    return results


def _attach_results(
    content: PlannerContent,
    directives: List[Dict[str, Any]],
    results: Dict[tuple, Optional[Dict[str, Any]]],
) -> int:
    """Resolve positional keys and set task.video / task.place. Returns #attached."""
    task_by_key = {}
    for day in content.days:
        for idx, task in enumerate(day.tasks):
            task_by_key[f"d{day.dayNumber}t{idx}"] = task

    attached = 0
    for d in directives:
        task = task_by_key.get(d["task"])
        if task is None:
            continue
        cache_key = (
            "yt" if d["kind"] == "video" else "pl",
            f"{d['query'].lower()}|{(d.get('area') or '').lower()}",
        )
        result = results.get(cache_key)
        if not result:
            continue
        try:
            if d["kind"] == "video" and task.video is None:
                task.video = TaskVideo(**result)
                # Legacy renderers only know `link` — backfill with the real URL.
                if not task.link:
                    task.link = task.video.url
                attached += 1
            elif d["kind"] == "place" and task.place is None:
                task.place = TaskPlace(**result)
                attached += 1
        except Exception as exc:
            print(f"Enrichment: failed to attach {d['kind']} to {d['task']}: {exc}")
    return attached


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def enrich_planner_content(
    content: PlannerContent,
    req: GeneratePlannerRequest,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    skip_existing: bool = False,
    config: Optional[EnrichmentConfig] = None,
) -> PlannerContent:
    """Attach real YouTube videos / Google Places to tasks. Never raises."""
    config = config or EnrichmentConfig()

    key = resolve_google_api_key()
    if not key:
        print("Enrichment: no GOOGLE_API_KEY configured — skipping")
        return content

    try:
        if progress_callback:
            progress_callback({
                "progress": 94,
                "progress_message": "Adding real videos and places...",
                "current_stage": "enriching",
            })

        directives = _generate_directives(content, req, config, skip_existing)
        if not directives:
            print("Enrichment: no directives generated — nothing to attach")
            return content

        n_video = sum(1 for d in directives if d["kind"] == "video")
        n_place = len(directives) - n_video
        print(f"Enrichment: {len(directives)} directives ({n_video} video, {n_place} place)")

        results = _execute_searches(directives, req.language, key, config)
        attached = _attach_results(content, directives, results)
        print(f"Enrichment: attached {attached} items "
              f"({len([r for r in results.values() if r])} of {len(results)} searches hit)")
    except Exception as exc:
        print(f"Enrichment: failed, returning plan unenriched: {exc}")

    return content
