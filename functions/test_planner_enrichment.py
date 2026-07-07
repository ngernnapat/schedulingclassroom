#!/usr/bin/env python3
"""
Tests for planner_enrichment.py (post-generation YouTube/Places enrichment).

Runs offline: OpenAI + Google HTTP calls are mocked.
Usage: python test_planner_enrichment.py
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

os.environ["OPENAI_API_KEY"] = "test-key"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planner_enrichment
from planner_enrichment import (
    EnrichmentConfig,
    _attach_results,
    _build_compact_plan,
    _generate_directives,
    enrich_planner_content,
)
from generate_planner_content import (
    DayPlan,
    GeneratePlannerRequest,
    PlannerContent,
    PlannerSummary,
    Task,
    TimeStamp,
)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


def make_content(n_days=2, tasks_per_day=2):
    days = []
    for d in range(1, n_days + 1):
        tasks = [
            Task(id="dup", text=f"Day {d} task {i}: practice squats")
            for i in range(tasks_per_day)
        ]
        days.append(DayPlan(dayNumber=d, title=f"Day {d}", summary="s", tasks=tasks))
    return PlannerContent(
        planName="Test Plan",
        category="exercise",
        totalDays=n_days,
        createdAt=TimeStamp(seconds=1, nanoseconds=0),
        days=days,
        summary=PlannerSummary(overview="o", keyMilestones=["m"]),
    )


def make_req(**kw):
    base = dict(planName="Test Plan", category="exercise", totalDays=2, language="en")
    base.update(kw)
    return GeneratePlannerRequest(**base)


def fresh_state():
    planner_enrichment._search_cache.clear()
    planner_enrichment._youtube_disabled_until = 0.0


def mock_openai(directives):
    """Mock get_openai_client returning the given directives from Stage A."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps({"directives": directives})
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=msg)]
    )
    return client


def test_positional_keys():
    print("\n🧪 Positional key building + resolution")
    content = make_content()
    compact = _build_compact_plan(content, skip_existing=False)
    check("4 tasks flattened", len(compact) == 4)
    check("keys are positional", [c["key"] for c in compact] == ["d1t0", "d1t1", "d2t0", "d2t1"])

    # Attach a video via positional key even though task ids collide ("dup")
    directives = [{"task": "d2t1", "kind": "video", "query": "squat form", "area": None}]
    results = {("yt", "squat form|"): {
        "videoId": "abc", "title": "Squat Form", "channel": "Ch",
        "url": "https://www.youtube.com/watch?v=abc", "thumbnail": None,
    }}
    attached = _attach_results(content, directives, results)
    check("attached exactly 1", attached == 1)
    check("right task got the video", content.days[1].tasks[1].video is not None
          and content.days[1].tasks[0].video is None)
    check("link backfilled from video", content.days[1].tasks[1].link == "https://www.youtube.com/watch?v=abc")


def test_link_backfill_only_when_empty():
    print("\n🧪 Link backfill only when link is empty")
    content = make_content(n_days=1, tasks_per_day=1)
    content.days[0].tasks[0].link = "https://example.com/existing"
    directives = [{"task": "d1t0", "kind": "video", "query": "q", "area": None}]
    results = {("yt", "q|"): {"videoId": "v", "title": "T", "channel": None,
                              "url": "https://www.youtube.com/watch?v=v", "thumbnail": None}}
    _attach_results(content, directives, results)
    check("existing link preserved", content.days[0].tasks[0].link == "https://example.com/existing")
    check("video still attached", content.days[0].tasks[0].video is not None)


def test_directive_caps():
    print("\n🧪 Directive caps enforced in code")
    fresh_state()
    content = make_content(n_days=10, tasks_per_day=3)  # 30 tasks
    req = make_req(totalDays=10)
    # LLM tries 30 video directives with 30 unique queries — must be truncated
    directives = [
        {"task": f"d{d}t{t}", "kind": "video", "query": f"query {d}-{t}", "area": None}
        for d in range(1, 11) for t in range(3)
    ]
    cfg = EnrichmentConfig(max_video_queries=3)
    with patch.object(planner_enrichment, "get_openai_client", return_value=mock_openai(directives)):
        kept = _generate_directives(content, req, cfg, skip_existing=False)
    unique = {k["query"].lower() for k in kept}
    check(f"unique video queries capped at 3 (got {len(unique)})", len(unique) <= 3)

    # Reused queries do NOT count against the unique cap
    directives = [
        {"task": f"d{d}t0", "kind": "video", "query": "same workout", "area": None}
        for d in range(1, 11)
    ]
    with patch.object(planner_enrichment, "get_openai_client", return_value=mock_openai(directives)):
        kept = _generate_directives(content, req, cfg, skip_existing=False)
    check("reused query serves many tasks", len(kept) == 10)

    # Invalid keys / kinds dropped
    directives = [
        {"task": "d99t9", "kind": "video", "query": "x", "area": None},
        {"task": "d1t0", "kind": "banana", "query": "x", "area": None},
        {"task": "d1t0", "kind": "place", "query": "", "area": None},
    ]
    with patch.object(planner_enrichment, "get_openai_client", return_value=mock_openai(directives)):
        kept = _generate_directives(content, req, cfg, skip_existing=False)
    check("invalid directives dropped", kept == [])


def test_skip_existing():
    print("\n🧪 skip_existing (refine path)")
    content = make_content()
    from generate_planner_content import TaskVideo
    content.days[0].tasks[0].video = TaskVideo(videoId="v", title="T", url="u")
    compact = _build_compact_plan(content, skip_existing=True)
    check("enriched task excluded", all(c["key"] != "d1t0" for c in compact))
    check("others still included", len(compact) == 3)


def test_soft_fail_no_key():
    print("\n🧪 Soft-fail: no key → content unchanged")
    fresh_state()
    content = make_content()
    before = content.model_dump_json()
    with patch.object(planner_enrichment, "resolve_google_api_key", return_value=""):
        out = enrich_planner_content(content, make_req())
    check("content unchanged", out.model_dump_json() == before)


def test_soft_fail_llm_error():
    print("\n🧪 Soft-fail: Stage A explodes → content unchanged")
    fresh_state()
    content = make_content()
    before = content.model_dump_json()
    boom = MagicMock()
    boom.chat.completions.create.side_effect = RuntimeError("llm down")
    with patch.object(planner_enrichment, "resolve_google_api_key", return_value="k"), \
         patch.object(planner_enrichment, "get_openai_client", return_value=boom):
        out = enrich_planner_content(content, make_req())
    check("content unchanged", out.model_dump_json() == before)


def test_soft_fail_network_error():
    print("\n🧪 Soft-fail: search network errors → content unchanged")
    fresh_state()
    content = make_content()
    before = content.model_dump_json()
    directives = [
        {"task": "d1t0", "kind": "video", "query": "q1", "area": None},
        {"task": "d1t1", "kind": "place", "query": "gym", "area": "Bangkok"},
    ]
    with patch.object(planner_enrichment, "resolve_google_api_key", return_value="k"), \
         patch.object(planner_enrichment, "get_openai_client", return_value=mock_openai(directives)), \
         patch.object(planner_enrichment.requests, "get", side_effect=OSError("net down")), \
         patch.object(planner_enrichment.requests, "post", side_effect=OSError("net down")):
        out = enrich_planner_content(content, make_req())
    check("content unchanged", out.model_dump_json() == before)


def test_enrich_disabled():
    print("\n🧪 Kill switches")
    content = make_content()
    from generate_planner_content import ChatWrapper, ChatWrapperConfig
    wrapper = ChatWrapper(ChatWrapperConfig())

    req = make_req(enrich=False)
    with patch.object(planner_enrichment, "resolve_google_api_key", side_effect=AssertionError("should not be called")):
        out = wrapper._maybe_enrich(content, req)
    check("req.enrich=False skips entirely", out is content)

    os.environ["PLANNER_ENRICHMENT_DISABLED"] = "1"
    try:
        with patch.object(planner_enrichment, "resolve_google_api_key", side_effect=AssertionError("should not be called")):
            out = wrapper._maybe_enrich(content, make_req())
        check("env kill switch skips entirely", out is content)
    finally:
        del os.environ["PLANNER_ENRICHMENT_DISABLED"]


def test_youtube_circuit_breaker():
    print("\n🧪 YouTube quotaExceeded circuit breaker")
    fresh_state()
    resp = MagicMock()
    resp.status_code = 403
    resp.text = '{"error": {"errors": [{"reason": "quotaExceeded"}]}}'
    with patch.object(planner_enrichment.requests, "get", return_value=resp) as mocked:
        r1 = planner_enrichment._search_youtube("q1", "en", "k", 5.0)
        r2 = planner_enrichment._search_youtube("q2", "en", "k", 5.0)
    check("both searches return None", r1 is None and r2 is None)
    check("second call short-circuits (1 HTTP call)", mocked.call_count == 1)
    check("breaker armed", planner_enrichment._youtube_disabled_until > time.time())
    fresh_state()


def test_end_to_end_mocked():
    print("\n🧪 End-to-end with mocked APIs")
    fresh_state()
    content = make_content()
    directives = [
        {"task": "d1t0", "kind": "video", "query": "squat form", "area": None},
        {"task": "d2t0", "kind": "place", "query": "gym", "area": "Bangkok"},
    ]

    yt_resp = MagicMock()
    yt_resp.status_code = 200
    yt_resp.ok = True
    yt_resp.json.return_value = {"items": [{
        "id": {"videoId": "vid1"},
        "snippet": {"title": "Squat &amp; Form", "channelTitle": "FitCh",
                    "thumbnails": {"medium": {"url": "https://i.ytimg.com/x.jpg"}}},
    }]}
    yt_resp.raise_for_status = MagicMock()

    pl_resp = MagicMock()
    pl_resp.ok = True
    pl_resp.json.return_value = {"places": [{
        "id": "places/ChIJabc",
        "displayName": {"text": "Best Gym"},
        "formattedAddress": "123 Sukhumvit, Bangkok",
        "location": {"latitude": 13.7, "longitude": 100.5},
        "rating": 4.6,
    }]}

    with patch.object(planner_enrichment, "resolve_google_api_key", return_value="k"), \
         patch.object(planner_enrichment, "get_openai_client", return_value=mock_openai(directives)), \
         patch.object(planner_enrichment.requests, "get", return_value=yt_resp), \
         patch.object(planner_enrichment.requests, "post", return_value=pl_resp):
        out = enrich_planner_content(content, make_req())

    v = out.days[0].tasks[0].video
    p = out.days[1].tasks[0].place
    check("video attached", v is not None and v.videoId == "vid1")
    check("HTML entities decoded", v is not None and v.title == "Squat & Form")
    check("place attached", p is not None and p.name == "Best Gym")
    check("place id prefix stripped", p is not None and p.placeId == "ChIJabc")
    check("mapsUrl is keyless", p is not None and "key=" not in (p.mapsUrl or "") and "query_place_id=ChIJabc" in p.mapsUrl)
    check("untouched tasks untouched", out.days[0].tasks[1].video is None and out.days[0].tasks[1].place is None)

    # model_dump round-trip keeps enrichment (what gets written to Firestore)
    dumped = out.model_dump()
    check("model_dump carries video/place",
          dumped["days"][0]["tasks"][0]["video"]["videoId"] == "vid1"
          and dumped["days"][1]["tasks"][0]["place"]["lat"] == 13.7)
    # And re-validating (refine path) preserves enrichment
    revalidated = PlannerContent.model_validate(dumped)
    check("revalidation preserves enrichment", revalidated.days[0].tasks[0].video is not None)
    fresh_state()


if __name__ == "__main__":
    print("=" * 60)
    print("planner_enrichment tests")
    print("=" * 60)
    test_positional_keys()
    test_link_backfill_only_when_empty()
    test_directive_caps()
    test_skip_existing()
    test_soft_fail_no_key()
    test_soft_fail_llm_error()
    test_soft_fail_network_error()
    test_enrich_disabled()
    test_youtube_circuit_breaker()
    test_end_to_end_mocked()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
