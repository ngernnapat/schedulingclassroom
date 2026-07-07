"""AI-generated images for planner content: one cover + one theme image per week.

Runs in the BACKGROUND after plan generation completes (Node's
V2_processPlannerImageJob trigger calls the generate_planner_images endpoint,
which calls generate_plan_images below). The plan is already usable when this
runs — images patch into the plan doc afterwards and appear live via Firestore
listeners on the client.

Design rules (same philosophy as planner_enrichment.py):
- Per-image soft-fail: a missing week image never blocks the others.
- Hard caps: 1 cover + <=13 weekly images (90-day plan) = 14 max per plan.
- Wall-clock budget; stragglers are abandoned, partial results returned.
- Kill switch: env PLANNER_IMAGES_DISABLED.
- Consistent visual series: shared style suffix + per-category palette matching
  the client's CATEGORY_META colors (plannerDayDetailView.js).
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PlannerImageConfig:
    cover_size: str = "1024x1536"     # portrait crops well in drafts card + plan header
    weekly_size: str = "1024x1024"
    cover_quality: str = "medium"
    weekly_quality: str = "low"
    max_weekly_images: int = 13       # 90-day cap; hard cap 14 incl. cover
    max_workers: int = 3
    total_budget_s: float = 300.0


# Palette mirrors the client's CATEGORY_META (plannerDayDetailView.js) so the
# generated art matches each category's accent color in the app.
CATEGORY_PALETTE: Dict[str, tuple] = {
    "exercise":             ("energetic coral red", "#E85D4C"),
    "learning":             ("calm periwinkle blue", "#5B7FD6"),
    "travel":               ("fresh teal green", "#2E9B8F"),
    "finance":              ("grounded olive green", "#6B8E4E"),
    "health":               ("warm rose pink", "#D45D8C"),
    "personal_development": ("soft violet", "#9B6BB8"),
    "other":                ("warm terracotta", "#CE4F3C"),
}

CATEGORY_LABEL: Dict[str, str] = {
    "exercise": "fitness",
    "learning": "learning",
    "travel": "travel",
    "finance": "personal finance",
    "health": "health & wellness",
    "personal_development": "personal growth",
    "other": "lifestyle practice",
}

_STYLE = (
    "Modern flat vector illustration, minimal geometric shapes, soft rounded forms, "
    "subtle grain, generous negative space, dominant color {name} ({hex}) on a warm "
    "cream background (#F4F2EC), consistent stroke weight and palette across a series. "
    "Absolutely no text, letters, numbers, logos, or watermarks."
)

COVER_PROMPT = (
    'Cover artwork for a {total_days}-day {cat_label} plan called "{plan_name}". '
    "Visual theme: {overview}. Portrait composition, a single central motif. "
)

WEEKLY_PROMPT = (
    "Illustration {week} of {week_count} in the same visual series, for week {week} "
    "of a {cat_label} plan. This week's theme: {focus}. Square composition, one simple "
    "central motif that evolves in complexity across the series. "
)


def _style_for(category: str) -> str:
    name, hex_code = CATEGORY_PALETTE.get(category, CATEGORY_PALETTE["other"])
    return _STYLE.format(name=name, hex=hex_code)


def build_cover_prompt(plan_name: str, category: str, total_days: int, overview: str) -> str:
    cat_label = CATEGORY_LABEL.get(category, CATEGORY_LABEL["other"])
    theme = (overview or plan_name or "").strip()[:200] or plan_name
    return COVER_PROMPT.format(
        total_days=total_days, cat_label=cat_label,
        plan_name=(plan_name or "My Plan")[:80], overview=theme,
    ) + _style_for(category)


def build_weekly_prompt(
    plan_name: str, category: str, week: int, week_count: int, focus: Optional[str]
) -> str:
    cat_label = CATEGORY_LABEL.get(category, CATEGORY_LABEL["other"])
    theme = (focus or plan_name or "").strip()[:160] or plan_name
    return WEEKLY_PROMPT.format(
        week=week, week_count=week_count, cat_label=cat_label, focus=theme,
    ) + _style_for(category)


def week_ranges(total_days: int, max_weeks: int) -> List[Dict[str, int]]:
    """[{week, dayStart, dayEnd}] with week = ceil(dayNumber/7), capped at max_weeks."""
    total_days = max(1, min(int(total_days), 90))
    count = min(math.ceil(total_days / 7), max_weeks)
    return [
        {"week": w, "dayStart": (w - 1) * 7 + 1, "dayEnd": min(w * 7, total_days)}
        for w in range(1, count + 1)
    ]


def _generate_one(prompt: str, size: str, quality: str, upload_fn: Callable[[str], str]) -> Optional[str]:
    """Generate one image and upload it. Returns URL or None (soft-fail)."""
    try:
        from chatgpt_wrapper import get_default_wrapper
        generated = get_default_wrapper().generate_image(prompt, size=size, quality=quality)
        image_b64 = generated.get("b64_json")
        if not image_b64:
            return None
        return upload_fn(image_b64)
    except Exception as exc:
        print(f"Planner image failed ({size}): {exc}")
        return None


def generate_plan_images(
    *,
    user_id: str,
    plan_name: str,
    category: str,
    language: str = "en",
    total_days: int,
    overview: str = "",
    weekly_focus: Optional[List[str]] = None,
    upload_fn: Callable[[str], str],
    config: Optional[PlannerImageConfig] = None,
) -> Dict[str, Any]:
    """Generate cover + weekly theme images in parallel with soft-fail semantics.

    Returns {"coverImageUrl": str|None,
             "weekly": [{week, dayStart, dayEnd, imageUrl}],
             "failed": int}
    """
    config = config or PlannerImageConfig()

    if os.getenv("PLANNER_IMAGES_DISABLED"):
        print("Planner images: disabled via PLANNER_IMAGES_DISABLED")
        return {"coverImageUrl": None, "weekly": [], "failed": 0, "disabled": True}

    weeks = week_ranges(total_days, config.max_weekly_images)
    focus_list = weekly_focus or []
    week_count = len(weeks)

    cover_prompt = build_cover_prompt(plan_name, category, total_days, overview)
    weekly_prompts = {
        wr["week"]: build_weekly_prompt(
            plan_name, category, wr["week"], week_count,
            focus_list[wr["week"] - 1] if wr["week"] - 1 < len(focus_list) else None,
        )
        for wr in weeks
    }

    deadline = time.monotonic() + config.total_budget_s
    cover_url: Optional[str] = None
    weekly_urls: Dict[int, str] = {}
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures: Dict[concurrent.futures.Future, Any] = {
            pool.submit(_generate_one, cover_prompt, config.cover_size,
                        config.cover_quality, upload_fn): "cover"
        }
        for wr in weeks:
            fut = pool.submit(_generate_one, weekly_prompts[wr["week"]],
                              config.weekly_size, config.weekly_quality, upload_fn)
            futures[fut] = wr["week"]

        remaining = max(deadline - time.monotonic(), 1.0)
        done, not_done = concurrent.futures.wait(futures, timeout=remaining)
        for fut in not_done:
            fut.cancel()
        if not_done:
            failed += len(not_done)
            print(f"Planner images: budget ({config.total_budget_s}s) elapsed, "
                  f"{len(not_done)} images abandoned")

        for fut in done:
            key = futures[fut]
            try:
                url = fut.result()
            except Exception:
                url = None
            if not url:
                failed += 1
                continue
            if key == "cover":
                cover_url = url
            else:
                weekly_urls[key] = url

    weekly = [
        {**wr, "imageUrl": weekly_urls[wr["week"]]}
        for wr in weeks
        if wr["week"] in weekly_urls
    ]
    print(f"Planner images: cover={'ok' if cover_url else 'none'}, "
          f"weekly={len(weekly)}/{week_count}, failed={failed}")
    return {"coverImageUrl": cover_url, "weekly": weekly, "failed": failed}
