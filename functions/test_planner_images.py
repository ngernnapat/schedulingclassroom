#!/usr/bin/env python3
"""
Tests for planner_images.py (AI cover + weekly theme images).

Runs offline: chatgpt_wrapper.get_default_wrapper is mocked.
Usage: python test_planner_images.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

os.environ["OPENAI_API_KEY"] = "test-key"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planner_images
from planner_images import (
    PlannerImageConfig,
    build_cover_prompt,
    build_weekly_prompt,
    generate_plan_images,
    week_ranges,
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


def mock_wrapper(fail_prompts=()):
    """Wrapper whose generate_image returns b64 keyed by prompt, or raises."""
    wrapper = MagicMock()

    def gen(prompt, *, size, quality, **kw):
        for frag in fail_prompts:
            if frag in prompt:
                raise RuntimeError(f"boom on {frag}")
        return {"b64_json": f"b64:{size}:{quality}", "size": size}

    wrapper.generate_image.side_effect = gen
    return wrapper


def run(total_days=10, weekly_focus=None, fail_prompts=(), config=None, uploads=None):
    uploads = uploads if uploads is not None else []

    def upload_fn(b64):
        uploads.append(b64)
        return f"https://storage.example/{len(uploads)}.png"

    with patch("chatgpt_wrapper.get_default_wrapper", return_value=mock_wrapper(fail_prompts)):
        return generate_plan_images(
            user_id="u1", plan_name="Test Plan", category="exercise",
            language="en", total_days=total_days, overview="get stronger",
            weekly_focus=weekly_focus, upload_fn=upload_fn, config=config,
        )


def test_week_ranges():
    print("\n🧪 Week ranges + caps")
    r = week_ranges(10, 13)
    check("10 days → 2 weeks", [x["week"] for x in r] == [1, 2])
    check("ranges correct", r[0] == {"week": 1, "dayStart": 1, "dayEnd": 7}
          and r[1] == {"week": 2, "dayStart": 8, "dayEnd": 10})
    check("90 days capped at 13", len(week_ranges(90, 13)) == 13)
    check("1 day → 1 week ending day 1", week_ranges(1, 13) == [{"week": 1, "dayStart": 1, "dayEnd": 1}])
    check("out-of-range totalDays clamped", len(week_ranges(500, 13)) == 13)


def test_prompts():
    print("\n🧪 Prompt templates")
    cp = build_cover_prompt("Muay Thai Basics", "exercise", 30, "learn striking fundamentals")
    check("cover mentions plan name + days", "Muay Thai Basics" in cp and "30-day" in cp)
    check("cover carries category palette", "#E85D4C" in cp)
    check("no-text rule present", "no text" in cp.lower())
    wp = build_weekly_prompt("Muay Thai Basics", "travel", 2, 4, "street food tour")
    check("weekly mentions week + focus + travel palette",
          "week 2" in wp and "street food tour" in wp and "#2E9B8F" in wp)
    wp2 = build_weekly_prompt("My Plan", "learning", 1, 2, None)
    check("missing focus falls back to plan name", "My Plan" in wp2)
    check("unknown category falls back to 'other' palette", "#CE4F3C" in build_cover_prompt("P", "banana", 7, ""))


def test_happy_path():
    print("\n🧪 Happy path (10 days → cover + 2 weekly)")
    uploads = []
    out = run(total_days=10, weekly_focus=["form basics", "add intensity"], uploads=uploads)
    check("cover url set", bool(out["coverImageUrl"]))
    check("2 weekly images", len(out["weekly"]) == 2)
    check("weekly carries ranges", out["weekly"][0]["dayStart"] == 1 and out["weekly"][1]["dayEnd"] == 10)
    check("no failures", out["failed"] == 0)
    check("3 uploads happened", len(uploads) == 3)
    check("cover used medium quality, weekly low",
          any(":1024x1536:medium" in u for u in uploads)
          and sum(":1024x1024:low" in u for u in uploads) == 2)


def test_partial_failure():
    print("\n🧪 Per-image soft-fail keeps the rest")
    # Cover prompt contains "Cover artwork"; make only the cover fail
    out = run(total_days=14, fail_prompts=("Cover artwork",))
    check("cover None", out["coverImageUrl"] is None)
    check("weekly still produced", len(out["weekly"]) == 2)
    check("failed counted", out["failed"] == 1)
    # One week fails, others survive
    out = run(total_days=21, fail_prompts=("week 2",))
    check("failed week missing, others present",
          [w["week"] for w in out["weekly"]] == [1, 3] and out["coverImageUrl"])


def test_all_fail():
    print("\n🧪 All images fail → empty result (endpoint returns 500 upstream)")
    out = run(total_days=7, fail_prompts=("Cover artwork", "Illustration"))
    check("cover None + weekly empty", out["coverImageUrl"] is None and out["weekly"] == [])
    check("failed == 2", out["failed"] == 2)


def test_kill_switch():
    print("\n🧪 PLANNER_IMAGES_DISABLED kill switch")
    os.environ["PLANNER_IMAGES_DISABLED"] = "1"
    try:
        called = MagicMock()
        with patch("chatgpt_wrapper.get_default_wrapper", called):
            out = generate_plan_images(
                user_id="u", plan_name="P", category="other", total_days=7,
                upload_fn=lambda b: "x",
            )
        check("disabled flag returned", out.get("disabled") is True and out["weekly"] == [])
        check("no wrapper calls", called.call_count == 0)
    finally:
        del os.environ["PLANNER_IMAGES_DISABLED"]


def test_upload_failure_is_soft():
    print("\n🧪 Upload failure counts as failed, not crash")
    def bad_upload(_b64):
        raise OSError("storage down")
    with patch("chatgpt_wrapper.get_default_wrapper", return_value=mock_wrapper()):
        out = generate_plan_images(
            user_id="u", plan_name="P", category="health", total_days=7,
            upload_fn=bad_upload,
        )
    check("all failed, no exception", out["coverImageUrl"] is None
          and out["weekly"] == [] and out["failed"] == 2)


if __name__ == "__main__":
    print("=" * 60)
    print("planner_images tests")
    print("=" * 60)
    test_week_ranges()
    test_prompts()
    test_happy_path()
    test_partial_failure()
    test_all_fail()
    test_kill_switch()
    test_upload_failure_is_soft()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
