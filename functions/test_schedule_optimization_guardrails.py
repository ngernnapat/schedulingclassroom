#!/usr/bin/env python3
"""Unit tests for schedule optimization guardrails (no OpenAI required)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from todo_generator import (  # noqa: E402
    sanitize_schedule_optimization_actions,
    _is_coach_style_note,
    _is_schedule_optimize_request,
)


def _update_action(title, start, date="2026-06-16"):
    return {
        "action": "update",
        "target_todo_id": "abc",
        "todo": {
            "title": title,
            "detail": "",
            "date": date,
            "start": start,
            "noSettingTime": False,
        },
    }


def test_detect_optimize_request_thai():
    assert _is_schedule_optimize_request(
        "ช่วยปรับตารางให้สมดุลและทำได้จริงขึ้น จากคำแนะนำนี้"
    )


def test_coach_note_detection():
    assert _is_coach_style_note(
        {"title": "ต่อไปถ้าจะทำอะไรเพิ่ม ขอให้เล็กมากๆ พอ", "detail": ""}
    )


def test_coach_note_cleared_on_optimize():
    actions = [
        _update_action("ต่อไปถ้าจะทำอะไรเพิ่ม ขอให้เล็กมากๆ พอ", "22:00")
    ]
    sanitized = sanitize_schedule_optimization_actions(
        actions,
        user_input="ช่วยปรับตารางให้สมดุลและทำได้จริงขึ้น",
        existing_todos=[],
    )
    todo = sanitized[0]["todo"]
    assert todo["start"] == ""
    assert todo["noSettingTime"] is True


def test_late_night_moved_to_afternoon_gap():
    actions = [_update_action("Day 10 review", "22:00")]
    existing = [
        {"title": "Meeting", "date": "2026-06-16", "start": "14:00"},
    ]
    sanitized = sanitize_schedule_optimization_actions(
        actions,
        user_input="Optimize my schedule to be more balanced",
        existing_todos=existing,
    )
    todo = sanitized[0]["todo"]
    assert todo["start"] == "15:00"
    assert todo["noSettingTime"] is False


def test_late_night_cleared_when_no_afternoon_gap():
    actions = [_update_action("Day 10 review", "22:00")]
    existing = [
        {"title": "A", "date": "2026-06-16", "start": "13:00"},
        {"title": "B", "date": "2026-06-16", "start": "13:30"},
        {"title": "C", "date": "2026-06-16", "start": "14:00"},
        {"title": "D", "date": "2026-06-16", "start": "15:00"},
        {"title": "E", "date": "2026-06-16", "start": "16:00"},
        {"title": "F", "date": "2026-06-16", "start": "17:00"},
    ]
    sanitized = sanitize_schedule_optimization_actions(
        actions,
        user_input="rebalance schedule",
        existing_todos=existing,
    )
    todo = sanitized[0]["todo"]
    assert todo["start"] == ""
    assert todo["noSettingTime"] is True


def test_non_optimize_request_unchanged():
    actions = [_update_action("Gym", "22:00")]
    sanitized = sanitize_schedule_optimization_actions(
        actions,
        user_input="Move gym to 10pm",
        existing_todos=[],
    )
    assert sanitized[0]["todo"]["start"] == "22:00"


if __name__ == "__main__":
    test_detect_optimize_request_thai()
    test_coach_note_detection()
    test_coach_note_cleared_on_optimize()
    test_late_night_moved_to_afternoon_gap()
    test_late_night_cleared_when_no_afternoon_gap()
    test_non_optimize_request_unchanged()
    print("All schedule optimization guardrail tests passed.")
