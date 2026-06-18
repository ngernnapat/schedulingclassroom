#!/usr/bin/env python3
"""Unit tests for todo weekday/date resolution (no OpenAI required)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from todo_generator import (  # noqa: E402
    normalize_todo_dates_in_actions,
    resolve_date_from_text,
)


def test_wednesday_thai_resolves_to_upcoming_date():
    # 2025-06-15 is a Sunday; next Wednesday is 2025-06-18
    resolved = resolve_date_from_text(
        "ส่งของวันพุธ 09:00",
        current_date="2025-06-15T10:00:00+07:00",
        timezone="Asia/Bangkok",
    )
    assert resolved == "2025-06-18"


def test_wednesday_english_resolves_to_upcoming_date():
    resolved = resolve_date_from_text(
        "Send package Wednesday 09:00",
        current_date="2025-06-15T10:00:00+07:00",
        timezone="Asia/Bangkok",
    )
    assert resolved == "2025-06-18"


def test_today_on_same_weekday():
    resolved = resolve_date_from_text(
        "ส่งของวันพุธ 09:00",
        current_date="2025-06-18T10:00:00+07:00",
        timezone="Asia/Bangkok",
    )
    assert resolved == "2025-06-18"


def test_normalize_fills_missing_action_date():
    actions = [
        {
            "action": "create",
            "target_todo_id": "",
            "target_todo_doc_id": "",
            "target_title": "",
            "reason": "",
            "todo": {"title": "ส่งของ", "date": "", "start": "09:00"},
        }
    ]
    normalized = normalize_todo_dates_in_actions(
        actions,
        user_input="ส่งของวันพุธ 09:00",
        current_date="2025-06-15T10:00:00+07:00",
        timezone="Asia/Bangkok",
    )
    assert normalized[0]["todo"]["date"] == "2025-06-18"


def test_normalize_keeps_existing_valid_date():
    actions = [
        {
            "action": "create",
            "todo": {"title": "ส่งของ", "date": "2025-06-20", "start": "09:00"},
        }
    ]
    normalized = normalize_todo_dates_in_actions(
        actions,
        user_input="ส่งของวันพุธ 09:00",
        current_date="2025-06-15T10:00:00+07:00",
        timezone="Asia/Bangkok",
    )
    assert normalized[0]["todo"]["date"] == "2025-06-20"


if __name__ == "__main__":
    test_wednesday_thai_resolves_to_upcoming_date()
    test_wednesday_english_resolves_to_upcoming_date()
    test_today_on_same_weekday()
    test_normalize_fills_missing_action_date()
    test_normalize_keeps_existing_valid_date()
    print("All todo date resolution tests passed.")
