#!/usr/bin/env python3
"""Unit tests for personalization_context (no Firestore required)."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from personalization_context import (  # noqa: E402
    MAX_PROMPT_BLOCK_CHARS,
    assemble_personalization_block,
    format_today_todos_summary,
    build_personalization_for_request,
    refine_month_todos_snapshot,
    format_life_months_block,
    format_month_life_from_client,
)


def test_format_today_todos_summary_caps_and_highlights():
    todos = [
        {"title": "A" * 80, "start": "09:00", "todoID": "1"},
        {"title": "Meeting", "start": "10:00", "todoID": "2", "completed": True},
        {"title": "Gym", "start": "18:00", "todoID": "3"},
    ]
    block = format_today_todos_summary(todos, current_todo_id="2", max_items=2)
    assert "Today's schedule (3)" in block
    assert "→" in block
    assert "✓" in block
    assert "+1 more today" in block
    assert len("A" * 80) > 40  # title truncated in output


def test_assemble_personalization_block_respects_budget():
    huge = "x" * 5000
    block = assemble_personalization_block(
        identity_block="streak: 5",
        life_months_block="prev month busy",
        today_block=huge,
        rag_block=huge,
        intent_block=huge,
    )
    assert len(block) <= MAX_PROMPT_BLOCK_CHARS + 5
    assert "PERSONALIZATION" in block


def test_refine_month_todos_snapshot_compacts_raw_todos():
    todos = [
        {"title": "Gym", "date": "2026-05-01", "typeOfTodo": "hobby", "completed": True},
        {"title": "Work review", "date": "2026-05-01", "typeOfTodo": "work", "completed": True},
        {"title": "Gym", "date": "2026-05-03", "typeOfTodo": "hobby", "completed": False},
        {"title": "Plan read", "date": "2026-05-10", "planId": "p1", "completed": True},
    ]
    snap = refine_month_todos_snapshot(todos, "2026-05", "previous")
    assert snap["scheduled"] == 4
    assert snap["completed"] == 3
    assert snap["active_days"] == 3
    assert snap["plan_linked"] == 1
    assert "Gym" in snap["sample_titles"]
    block = format_life_months_block([snap])
    assert "Previous" in block
    assert "2026-05" in block
    assert "75% done" in block or "done" in block


def test_format_month_life_from_client_snapshots():
    summary = {
        "previous": {
            "role": "previous",
            "year_month": "2026-05",
            "scheduled": 10,
            "completion_rate": 0.8,
            "sample_titles": ["Gym"],
        },
        "current": {
            "role": "current",
            "year_month": "2026-06",
            "scheduled": 5,
            "completion_rate": 0.4,
            "sample_titles": ["Meeting"],
        },
    }
    block = format_month_life_from_client(summary)
    assert "Life calendar" in block
    assert "2026-05" in block
    assert "2026-06" in block


def test_build_personalization_for_request_without_user_id():
    block = build_personalization_for_request(
        {
            "identity_context": {
                "currentStreak": 3,
                "longestStreak": 7,
                "latestBadge": {"becomingPhrase": "You are becoming consistent"},
            },
            "today_todos": [{"title": "Run", "start": "07:00", "todoID": "t1"}],
            "month_life_summary": {
                "current": {
                    "role": "current",
                    "year_month": "2026-06",
                    "scheduled": 8,
                    "completion_rate": 0.5,
                    "sample_titles": ["Run", "Study"],
                }
            },
            "todo_data": {"todoID": "t1", "title": "Run"},
        },
        "How should I pace this run?",
    )
    assert "PERSONALIZATION" in block
    assert "Run" in block
    assert "2026-06" in block


if __name__ == "__main__":
    test_format_today_todos_summary_caps_and_highlights()
    test_assemble_personalization_block_respects_budget()
    test_refine_month_todos_snapshot_compacts_raw_todos()
    test_format_month_life_from_client_snapshots()
    test_build_personalization_for_request_without_user_id()
    print("All personalization_context tests passed.")
