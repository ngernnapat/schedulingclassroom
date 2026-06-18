# todo_generator.py
# Module for extracting structured todo data from natural language input using AI

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Lazy-loaded OpenAI client
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client with lazy initialization."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


@dataclass
class TodoExtractionConfig:
    """Configuration for todo extraction"""
    model: str = "gpt-5.1"
    temperature: float = 0.3
    max_input_length: int = 5000
    default_language: str = "thai"
    default_timezone: str = "Asia/Bangkok"


# Color mapping for different todo types (color and type must match by index)
TODO_COLORS = [
    "#E18683",  # Work
    "#fdae61",  # Hobby
    "#f9c802",  # General
    "#4cbb17",  # Improvement
    "#3a9bdc",  # Event
    "#795695",  # Relax
]

TODO_TYPES = [
    "Work",
    "Hobby",
    "General",
    "Improvement",
    "Event",
    "Relax",
]

# Mapping from type to color
TODO_TYPE_COLORS = dict(zip(TODO_TYPES, TODO_COLORS))

# Single todo item schema (reusable)
TODO_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Clear title with key context. Can be longer if needed to capture the main idea."
        },
        "detail": {
            "type": "string",
            "description": "Extra context from long inputs OR helpful explanation for short inputs."
        },
        "link": {
            "type": "string",
            "description": "URL/link ONLY if explicitly mentioned, empty string if none"
        },
        "location": {
            "type": "string",
            "description": "Location ONLY if explicitly mentioned, empty string if none"
        },
        "everyone" : {
            "type": "boolean",
            "description": "True if explicitly shared with everyone, default false"
        },
        "onlyFollower": {
            "type": "boolean",
            "description": "True by default unless user specifies otherwise"
        },
        "onlyMe": {
            "type": "boolean",
            "description": "True only if user explicitly wants it private/personal, default false"
        },
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format. REQUIRED when user mentions a day (today, tomorrow, weekday, or specific date). Resolve weekday names to the next occurrence on or after the current date."
        },
        "start": {
            "type": "string",
            "description": "Start time in HH:mm 24-hour format, or empty string if not specified"
        },
        "color": {
            "type": "string",
            "enum": ["#E18683", "#fdae61", "#f9c802", "#4cbb17", "#3a9bdc", "#795695"],
            "description": "Color hex code matching the typeOfTodo (Work=#E18683, Hobby=#fdae61, General=#f9c802, Improvement=#4cbb17, Event=#3a9bdc, Relax=#795695)"
        },
        "typeOfTodo": {
            "type": "string",
            "enum": ["Work", "Hobby", "General", "Improvement", "Event", "Relax"],
            "description": "Category of the todo"
        },
        "noSettingTime": {
            "type": "boolean",
            "description": "True if no specific time was mentioned"
        },
        "repeatTodo": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether the todo repeats"
                },
                "type": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly", "yearly", "none"],
                    "description": "Repetition frequency"
                },
                "interval": {
                    "type": "integer",
                    "description": "Interval between repetitions"
                },
                "daysOfWeek": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Days of week for weekly repeat (0=Sun, 6=Sat)"
                }
            },
            "required": ["enabled", "type", "interval", "daysOfWeek"],
            "additionalProperties": False
        },
        "reminder": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Default true. Set false only if user explicitly says no reminder."
                },
                "minutesBefore": {
                    "type": "integer",
                    "description": "Default 15. Options: 0, 5, 10, 15, 30, 60, 1440 (1 day). Change only if user specifies."
                }
            },
            "required": ["enabled", "minutesBefore"],
            "additionalProperties": False
        },
        "suggestedTags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Suggested tags or keywords for the todo"
        }
    },
    "required": ["title", "detail", "link", "everyone", "onlyFollower", "onlyMe", "location", "date", "start", "color", "typeOfTodo", "noSettingTime", "repeatTodo", "reminder", "suggestedTags"],
    "additionalProperties": False
}

# JSON schema for structured todo extraction (supports multiple todos)
TODO_EXTRACTION_SCHEMA = {
    "name": "todo_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of extracted todo items from the user input",
                "items": TODO_ITEM_SCHEMA
            }
        },
        "required": ["todos"],
        "additionalProperties": False
    }
}

# JSON schema for todo action extraction (create/update/delete)
TODO_ACTION_EXTRACTION_SCHEMA = {
    "name": "todo_action_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "description": "List of extracted todo operations",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "update", "delete"]
                        },
                        "target_todo_id": {
                            "type": "string",
                            "description": "Existing todoID for update/delete, empty string for create"
                        },
                        "target_todo_doc_id": {
                            "type": "string",
                            "description": "Existing todoDocID for update/delete, empty string for create"
                        },
                        "target_title": {
                            "type": "string",
                            "description": "Matched existing todo title for update/delete, empty string for create"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short explanation for this action"
                        },
                        "todo": TODO_ITEM_SCHEMA
                    },
                    "required": ["action", "target_todo_id", "target_todo_doc_id", "target_title", "reason", "todo"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["actions"],
        "additionalProperties": False
    }
}


def build_extraction_system_prompt(
    current_date: str,
    timezone: str,
    language: str
) -> str:
    """Build the system prompt for todo extraction."""
    return f"""You are an intelligent assistant that extracts structured todo/task information from natural language input.
Your job is to parse the user's description and extract ALL todo items mentioned. Users may describe one or multiple tasks in a single message.

Current date/time context: {current_date}
User's timezone: {timezone}
Response language: {language}

IMPORTANT: Extract ALL separate tasks/events/todos mentioned in the input. Return them as a list in the 'todos' array.

=== HANDLING AWKWARD OR UNCLEAR INPUTS ===
Users may type in informal, incomplete, or unclear ways. CAREFULLY ANALYZE the message to understand the true intent:

- Typos/misspellings: "meting tmrw" → "Meeting tomorrow", "gim" → "Gym"
- Informal/slang: "gonna hit the gym", "นัดเจอปุ๊บ" → understand casual language
- Incomplete sentences: "dinner 7pm john" → "Dinner with John at 7pm"
- Mixed languages: "meeting พรุ่งนี้ 10am" → understand bilingual input
- Run-on text: "gymthenmeetingthenlaunch" → separate into "Gym", "Meeting", "Lunch"
- Abbreviations: "mtg", "appt", "bday" → "Meeting", "Appointment", "Birthday"
- Thai informal: "กิน 6โมง", "นอน", "ทำงาน" → understand Thai shorthand
- Unclear structure: "tomorrow john lunch project talk" → "Lunch with John to talk about project"

Think step by step:
1. What is the user trying to do/schedule?
2. When (date/time)?
3. Where (location)?
4. With whom?
5. What type of activity is this?

=== HANDLING DATE RANGES AND RECURRING PATTERNS ===
When users specify a DATE RANGE or RECURRING PATTERN within a specific period, you MUST EXPAND it into INDIVIDUAL todos for each occurrence.

IMPORTANT: Only create todos for dates FROM TODAY ONWARDS. Do NOT include past dates.
- If today is Feb 5 and user says "every Monday in February", only include Feb 10, 17, 24 (skip Feb 3 which is past)
- If today is Feb 5 and user says "every day this week", only include Feb 5, 6, 7, 8 (remaining days)

Examples of range/recurring patterns to EXPAND (starting from today):
- "Running every Monday in February" → Create todos only for remaining Mondays from today
- "Gym on MWF next week" → Create 3 separate todos (Mon, Wed, Fri of next week)
- "Meeting every day from Feb 10-14" → Create 5 separate todos (if all dates are in future)
- "วิ่งทุกวันจันทร์ในเดือนกุมภาพันธ์" → Create todos only for remaining Mondays from today
- "เรียนภาษาอังกฤษทุกวันอังคารและพฤหัสบดีในสัปดาห์หน้า" → Create 2 separate todos

For each expanded todo:
- Set the specific 'date' for that occurrence (YYYY-MM-DD) - must be today or future
- Keep the same 'title', 'typeOfTodo', 'color', 'start' time (if specified)
- Set 'repeatTodo.enabled' to false (since we're creating individual instances)

=== HANDLING DIARY NOTES / NON-TODO INPUT ===
If the user's input doesn't seem like a typical task/todo but more like a diary entry, thought, or general note:
- Still create a todo entry as a "diary note"
- Set 'typeOfTodo' to "General" and 'color' to "#f9c802"
- Use the input as the 'title' (summarize if too long)
- Set 'date' to TODAY (current date)
- Set 'start' to CURRENT TIME (from the current_date context, format HH:mm)
- Set 'noSettingTime' to false (since we're recording when it was written)

Examples of diary-style inputs:
- "วันนี้รู้สึกดีมาก" → General todo, today's date, current time, noSettingTime: false
- "Had a great conversation with mom" → General todo, today's date, current time
- "เจอเพื่อนเก่าที่ห้าง" → General todo, today's date, current time
- "Random thought: should learn cooking" → General todo, today's date, current time

=== EXAMPLES ===
Multiple separate tasks:
- "Meeting at 10am and lunch with John at noon" → 2 todos
- "Tomorrow: gym, grocery shopping, and call mom" → 3 todos
- "พรุ่งนี้ประชุม 10 โมง แล้วก็ไปหาหมอตอนบ่าย 2" → 2 todos

=== GUIDELINES FOR EACH TODO ===
- 'title': Include key context to make it clear and meaningful. Don't make it too short if context is important.
- 'detail' handling depends on input length:
  
  FOR LONG INPUTS (user provided lots of context):
  - Put the main idea in 'title'
  - Move extra/overflow context to 'detail', preserve original wording
  * Example: "Meeting with John tomorrow at 2pm to discuss the project budget and quarterly review" 
    → title: "Meeting with John to discuss project budget", detail: "quarterly review"
  
  FOR SHORT INPUTS (user typed briefly):
  - Keep the user's intent in 'title'
  - ADD helpful explanation/context in 'detail' to clarify what the todo is about
  * Example: "ประชุม" → title: "ประชุม", detail: "นัดประชุมงาน"
  * Example: "gym" → title: "Gym", detail: "Workout session"
  * Example: "หมอ" → title: "หมอ", detail: "นัดพบแพทย์"
  * Example: "call mom" → title: "Call mom", detail: "Phone call with mom"

- Only fill 'link' if a URL is explicitly mentioned, otherwise empty string ""
- Only fill 'location' if a location is explicitly mentioned, otherwise empty string ""
- Parse date references (today, tomorrow, next Monday, specific dates) into 'date' field (format: YYYY-MM-DD)
- Parse time references (2pm, 14:00, morning, afternoon) into 'start' field (format: HH:mm in 24-hour)
- If no specific time is mentioned for a todo, set 'noSettingTime' to true
- Determine 'typeOfTodo' and 'color' together (they must match):
  * Work (#E18683) - work tasks, job-related, professional duties
  * Hobby (#fdae61) - hobbies, fun activities, personal interests
  * General (#f9c802) - general tasks, errands, miscellaneous
  * Improvement (#4cbb17) - self-improvement, learning, exercise, health
  * Event (#3a9bdc) - events, meetings, appointments, social gatherings
  * Relax (#795695) - relaxation, rest, leisure, entertainment
- For EXPANDED date range todos: set 'repeatTodo.enabled' to false
- For TRUE recurring todos (no end date specified, e.g., "every Monday"): set 'repeatTodo' with type and interval
- Reminder defaults (unless user explicitly specifies otherwise):
  * 'reminder.enabled': true (default)
  * 'reminder.minutesBefore': 15 (default)
  * Only change if user specifies different timing (e.g., "remind me 1 hour before" → 60, "no reminder" → enabled: false)
- Visibility defaults (unless user explicitly specifies otherwise):
  * 'onlyFollower': true (default)
  * 'everyone': false
  * 'onlyMe': false
  * Only change these if user explicitly mentions sharing publicly (everyone=true) or keeping private (onlyMe=true)

Always respond in valid JSON format with a 'todos' array containing all extracted items."""

_WEEKDAY_HINTS: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (0, ("วันจันทร์", "จันทร์", r"\bmonday\b", r"\bmon\b")),
    (1, ("วันอังคาร", "อังคาร", r"\btuesday\b", r"\btue\b")),
    (2, ("วันพุธ", r"\bพุธ\b", r"\bwednesday\b", r"\bwed\b")),
    (3, ("วันพฤหัส", "พฤหัสบดี", "พฤหัส", r"\bthursday\b", r"\bthu\b")),
    (4, ("วันศุกร์", "ศุกร์", r"\bfriday\b", r"\bfri\b")),
    (5, ("วันเสาร์", "เสาร์", r"\bsaturday\b", r"\bsat\b")),
    (6, ("อาทิตย์", r"\bsunday\b", r"\bsun\b")),
)


def _strip_enriched_user_input(user_input: str) -> str:
    """Remove server-side enrichment blocks before date heuristics."""
    text = str(user_input or "")
    for marker in (
        "\n\n[USER_INTENT_PROFILE]",
        "\n\n[RECENT_CHAT_HISTORY]",
        "\n\n[UNSAVED_DRAFT_ACTIONS",
        "\n\n[REFINEMENT_MODE]",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _parse_reference_date(current_date: str, timezone: str):
    try:
        dt = datetime.fromisoformat(str(current_date).replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now()

    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone or "Asia/Bangkok")
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz)
        else:
            dt = dt.replace(tzinfo=tz)
    except Exception:
        pass
    return dt.date()


def _is_valid_date_string(date_value: str) -> bool:
    if not date_value or not isinstance(date_value, str):
        return False
    try:
        datetime.strptime(date_value[:10], "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _extract_weekday_from_text(text: str) -> Optional[int]:
    haystack = str(text or "")
    lower = haystack.lower()
    for weekday, patterns in _WEEKDAY_HINTS:
        for pattern in patterns:
            target = haystack if pattern.isascii() is False else lower
            if re.search(pattern, target, re.IGNORECASE):
                return weekday
    return None


def _next_weekday_on_or_after(from_date, weekday: int):
    days_ahead = weekday - from_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return from_date + timedelta(days=days_ahead)


def resolve_date_from_text(
    text: str,
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> Optional[str]:
    """Resolve relative day references (today, tomorrow, weekday) to YYYY-MM-DD."""
    raw = _strip_enriched_user_input(text)
    if not raw:
        return None

    ref = _parse_reference_date(current_date, timezone)

    if re.search(r"วันนี้|\btoday\b", raw, re.IGNORECASE):
        return ref.isoformat()
    if re.search(r"มะรืน|วันมะรืน|วันรืน|\bday after tomorrow\b", raw, re.IGNORECASE):
        return (ref + timedelta(days=2)).isoformat()
    if re.search(r"พรุ่งนี้|\btomorrow\b", raw, re.IGNORECASE):
        return (ref + timedelta(days=1)).isoformat()

    weekday = _extract_weekday_from_text(raw)
    if weekday is not None:
        return _next_weekday_on_or_after(ref, weekday).isoformat()
    return None


def resolve_source_target_dates_from_move_request(
    text: str,
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort parse source/target calendar dates from bulk-move phrasing."""
    raw = _strip_enriched_user_input(text)
    if not raw:
        return None, None

    ref = _parse_reference_date(current_date, timezone)

    def _offset_date(days: int) -> str:
        return (ref + timedelta(days=days)).isoformat()

    source = None
    target = None

    if re.search(r"(?:จาก)?\s*พรุ่งนี้|\bfrom tomorrow\b", raw, re.IGNORECASE):
        source = _offset_date(1)
    elif re.search(r"วันนี้|\btoday\b", raw, re.IGNORECASE):
        source = _offset_date(0)

    if re.search(r"ไป(?:ที่)?\s*มะรืน|ไป(?:ที่)?\s*วันมะรืน|\bto day after tomorrow\b", raw, re.IGNORECASE):
        target = _offset_date(2)
    elif re.search(r"ไป(?:ที่)?\s*พรุ่งนี้|\bto tomorrow\b", raw, re.IGNORECASE):
        target = _offset_date(1)

    if source and not target:
        target = resolve_date_from_text(raw, current_date, timezone)
    if target and not source:
        # e.g. "move tomorrow's work" — source is tomorrow when paired with target later
        if re.search(r"พรุ่งนี้|\btomorrow\b", raw, re.IGNORECASE):
            source = _offset_date(1)

    return source, target


BULK_ALL_PATTERN = re.compile(
    r"(?:ให้หมด|ทั้งหมด|ทุก(?:อย่าง|รายการ)|all\b|everything|every\s+(?:item|task))",
    re.IGNORECASE,
)
MOVE_REQUEST_PATTERN = re.compile(
    r"(?:ย้าย|เลื่อน|move|reschedule|shift|postpone)",
    re.IGNORECASE,
)
WORK_SCOPE_PATTERN = re.compile(
    r"(?:งาน\b|work(?:\s+items?)?|tasks?\b|alPlanner|Day\s*\d+)",
    re.IGNORECASE,
)
LEISURE_TITLE_PATTERN = re.compile(
    r"(?:ว่ายน้ำ|swim|พักยาว|ชิล|relax|rest block|brain reset)",
    re.IGNORECASE,
)


def _todo_date_key(todo: Dict[str, Any]) -> str:
    return str((todo or {}).get("date") or "")[:10]


def _should_include_todo_for_bulk_move(todo: Dict[str, Any], work_only: bool) -> bool:
    title = str((todo or {}).get("title") or "")
    if not work_only:
        return True
    if LEISURE_TITLE_PATTERN.search(title):
        return False
    if WORK_SCOPE_PATTERN.search(title):
        return True
    todo_type = str((todo or {}).get("typeOfTodo") or "").lower()
    if todo_type in ("general", "work", "learning", "exercise", "planner"):
        return True
    return not LEISURE_TITLE_PATTERN.search(title)


def expand_bulk_day_move_actions(
    actions: List[Dict[str, Any]],
    user_input: str,
    existing_todos: Optional[List[Dict[str, Any]]],
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> List[Dict[str, Any]]:
    """
  When the user asks to move *all* items from one day to another, expand to one
  update action per matching calendar row instead of a single vague update.
    """
    raw = _strip_enriched_user_input(user_input)
    if not raw or not MOVE_REQUEST_PATTERN.search(raw) or not BULK_ALL_PATTERN.search(raw):
        return actions

    source_date, target_date = resolve_source_target_dates_from_move_request(
        raw, current_date, timezone
    )
    if not source_date or not target_date or source_date == target_date:
        return actions

    work_only = bool(WORK_SCOPE_PATTERN.search(raw))
    pool = [t for t in (existing_todos or []) if isinstance(t, dict)]
    candidates = [
        t
        for t in pool
        if _todo_date_key(t) == source_date and _should_include_todo_for_bulk_move(t, work_only)
    ]
    if len(candidates) < 2:
        return actions

    expanded: List[Dict[str, Any]] = []
    for todo in candidates:
        updated = dict(todo)
        updated["date"] = target_date
        expanded.append(
            {
                "action": "update",
                "target_todo_id": str(todo.get("todoID") or ""),
                "target_todo_doc_id": str(todo.get("todoDocID") or ""),
                "target_title": str(todo.get("title") or ""),
                "reason": f"Bulk move {source_date} → {target_date}",
                "todo": updated,
            }
        )

    create_actions = [
        a for a in (actions or []) if str((a or {}).get("action", "")).lower() == "create"
    ]
    logger.info(
        "Expanded bulk day move: %s items from %s to %s (work_only=%s)",
        len(expanded),
        source_date,
        target_date,
        work_only,
    )
    return expanded + create_actions


DELETE_REQUEST_PATTERN = re.compile(
    r"(?:ลบ|ยกเลิก|delete|remove|cancel|ตัดออก|เอาออก|clear(?:\s+out)?)",
    re.IGNORECASE,
)


def resolve_delete_source_date(
    text: str,
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> Optional[str]:
    """Resolve which calendar day the user wants to clear/delete from."""
    raw = _strip_enriched_user_input(text)
    if not raw:
        return None

    ref = _parse_reference_date(current_date, timezone)
    if re.search(r"(?:จาก)?\s*พรุ่งนี้|\bfrom tomorrow\b", raw, re.IGNORECASE):
        return (ref + timedelta(days=1)).isoformat()
    if re.search(r"มะรืน|วันมะรืน|\bday after tomorrow\b", raw, re.IGNORECASE):
        return (ref + timedelta(days=2)).isoformat()
    if re.search(r"วันนี้|\btoday\b", raw, re.IGNORECASE):
        return ref.isoformat()

    weekday = _extract_weekday_from_text(raw)
    if weekday is not None:
        return _next_weekday_on_or_after(ref, weekday).isoformat()
    return None


def expand_bulk_day_delete_actions(
    actions: List[Dict[str, Any]],
    user_input: str,
    existing_todos: Optional[List[Dict[str, Any]]],
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> List[Dict[str, Any]]:
    """When user asks to delete all items on a day, emit one delete per match."""
    raw = _strip_enriched_user_input(user_input)
    if not raw or not DELETE_REQUEST_PATTERN.search(raw) or not BULK_ALL_PATTERN.search(raw):
        return actions

    source_date = resolve_delete_source_date(raw, current_date, timezone)
    if not source_date:
        return actions

    work_only = bool(WORK_SCOPE_PATTERN.search(raw))
    pool = [t for t in (existing_todos or []) if isinstance(t, dict)]
    candidates = [
        t
        for t in pool
        if _todo_date_key(t) == source_date and _should_include_todo_for_bulk_move(t, work_only)
    ]
    if not candidates:
        return actions

    existing_deletes = [
        a for a in (actions or []) if str((a or {}).get("action", "")).lower() == "delete"
    ]
    if len(existing_deletes) >= len(candidates):
        return actions

    expanded: List[Dict[str, Any]] = []
    for todo in candidates:
        expanded.append(
            {
                "action": "delete",
                "target_todo_id": str(todo.get("todoID") or ""),
                "target_todo_doc_id": str(todo.get("todoDocID") or ""),
                "target_title": str(todo.get("title") or ""),
                "reason": f"Bulk delete on {source_date}",
                "todo": dict(todo),
            }
        )

    logger.info(
        "Expanded bulk day delete: %s items on %s (work_only=%s)",
        len(expanded),
        source_date,
        work_only,
    )
    return expanded


def normalize_todo_dates_in_actions(
    actions: List[Dict[str, Any]],
    user_input: str,
    current_date: str,
    timezone: str = "Asia/Bangkok",
) -> List[Dict[str, Any]]:
    """Fill missing todo.date values using weekday/relative-day hints in user text."""
    if not isinstance(actions, list):
        return actions

    fallback_date = resolve_date_from_text(user_input, current_date, timezone)

    for action in actions:
        if not isinstance(action, dict):
            continue
        todo = action.get("todo")
        if not isinstance(todo, dict):
            continue
        date_val = str(todo.get("date", "")).strip()
        if _is_valid_date_string(date_val):
            continue
        if fallback_date:
            todo["date"] = fallback_date
    return actions


OPTIMIZE_REQUEST_PATTERN = re.compile(
    r"(?:ปรับตาราง|optimize|rebalance|จัด(?:ตาราง)?ใหม่|เบาลง|กระจาย|spread|balance|lighter|สมดุล|ทำได้จริง|แน่น|packed|too busy|ลด(?:ภาระ|งาน))",
    re.IGNORECASE,
)

COACH_NOTE_PATTERNS = (
    re.compile(r"ต่อไปถ้าจะทำ", re.IGNORECASE),
    re.compile(r"ขอให้เล็ก", re.IGNORECASE),
    re.compile(r"เล็กมาก", re.IGNORECASE),
    re.compile(r"ทำเล็ก\s*ๆ", re.IGNORECASE),
    re.compile(r"next time if you (?:want to )?do more", re.IGNORECASE),
    re.compile(r"keep it (?:very )?small", re.IGNORECASE),
    re.compile(r"small step", re.IGNORECASE),
    re.compile(r"(?:coach|reflection|journal|บันทึก|สะท้อน)", re.IGNORECASE),
)

AFTERNOON_SLOT_CANDIDATES = ("14:00", "15:00", "16:00", "17:00", "13:30", "13:00")
LATE_NIGHT_START_HOUR = 21
EARLY_MORNING_END_HOUR = 7


def _parse_hhmm(value: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d{1,2}):(\d{2})$", str(value or "").strip())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _minutes_from_midnight(value: str) -> Optional[int]:
    parsed = _parse_hhmm(value)
    if not parsed:
        return None
    hour, minute = parsed
    return hour * 60 + minute


def _is_late_night_time(start: str) -> bool:
    parsed = _parse_hhmm(start)
    return parsed is not None and parsed[0] >= LATE_NIGHT_START_HOUR


def _is_early_morning_time(start: str) -> bool:
    parsed = _parse_hhmm(start)
    return parsed is not None and parsed[0] < EARLY_MORNING_END_HOUR


def _is_schedule_optimize_request(user_input: str) -> bool:
    return bool(OPTIMIZE_REQUEST_PATTERN.search(_strip_enriched_user_input(user_input)))


def _is_coach_style_note(todo: Dict[str, Any]) -> bool:
    combined = f"{todo.get('title', '')} {todo.get('detail', '')}".strip()
    if not combined:
        return False
    return any(pattern.search(combined) for pattern in COACH_NOTE_PATTERNS)


def _clear_todo_time(todo: Dict[str, Any]) -> None:
    todo["start"] = ""
    todo["noSettingTime"] = True


def _occupied_minutes_for_date(
    existing_todos: List[Dict[str, Any]],
    date_str: str,
) -> List[int]:
    occupied: List[int] = []
    target_date = date_str[:10]
    for todo in existing_todos:
        if not isinstance(todo, dict):
            continue
        if str(todo.get("date", ""))[:10] != target_date:
            continue
        minutes = _minutes_from_midnight(str(todo.get("start", "")).strip())
        if minutes is not None:
            occupied.append(minutes)
    return occupied


def _find_afternoon_slot(
    date_str: str,
    existing_todos: List[Dict[str, Any]],
) -> Optional[str]:
    occupied = _occupied_minutes_for_date(existing_todos, date_str)
    for slot in AFTERNOON_SLOT_CANDIDATES:
        slot_minutes = _minutes_from_midnight(slot)
        if slot_minutes is None:
            continue
        if any(abs(slot_minutes - busy) < 60 for busy in occupied):
            continue
        return slot
    return None


def sanitize_schedule_optimization_actions(
    actions: List[Dict[str, Any]],
    user_input: str,
    existing_todos: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Guardrails for schedule rebalance/optimize requests:
    - Coach-style notes become flexible (no fixed time).
    - Late-night / very-early slots are moved to afternoon gaps or cleared.
    """
    if not isinstance(actions, list) or not actions:
        return actions
    if not _is_schedule_optimize_request(user_input):
        return actions

    existing = existing_todos if isinstance(existing_todos, list) else []

    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("action", "")).lower() not in ("create", "update"):
            continue
        todo = action.get("todo")
        if not isinstance(todo, dict):
            continue

        if _is_coach_style_note(todo):
            _clear_todo_time(todo)
            continue

        start = str(todo.get("start", "")).strip()
        if not start:
            continue
        if not (_is_late_night_time(start) or _is_early_morning_time(start)):
            continue

        date_str = str(todo.get("date", "")).strip()
        afternoon_slot = (
            _find_afternoon_slot(date_str, existing)
            if _is_valid_date_string(date_str)
            else None
        )
        if afternoon_slot:
            todo["start"] = afternoon_slot
            todo["noSettingTime"] = False
        else:
            _clear_todo_time(todo)

    return actions


def build_action_extraction_system_prompt(
    current_date: str,
    timezone: str,
    language: str,
    existing_todos: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build the system prompt for todo action extraction."""
    existing_context = json.dumps(existing_todos or [], ensure_ascii=False)
    return f"""You are an intelligent assistant that converts user requests into todo operations.
You must decide which operations are needed: create, update, or delete.

Current date/time context: {current_date}
User's timezone: {timezone}
Response language: {language}

Existing todos for matching (may be empty):
{existing_context}

Rules:
1) If user asks to add/new/create something -> action=create.
2) If user asks to modify/edit/change/reschedule an existing todo -> action=update.
3) If user asks to remove/delete/cancel an existing todo -> action=delete.
4) For update/delete, match the target from existing todos and fill:
   - target_todo_id
   - target_todo_doc_id
   - target_title
5) If no confident match exists for update/delete, still produce the action but leave target ids empty strings.
6) Always fill 'todo' with a valid todo object:
   - For create/update: todo is the new desired state.
   - For delete: copy the matched todo content if available, otherwise use a minimal reasonable todo object.
7) Return all operations in execution order.
8) BULK MOVE: when user says all/ทั้งหมด/ให้หมด/everything on a source day, emit one update per matching existing todo on that day (not a single combined action).
9) BULK DELETE: when user says delete/remove all/ลบ...ทั้งหมด/ให้หมด on a day, emit one delete action per matching existing todo on that day.
10) REFINEMENT: when UNSAVED_DRAFT_ACTIONS are present, merge the user's latest instruction with those drafts and return the full final action set.

DATE AND TIME (CRITICAL):
- Parse date references into todo.date as YYYY-MM-DD. Never leave date empty when the user mentions a day.
- Relative days: today/วันนี้ -> current date; tomorrow/พรุ่งนี้ -> next day; day after tomorrow/มะรืน/วันมะรืน -> +2 days.
- Weekdays (Monday/วันจันทร์, Wednesday/วันพุธ, etc.): resolve to the next occurrence on or after the current date (include today if it matches).
- Specific dates: convert to YYYY-MM-DD in the user's timezone.
- Parse time references (09:00, 2pm, บ่าย 2) into todo.start as HH:mm (24-hour).
- If no specific time is mentioned, set noSettingTime=true and leave start as empty string.
- Only create todos for today or future dates, never past dates.

SCHEDULE OPTIMIZATION (when user asks to rebalance, optimize, lighten, or make the schedule realistic):
- Prefer afternoon gaps (13:00–18:00) when moving items to ease a packed morning.
- Never schedule after 20:30 unless the user explicitly asks for a late-night slot.
- Protect sleep/recovery: do NOT move items to 21:00+ when rebalancing a busy day.
- Coach-style notes, reflections, or meta-guidance (e.g. "keep additions small", "ต่อไปถ้าจะทำอะไรเพิ่ม ขอให้เล็กมากๆ") are NOT calendar blocks — set noSettingTime=true and leave start empty.
- When easing overload, move real tasks/events — not diary/coach reminders.

Output JSON ONLY with an 'actions' array."""


class TodoExtractor:
    """Class for extracting structured todo data from natural language input."""
    
    def __init__(self, config: Optional[TodoExtractionConfig] = None):
        """Initialize the TodoExtractor with optional configuration."""
        self.config = config or TodoExtractionConfig()
    
    def extract_todo_data(
        self,
        user_input: str,
        language: Optional[str] = None,
        current_date: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract structured todo data from natural language input.
        Supports extracting multiple todos from a single input.
        
        Args:
            user_input: Natural language description of one or more todos
            language: Language for processing (default: from config)
            current_date: Current date in ISO format for context
            timezone: User's timezone
            
        Returns:
            List of dictionaries, each containing structured todo data
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If AI extraction fails
        """
        # Apply defaults
        language = language or self.config.default_language
        current_date = current_date or datetime.now().isoformat()
        timezone = timezone or self.config.default_timezone
        
        # Validate input
        if not user_input or not isinstance(user_input, str):
            raise ValueError("user_input must be a non-empty string")
        
        if len(user_input) > self.config.max_input_length:
            raise ValueError(f"user_input must be less than {self.config.max_input_length} characters")
        
        logger.info(f"Extracting todos from input: {user_input[:100]}... in language: {language}")
        
        # Build prompts
        system_prompt = build_extraction_system_prompt(current_date, timezone, language)
        user_message = f"Please extract all todo items from this text:\n\n{user_input}"
        
        # Make API call
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": TODO_EXTRACTION_SCHEMA
            }
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Empty response from AI model")
        
        # Parse response
        raw_response = response.choices[0].message.content
        result = json.loads(raw_response)
        
        # Extract the todos list
        todos = result.get('todos', [])
        
        logger.info(f"Successfully extracted {len(todos)} todo(s)")
        
        return todos

    def extract_todo_actions(
        self,
        user_input: str,
        language: Optional[str] = None,
        current_date: Optional[str] = None,
        timezone: Optional[str] = None,
        existing_todos: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured todo actions (create/update/delete) from natural language.
        """
        language = language or self.config.default_language
        current_date = current_date or datetime.now().isoformat()
        timezone = timezone or self.config.default_timezone
        existing_todos = existing_todos or []

        if not user_input or not isinstance(user_input, str):
            raise ValueError("user_input must be a non-empty string")

        if len(user_input) > self.config.max_input_length:
            raise ValueError(f"user_input must be less than {self.config.max_input_length} characters")

        logger.info("Extracting todo actions from input in language: %s", language)

        system_prompt = build_action_extraction_system_prompt(
            current_date=current_date,
            timezone=timezone,
            language=language,
            existing_todos=existing_todos
        )
        user_message = f"Please convert this request into todo actions:\n\n{user_input}"

        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": TODO_ACTION_EXTRACTION_SCHEMA
            }
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Empty response from AI model")

        raw_response = response.choices[0].message.content
        result = json.loads(raw_response)
        actions = normalize_todo_dates_in_actions(
            result.get("actions", []),
            user_input=user_input,
            current_date=current_date,
            timezone=timezone,
        )
        actions = sanitize_schedule_optimization_actions(
            actions,
            user_input=user_input,
            existing_todos=existing_todos,
        )
        actions = expand_bulk_day_move_actions(
            actions,
            user_input=user_input,
            existing_todos=existing_todos,
            current_date=current_date,
            timezone=timezone,
        )
        actions = expand_bulk_day_delete_actions(
            actions,
            user_input=user_input,
            existing_todos=existing_todos,
            current_date=current_date,
            timezone=timezone,
        )

        # Backward compatible create-list for existing callers.
        create_like_todos = []
        for action in actions:
            todo = action.get("todo", {})
            if action.get("action") in ("create", "update") and isinstance(todo, dict):
                create_like_todos.append(todo)

        logger.info("Successfully extracted %s todo action(s)", len(actions))
        return {
            "actions": actions,
            "todos": create_like_todos
        }


# Module-level function for convenience
_default_extractor = None

def get_default_extractor() -> TodoExtractor:
    """Get or create the default TodoExtractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = TodoExtractor()
    return _default_extractor


def extract_todo_from_text(
    user_input: str,
    language: str = "thai",
    current_date: Optional[str] = None,
    timezone: str = "Asia/Bangkok"
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract todo data from text.
    Supports extracting multiple todos from a single input.
    
    Args:
        user_input: Natural language description of one or more todos
        language: Language for processing
        current_date: Current date in ISO format for context
        timezone: User's timezone
        
    Returns:
        List of dictionaries, each containing structured todo data
        
    Example:
        >>> todos = extract_todo_from_text("Meeting at 10am and lunch with John at noon")
        >>> len(todos)  # Returns 2
        2
    """
    extractor = get_default_extractor()
    return extractor.extract_todo_data(
        user_input=user_input,
        language=language,
        current_date=current_date,
        timezone=timezone
    )


def extract_todo_actions_from_text(
    user_input: str,
    language: str = "thai",
    current_date: Optional[str] = None,
    timezone: str = "Asia/Bangkok",
    existing_todos: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract todo actions (create/update/delete) from text.
    """
    extractor = get_default_extractor()
    return extractor.extract_todo_actions(
        user_input=user_input,
        language=language,
        current_date=current_date,
        timezone=timezone,
        existing_todos=existing_todos
    )
