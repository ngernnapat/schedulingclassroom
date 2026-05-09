# todo_generator.py
# Module for extracting structured todo data from natural language input using AI

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
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
            "description": "Date in YYYY-MM-DD format, or empty string if not specified"
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
        actions = result.get("actions", [])

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
