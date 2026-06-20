# app/planner_utils.py

import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from chatgpt_wrapper import chat_with_gpt, ChatGPTWrapper, get_default_wrapper, RateLimitExceededError
from rune_llm_catalog import normalize_earned_runes_for_llm

# Configure logging
logger = logging.getLogger(__name__)


def _format_month_context(month_context: Optional[Dict[str, Any]] = None) -> str:
    """Format previous/current/next month data for injection into prompts. Improves RAG-style text generation."""
    if not month_context or not isinstance(month_context, dict):
        return ""
    parts = []
    for key, label in [("previous", "Previous month"), ("current", "Current month"), ("next", "Next month")]:
        val = month_context.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            parts.append(f"{label}: {val.strip()}")
        elif isinstance(val, list):
            lines = []
            for item in val[:15]:
                if isinstance(item, str):
                    lines.append(f"  • {item}")
                elif isinstance(item, dict):
                    title = item.get("title") or item.get("name") or str(item)[:80]
                    lines.append(f"  • {title}")
            if lines:
                parts.append(f"{label}:\n" + "\n".join(lines))
    if not parts:
        return ""
    return "Month context (use to improve relevance and continuity):\n" + "\n\n".join(parts)


def _task_start_sort_key(task: Dict[str, Any]) -> tuple:
    start = str(task.get("start") or "").strip()
    if not start:
        return (1, 9999)
    parts = start.split(":")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1][:2].isdigit():
        return (0, int(parts[0]) * 60 + int(parts[1][:2]))
    return (0, start)


def _sorted_today_tasks(
    today_todo_list_data: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    tasks = [
        task for task in (today_todo_list_data or [])
        if isinstance(task, dict)
    ]
    return sorted(tasks, key=_task_start_sort_key)


def _format_today_tasks_for_notification(
    today_todo_list_data: Optional[List[Dict[str, Any]]],
    language: str = "thai",
) -> str:
    """Build a full, explicit bullet list of today's tasks for morning push notifications."""
    tasks = _sorted_today_tasks(today_todo_list_data)
    if not tasks:
        return ""

    lang = (language or "thai").strip().lower()
    is_thai = lang in ("thai", "th", "ไทย")
    header = (
        f"📋 รายการวันนี้ ({len(tasks)}):"
        if is_thai
        else f"📋 Today's list ({len(tasks)}):"
    )

    lines = []
    for task in tasks:
        title = str(task.get("title") or "Task").strip()[:50]
        start = str(task.get("start") or "").strip()
        done = bool(task.get("completed"))
        if done:
            if start:
                line = f"• ✓ {start} {title}"
            else:
                line = f"• ✓ {title}"
        elif start:
            line = f"• {start} {title}"
        else:
            line = f"• {title}"
        lines.append(line)

    return header + "\n" + "\n".join(lines)


def _format_identity_context(
    identity_context: Optional[Dict[str, Any]] = None,
    last_week_completion_rate: Optional[float] = None,
) -> str:
    """
    Format the user's identity / behavior signals into a prompt block.

    Per docs/ORCHESTRATION.md "Relevance second" priority — coaching should
    reference the user's actual streaks, badges, and recent completion rate
    so output is grounded in real behavior rather than generic. The block
    intentionally uses second-person framing so the LLM can echo identity
    language ("you're becoming consistent") rather than third-person stats.

    Returns an empty string if no usable context is supplied.
    """
    if (not identity_context or not isinstance(identity_context, dict)) and last_week_completion_rate is None:
        return ""

    ctx = identity_context if isinstance(identity_context, dict) else {}
    lines = []

    current = ctx.get("currentStreak")
    longest = ctx.get("longestStreak")
    if isinstance(current, (int, float)) and current > 0:
        lines.append(f"Recent return streak (background only, do not push): {int(current)} days")
    if isinstance(longest, (int, float)) and longest > 0:
        lines.append(f"Longest return streak (background only): {int(longest)} days")

    last_done = ctx.get("lastCompletionDate")
    if isinstance(last_done, str) and last_done.strip():
        lines.append(f"Last completion date: {last_done}")

    dow = ctx.get("dayOfWeek")
    if isinstance(dow, list) and len(dow) == 7 and any(isinstance(v, (int, float)) and v > 0 for v in dow):
        names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        # Highlight the user's strongest day so the LLM can build around it.
        max_idx = max(range(7), key=lambda i: (dow[i] if isinstance(dow[i], (int, float)) else 0))
        if dow[max_idx]:
            lines.append(f"Most consistent weekday: {names[max_idx]}")

    badge = ctx.get("latestBadge")
    if isinstance(badge, dict):
        phrase = badge.get("becomingPhrase")
        title = badge.get("title")
        if phrase or title:
            label = title or "milestone"
            if phrase:
                lines.append(f"Most recent badge: {label} — \"{phrase}\"")
            else:
                lines.append(f"Most recent badge: {label}")

    if isinstance(last_week_completion_rate, (int, float)):
        rate_pct = round(float(last_week_completion_rate) * 100)
        lines.append(
            f"Last week's task-touch rate: {rate_pct}% (context only — never guilt-trip)"
        )

    if not lines:
        return ""

    header = (
        "User behavior signals (ground coaching in goals, return rhythm, and how "
        "they likely feel — not streak pressure. Reference the most recent badge "
        "identity phrase when relevant. If load was low, suggest lighter sustainable "
        "next steps; if strong, reinforce alignment with what matters to them):"
    )
    return header + "\n" + "\n".join(f"- {ln}" for ln in lines)


# Analysis aims when using RAG todo context: what the model should do with the user's todo list
RAG_TODO_ANALYSIS_AIMS = """
When analyzing the user's todo list (from context above), always address these aims. Be practical, not theoretical—tie every point to their actual tasks and dates.

1. **Prevent overload**: Flag days or weeks with too many tasks, back-to-back meetings with no buffer, or unrealistic density. Suggest what to move, drop, or defer.
2. **Protect deep work time**: Identify focus-needed tasks (e.g. report, coding, study) and suggest when to block uninterrupted time; warn if they're squeezed between meetings.
3. **Maintain goal momentum**: Spot recurring or goal-related items (e.g. gym, learning, project milestones). Encourage consistency and suggest how to keep them visible and achievable.
4. **Be practical**: Give 2–4 concrete, actionable suggestions only. No generic advice—reference specific titles, dates, or patterns from their list. Use their language.
5. **Protect recovery**: When suggesting moves, avoid dumping items into late night (after 21:00). Prefer afternoon gaps or making flexible notes unscheduled. Coach reflections and meta-reminders are not calendar blocks.
"""


class PlannerType(Enum):
    """Types of planner operations"""
    SUMMARIZE = "summarize"
    MOTIVATE = "motivate"
    TRACK_PROGRESS = "track_progress"
    RESPOND = "respond"
    MOOD_BOOST = "mood_boost"

@dataclass
class PlannerConfig:
    """Configuration for planner operations"""
    max_completion_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    enable_emojis: bool = True
    enable_motivation: bool = True
    language: str = "thai"

class PlannerValidator:
    """Input validation for planner operations"""
    
    @staticmethod
    def validate_planner_data(data: Dict[str, Any]) -> bool:
        """Validate planner data structure"""
        if not isinstance(data, dict):
            raise ValueError("Planner data must be a dictionary")
        
        if not data:
            raise ValueError("Planner data cannot be empty")
        
        return True
    
    @staticmethod
    def validate_language(language: str) -> str:
        """Validate and normalize language code"""
        if not language or not isinstance(language, str):
            return "thai"
        
        # Normalize language codes
        language_map = {
            "thai": "thai", "th": "thai", "thailand": "thai",
            "english": "english", "en": "english", "eng": "english",
            "chinese": "chinese", "zh": "chinese", "cn": "chinese",
            "japanese": "japanese", "ja": "japanese", "jp": "japanese",
            "korean": "korean", "ko": "korean", "kr": "korean"
        }
        
        normalized = language_map.get(language.lower(), language.lower())
        return normalized
    
    @staticmethod
    def validate_user_input(user_input: str) -> str:
        """Validate user input"""
        if not user_input or not isinstance(user_input, str):
            raise ValueError("User input must be a non-empty string")
        
        # Check for potential injection or inappropriate content
        suspicious_patterns = ['<script>', 'javascript:', 'data:text/html', 'eval(']
        for pattern in suspicious_patterns:
            if pattern.lower() in user_input.lower():
                logger.warning(f"Potential injection attempt in user input: {pattern}")
                raise ValueError("Invalid input detected")
        
        return user_input.strip()

class PromptBuilder:
    """Build optimized prompts for different planner operations"""
    
    @staticmethod
    def build_summarize_prompt(planner_data: Dict[str, Any], language: str) -> tuple[str, str]:
        """Build prompt for plan summarization"""
        system_prompt = (
            "You are Evo, a friendly and inspiring AI lifestyle coach helping users grow and evolve. "
            "Your responses should be compact, friendly, and motivational. "
            "Use relevant emojis to make responses more engaging and keep them concise."
        )
        
        user_prompt = (
            f"Please summarize this planner data in {language} language: {json.dumps(planner_data, ensure_ascii=False)}. "
            "Make it compact, friendly, and motivational with relevant emojis. Keep the response short and engaging."
        )
        
        return system_prompt, user_prompt
    
    @staticmethod
    def build_motivate_prompt(summary: str) -> tuple[str, str]:
        """Build prompt for user motivation"""
        system_prompt = (
            "You are Evo, a supportive AI assistant/secretary/coach who encourages users warmly and humanly. "
            "Provide motivational advice that is compact, friendly, and energetic."
        )
        
        user_prompt = (
            f"Based on this planner summary: {summary}, "
            "give motivational advice in a compact, friendly, and energetic tone with relevant emojis."
        )
        
        return system_prompt, user_prompt
    
    @staticmethod
    def build_progress_prompt(user_update: str, summary: str, todo_data: Dict[str, Any]) -> tuple[str, str]:
        """Build prompt for progress tracking"""
        system_prompt = (
            "You are Evo, a positive AI assistant/secretary/coach that tracks progress and encourages improvement. "
            "Provide constructive feedback that is specific, actionable, and encouraging."
        )
        
        # Format todo data for clarity
        todo_info = "\n".join([f"• {key}: {value}" for key, value in todo_data.items()])
        
        user_prompt = (
            f"User update: {user_update}\n\n"
            f"Todo details:\n{todo_info}\n\n"
            f"Planner summary: {summary}\n\n"
            "Give positive, constructive feedback in a friendly and motivational tone. "
            "Make it compact, specific, actionable, and encouraging with relevant emojis."
        )
        
        return system_prompt, user_prompt
    
    @staticmethod
    def build_response_prompt(user_input: str, summary: str) -> tuple[str, str]:
        """Build prompt for user input response"""
        system_prompt = (
            "You are Evo, a personal AI assistant/secretary/coach helping the user with their lifestyle planner. "
            "Respond naturally, supportively, and with a motivational tone."
        )
        
        user_prompt = (
            f"User says: {user_input}\n\n"
            f"Planner context: {summary}\n\n"
            "Respond naturally, compactly, and supportively with motivational tone and emojis."
        )
        
        return system_prompt, user_prompt
    
    @staticmethod
    def build_mood_boost_prompt(summary: str) -> tuple[str, str]:
        """Build prompt for mood boosting"""
        system_prompt = (
            "You are Evo, a cheerful AI assistant/secretary/coach boosting users' moods when needed. "
            "Provide bursts of positive energy, motivation, and encouragement."
        )
        
        user_prompt = (
            f"Based on this planner summary: {summary}, "
            "give the user a burst of positive energy, motivation, and encouragement "
            "with many positive emojis and uplifting language."
        )
        
        return system_prompt, user_prompt

    @staticmethod
    def build_evening_compliment_prompt(
        *,
        has_tasks: bool,
        first_name: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build prompt for a single evening compliment push (light model)."""
        name_bit = ""
        if first_name and str(first_name).strip():
            name_bit = f" Address them gently as {str(first_name).strip()[:24]}."
        system_prompt = (
            "You are EVO — a warm evening companion inside a planning app. "
            "Write ONE sincere compliment (1-2 short sentences, max ~200 characters). "
            "Sound human, specific when possible, never preachy. "
            "No guilt, no streak talk, no task lists, no bullet points."
            + name_bit
        )
        if has_tasks:
            user_prompt = (
                "Give an evening compliment grounded in what they showed up for today "
                "(use the task titles provided — do not invent tasks). "
                "Celebrate effort and identity, not perfection. Use 0-1 emoji."
            )
        else:
            user_prompt = (
                "They had a light or empty calendar today. "
                "Give a gentle evening compliment about rest, showing up, or self-kindness. "
                "Use 0-1 emoji."
            )
        return system_prompt, user_prompt
    
    @staticmethod
    def build_todo_info_prompt(
        user_query: str,
        todo_data: Dict[str, Any],
        personalization_block: Optional[str] = None,
        general_coach: bool = False,
    ) -> tuple[str, str]:
        """Build prompt for providing information about todo_data"""
        if general_coach:
            system_prompt = (
                "You are EVO AI — a warm lifestyle coach inside a planning app. "
                "Read the user's planner, calendar, lifestyle profile, and behavior signals to understand their real life. "
                "Help them reach what they want: personalized routines, honest reflection, and organization that reduces stress. "
                "When MBTI is provided, tailor new task/planner ideas to that style (structure vs flexibility, social vs solo, etc.). "
                "If MBTI is unknown, infer gently from planner themes and intent signals — never stereotype. "
                "Proactively suggest new tasks or lifestyle plans that fit their patterns when they ask for ideas or feel stuck. "
                "Talk like a supportive friend who knows their calendar across last month, this month, and next month, "
                "plus today's tasks, active plans, and habits. "
                "Users keep tasks as reminders (they rarely tap complete) — treat past dates/times as handled. "
                "Never mention completion %, checkmarks, or guilt about unfinished tasks. "
                "Optimize for consistency on their real goals and how they feel — not streak anxiety. "
                "When load looks heavy, suggest trimming, spacing, or a lighter day — stress relief is valid coaching. "
                "When Planner content or PLAN ARC is provided, say what the plan/day is actually about "
                "(topic, day focus, concrete steps) — not just the calendar task title. "
                "If they ask for an image, acknowledge it warmly and keep text brief — default is 9:16 portrait; they can request 16:9 or 1:1. "
                "When saved profile fields are missing (MBTI, date of birth, current work) and would improve advice, "
                "ask for ONE field gently and mention the app can save it to their profile — never interrogate. "
                "Use planner data honestly — never invent schedules or progress you were not given. "
                "Stay encouraging without toxic positivity; lighter days and rest are okay."
            )
            user_prompt = (
                f"User query: {user_query}\n\n"
                "Friendly lifestyle check-in (not a formal report).\n"
                "Reply in 2-4 short, warm sentences:\n"
                "- Ground advice in planner/calendar context when provided\n"
                "- If they want goals or 'what's new', offer one personalized lifestyle or task idea that fits MBTI/behavior when known\n"
                "- If the schedule looks packed, name one stress-reducing organize/trim move\n"
                "- Acknowledge how they're doing — goals and feeling, not streaks\n"
                "- Suggest one small realistic next step when helpful\n"
                "Use 0-1 emoji max. Sound human and kind, not corporate or preachy."
            )
        else:
            system_prompt = (
                "You are EVO AI — a warm coach for a specific calendar task. "
                "When PLAN ARC / planner content is provided, ground every suggestion in that plan's "
                "goal, day focus, and steps — do not invent unrelated advice. "
                "Be honest, concise, and actionable. Never guilt-trip about streaks or completion %. "
                "When personalization context is provided, tailor advice to the user's schedule and goals — "
                "never invent facts beyond that context."
            )
            todo_info = "\n".join([f"• {key}: {value}" for key, value in todo_data.items()])
            user_prompt = (
                f"User query: {user_query}\n\n"
                f"Todo data:\n{todo_info}\n\n"
                "Provide a CONCISE response (2-4 sentences maximum) focusing only on what matters:\n\n"
                "- If planner/plan content is provided, tie advice to that plan day and main goal\n"
                "- If user asks about a specific point, give focused details\n"
                "- If user's query doesn't align with the todo data, point it out honestly\n"
                "- When relevant, add 1 practical suggestion for improvement\n"
                "- Only suggest links/resources if user explicitly asks\n\n"
                "Be honest: cheer up only when deserved, but be straightforward when something doesn't make sense. "
                "Use minimal emojis. Keep it short and practical."
            )

        if personalization_block and personalization_block.strip():
            user_prompt = f"{personalization_block.strip()}\n\n{user_prompt}"
        
        return system_prompt, user_prompt

class PlannerUtils:
    """Enhanced planner utilities with better error handling and validation"""
    
    def __init__(self, config: Optional[PlannerConfig] = None, wrapper: Optional[ChatGPTWrapper] = None):
        """Initialize planner utilities"""
        self.config = config or PlannerConfig()
        self.wrapper = wrapper or get_default_wrapper()
        self.validator = PlannerValidator()
        self.prompt_builder = PromptBuilder()
        
        logger.info("PlannerUtils initialized")
    
    def _safe_chat_call(self, system_prompt: str, user_prompt: str, language: str = "thai", model: str = "gpt-5.1", **kwargs) -> str:
        """Make a safe chat call with error handling and graceful degradation"""
        try:
            # Extract specific parameters from kwargs to avoid conflicts
            max_completion_tokens = kwargs.pop('max_completion_tokens', self.config.max_completion_tokens)
            temperature = kwargs.pop('temperature', self.config.temperature)
            top_p = kwargs.pop('top_p', self.config.top_p)
            
            return self.wrapper.chat_with_gpt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                language=language,
                model=model,
                **kwargs
            )
        except RateLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Chat call failed: {str(e)}")
            # Return graceful fallback based on language
            fallback_responses = {
                'thai': "ขออภัยครับ ระบบกำลังมีปัญหาเล็กน้อย กรุณาลองใหม่อีกครั้งในสักครู่ 😊",
                'english': "Sorry, I'm having a small issue right now. Please try again in a moment! 😊",
                'chinese': "抱歉，我现在有点小问题。请稍后再试！😊",
                'japanese': "申し訳ありませんが、少し問題があります。もう一度お試しください！😊",
                'korean': "죄송합니다. 지금 작은 문제가 있습니다. 잠시 후 다시 시도해주세요! 😊"
            }
            return fallback_responses.get(language, fallback_responses['english'])
    
    def summarize_plan(self, planner_data: Dict[str, Any], plan_type: str = "general", language: str = "thai") -> str:
        """
        Summarize planner data in a compact, friendly, and motivational tone.
        
        Args:
            planner_data: The planner data to summarize
            plan_type: Type of plan (general, specific, etc.)
            language: Language for the response
            
        Returns:
            Summarized plan with motivational tone
        """
        try:
            # Validate inputs
            self.validator.validate_planner_data(planner_data)
            normalized_language = self.validator.validate_language(language)
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_summarize_prompt(
                planner_data, normalized_language
            )
            
            # Make API call
            response = self._safe_chat_call(
                system_prompt, user_prompt, language=normalized_language
            )
            
            logger.info(f"Plan summarized successfully for language: {normalized_language}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to summarize plan: {str(e)}")
            return f"Sorry, I couldn't summarize your plan right now. Please try again. (Error: {str(e)})"
    
    def motivate_user(self, summary: str) -> str:
        """
        Provide motivational advice based on planner summary.
        
        Args:
            summary: The planner summary to base motivation on
            
        Returns:
            Motivational advice with energetic tone
        """
        try:
            # Validate input
            if not summary or not isinstance(summary, str):
                raise ValueError("Summary must be a non-empty string")
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_motivate_prompt(summary)
            
            # Make API call
            response = self._safe_chat_call(system_prompt, user_prompt)
            
            logger.info("Motivational advice generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate motivational advice: {str(e)}")
            return "Keep going! You're doing great! 💪✨"
    
    def track_progress(self, user_update: str, summary: str, todo_data: Dict[str, Any]) -> str:
        """
        Track user progress and provide constructive feedback.
        
        Args:
            user_update: User's progress update
            summary: Current planner summary
            todo_data: Todo list data
            
        Returns:
            Constructive feedback and encouragement
        """
        try:
            # Validate inputs
            user_update = self.validator.validate_user_input(user_update)
            self.validator.validate_planner_data(todo_data)
            
            if not summary or not isinstance(summary, str):
                summary = "No previous summary available"
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_progress_prompt(
                user_update, summary, todo_data
            )
            
            # Make API call with language detection
            response = self._safe_chat_call(
                system_prompt, user_prompt, reply_language=user_update
            )
            
            logger.info("Progress tracking feedback generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to track progress: {str(e)}")
            return "Thanks for the update! Keep up the great work! 🌟"
    
    def respond_to_user_input(
        self,
        user_input: str,
        summary: str,
        identity_context: Optional[Dict[str, Any]] = None,
        last_week_completion_rate: Optional[float] = None,
    ) -> str:
        """
        Respond to user input naturally and supportively.

        Args:
            user_input: User's input message
            summary: Current planner summary for context
            identity_context: Optional behavior signals (streaks, badges,
                day-of-week histogram). Per docs/ORCHESTRATION.md grounds
                coaching in this user's actual behavior.
            last_week_completion_rate: Optional 0..1 ratio for last 7 days.

        Returns:
            Natural and supportive response
        """
        try:
            # Validate inputs
            user_input = self.validator.validate_user_input(user_input)

            if not summary or not isinstance(summary, str):
                summary = "No planner context available"

            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_response_prompt(
                user_input, summary
            )

            # Inject identity / completion-rate signals into the user prompt
            # so the coach can reference real streaks and lean into the
            # latest "becoming" phrase. Append only — do not modify the
            # validator-built parts above so existing behavior is preserved
            # when no context is supplied.
            identity_block = _format_identity_context(identity_context, last_week_completion_rate)
            if identity_block:
                user_prompt = f"{identity_block}\n\n{user_prompt}"

            # Allow full supportive replies (Thai needs more tokens than Latin).
            response = self._safe_chat_call(
                system_prompt, user_prompt, max_completion_tokens=512
            )

            logger.info("User input response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Failed to respond to user input: {str(e)}")
            return "I'm here to help! What can I assist you with today? 😊"
    
    def mood_boost(self, summary: str) -> str:
        """
        Provide a burst of positive energy and motivation.
        
        Args:
            summary: Current planner summary
            
        Returns:
            Energetic and uplifting message
        """
        try:
            # Validate input
            if not summary or not isinstance(summary, str):
                summary = "Your amazing journey"
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_mood_boost_prompt(summary)
            
            # Make API call with higher temperature for more creativity
            response = self._safe_chat_call(
                system_prompt, user_prompt, temperature=0.9
            )
            
            logger.info("Mood boost generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate mood boost: {str(e)}")
            return "You're absolutely amazing! Keep shining bright! ✨🌟💫"

    def evening_compliment_message(
        self,
        today_todo_list_data: Optional[List[Dict[str, Any]]] = None,
        language: str = "thai",
        first_name: Optional[str] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        today_date: Optional[str] = None,
    ) -> str:
        """Generate a single evening compliment sentence (light model)."""
        try:
            tasks = [t for t in (today_todo_list_data or []) if isinstance(t, dict)]
            normalized_language = self.validator.validate_language(language)
            system_prompt, user_prompt = self.prompt_builder.build_evening_compliment_prompt(
                has_tasks=len(tasks) > 0,
                first_name=first_name,
            )

            if tasks:
                tasks_info = "\n".join([
                    (
                        f"• ✓ {str(task.get('title', 'Task'))[:48]}".strip()
                        if task.get("completed") or task.get("isCompleted")
                        else f"• {str(task.get('title', 'Task'))[:48]}".strip()
                    )
                    for task in _sorted_today_tasks(tasks)[:8]
                ])
                user_prompt += f"\n\nToday's tasks ({len(tasks)}):\n{tasks_info}"

            if today_date:
                user_prompt += f"\n\nDate: {today_date}"

            if isinstance(identity_context, dict) and identity_context:
                id_block = _format_identity_context(identity_context)
                if id_block.strip():
                    user_prompt = f"Identity context:\n{id_block[:400]}\n\n{user_prompt}"

            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                language=normalized_language,
                model="gpt-5.4-mini",
                max_completion_tokens=120,
                temperature=0.95,
            )
            text = (response or "").strip()
            if text:
                return text[:280]
            raise ValueError("empty evening compliment response")
        except Exception as e:
            logger.error("Failed to generate evening compliment: %s", e)
            if language == "thai":
                return "วันนี้คุณทำดีแล้ว — พักผ่อนให้หัวใจเบาลงนะ 💛"
            return "You showed up today — that's worth being proud of. Rest well. 💛"
    
    def get_todo_information_generator_response(
        self,
        user_query: str,
        todo_data: Dict[str, Any],
        language: str = "thai",
        personalization_block: Optional[str] = None,
        general_coach: bool = False,
    ) -> str:
        """
        Provide concise, honest, and practical information about todo_data.
        
        Features:
        - Brief, focused responses (2-4 sentences)
        - Honest feedback when queries don't align with todo data
        - Practical suggestions when relevant
        - Minimal emojis and cheering (only when deserved)
        
        Args:
            user_query: User's question about the todo list
            todo_data: Todo list data
            language: Language for the response
            personalization_block: Optional bounded identity/schedule/RAG context
            
        Returns:
            Concise, honest information about the todo list
        """
        try:
            # Validate inputs
            if not user_query or not isinstance(user_query, str):
                user_query = "Tell me about this todo list"
            
            if not general_coach:
                self.validator.validate_planner_data(todo_data)
            normalized_language = self.validator.validate_language(language)
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_todo_info_prompt(
                user_query,
                todo_data,
                personalization_block=personalization_block,
                general_coach=general_coach,
            )
            
            # Coach chat needs enough tokens for 2–4 complete Thai sentences.
            token_budget = 768 if general_coach else 400
            response = self._safe_chat_call(
                system_prompt, user_prompt, language=normalized_language, max_completion_tokens=token_budget
            )
            
            logger.info("Todo information generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get todo information: {str(e)}")
            return "I'm here to help! Could you tell me more about what you'd like to know? 😊"

    def _morning_mode_prompts(self, mode: str, language: str, total_tasks: int) -> Tuple[str, str]:
        """Return (system_prompt, user_task_suffix) for a morning push personality."""
        normalized = (mode or "todo_coach").strip().lower()
        char_limit = "Your response MUST be 180 characters or less (including spaces and emojis)."
        voice = (
            "You are Evo — a warm, lively friend who texts the user at 8 AM. "
            "Sound human, friendly, gently funny when appropriate, never preachy or corporate. "
            "Use 1–2 emojis max. No lucky/auspicious colors. No guilt about streaks. "
            f"{char_limit} Write entirely in the user's language ({language})."
        )

        mode_specs = {
            "todo_coach": (
                voice + " Energize them for today's tasks: mention task count and 1–2 times if they fit.",
                "Give a punchy morning coach line about today's todos — practical, upbeat, zero nagging.",
            ),
            "love_warmth": (
                voice + " Self-compassion and heart energy — like a kind friend, not a romance novel.",
                "Write a warm morning love-note to the reader (self-love / gentle belonging). No task list required.",
            ),
            "funny_boost": (
                voice + " Light, wholesome humor — playful, never sarcastic or mean. One tiny joke OK.",
                "Make them smile this morning with gentle humor plus real encouragement.",
            ),
            "identity_cheer": (
                voice + " Celebrate who they're becoming — identity, goals, and honest feeling, not streak scores.",
                "Cheer them on using identity/badge context if provided; focus on becoming and consistency on what matters.",
            ),
            "gentle_rest": (
                voice + " Cozy permission to breathe when the day is light — still hopeful, never lazy-shaming.",
                "They may have few or no todos — validate rest, soft momentum, or a tiny optional step.",
            ),
        }

        if normalized not in mode_specs:
            normalized = "todo_coach"
        system_extra, user_task = mode_specs[normalized]
        return system_extra, user_task

    def morning_message(
        self,
        today_todo_list_data: List[Dict[str, Any]],
        language: str = "thai",
        user_context: Optional[List[str]] = None,
        month_context: Optional[Dict[str, Any]] = None,
        earned_runes: Optional[List[Dict[str, Any]]] = None,
        behavior_stats: Optional[Dict[str, Any]] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        morning_mode: Optional[str] = "todo_coach",
        week_tasks: Optional[List[Dict[str, Any]]] = None,
        today_date: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a compact morning push message. ``morning_mode`` rotates personality:
        todo_coach, love_warmth, funny_boost, identity_cheer, gentle_rest.
        """
        try:
            normalized_mode = (morning_mode or "todo_coach").strip().lower()
            total_tasks = len(today_todo_list_data or [])

            if normalized_mode == "todo_coach" and total_tasks == 0:
                logger.info("todo_coach mode with no tasks — falling back to gentle_rest tone")
                normalized_mode = "gentle_rest"

            system_prompt, user_task = self._morning_mode_prompts(
                normalized_mode, language, total_tasks
            )

            tasks_info = ""
            if total_tasks > 0:
                tasks_info = "\n".join([
                    (
                        f"• ✓ {task.get('start', '')} {str(task.get('title', 'Task'))[:40]}".strip()
                        if task.get("completed")
                        else f"• {task.get('start', '')} {str(task.get('title', 'Task'))[:40]}".strip()
                    )
                    for task in _sorted_today_tasks(today_todo_list_data)
                ])

            user_prompt = user_task
            if total_tasks > 0:
                user_prompt += f"\n\nToday's tasks ({total_tasks}):\n{tasks_info}"
            else:
                user_prompt += "\n\nToday's tasks: none or very light."

            week_list = [
                t for t in (week_tasks or []) if isinstance(t, dict)
            ][:40]
            if week_list:
                week_by_date: Dict[str, List[Dict[str, Any]]] = {}
                for task in _sorted_today_tasks(week_list):
                    day_key = str(task.get("date") or "").strip() or "?"
                    week_by_date.setdefault(day_key, []).append(task)

                week_lines: List[str] = []
                for day_key in sorted(week_by_date.keys()):
                    day_tasks = week_by_date[day_key]
                    day_header = (
                        f"{day_key} ({len(day_tasks)})"
                        + (" — today" if today_date and day_key == today_date else "")
                    )
                    week_lines.append(day_header)
                    for task in day_tasks[:6]:
                        title = str(task.get("title") or "Task").strip()[:40]
                        start = str(task.get("start") or "").strip()
                        mark = "✓ " if task.get("completed") else ""
                        week_lines.append(f"  • {mark}{start} {title}".strip())
                    if len(day_tasks) > 6:
                        week_lines.append(f"  • … +{len(day_tasks) - 6} more")

                user_prompt += (
                    f"\n\nCalendar week ({len(week_list)} tasks):\n"
                    + "\n".join(week_lines)
                )

            if user_context:
                context_block = "\n".join(f"• {c}" for c in user_context[:8])
                user_prompt = (
                    f"Relevant user context:\n{context_block}\n\n{user_prompt}"
                )

            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"

            earned = normalize_earned_runes_for_llm(earned_runes)
            grounding_lines = []
            if earned:
                grounding_lines.append(
                    "Earned rune identities: "
                    + "; ".join(f"{e['name']} ({e['key']})" for e in earned[:8])
                    + (" …" if len(earned) > 8 else "")
                )
            if isinstance(behavior_stats, dict) and behavior_stats:
                streak = behavior_stats.get("current_streak")
                if streak:
                    grounding_lines.append(f"Current streak: {streak}")
            if isinstance(identity_context, dict) and identity_context:
                id_block = _format_identity_context(identity_context)
                if id_block.strip():
                    grounding_lines.append("Identity profile:\n" + id_block[:600])
            if grounding_lines:
                user_prompt = "\n".join(grounding_lines) + "\n\n" + user_prompt

            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=320,
                temperature=1.05,
                language=language,
            )

            task_block = _format_today_tasks_for_notification(
                today_todo_list_data, language
            )
            if task_block:
                if response and response.strip():
                    response = f"{task_block}\n\n{response.strip()}"
                else:
                    response = task_block

            logger.info(
                "Morning message generated (mode=%s, tasks=%s)",
                normalized_mode,
                total_tasks,
            )
            return response

        except Exception as e:
            logger.error(f"Failed to generate morning message: {str(e)}")
            task_block = _format_today_tasks_for_notification(
                today_todo_list_data, language
            )
            return task_block or None

    def reminder_message(
        self,
        target_task: Dict[str, Any],
        week_tasks: Optional[List[Dict[str, Any]]] = None,
        month_tasks: Optional[List[Dict[str, Any]]] = None,
        language: str = "thai",
        minutes_until: int = 0,
        time_until_text: Optional[str] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        earned_runes: Optional[List[Dict[str, Any]]] = None,
        behavior_stats: Optional[Dict[str, Any]] = None,
        today_date: Optional[str] = None,
        user_context: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generate a compact, personalized push notification body for an upcoming todo.
        Grounds the nudge in the target task plus light week/month calendar context.
        """
        try:
            normalized_language = self.validator.validate_language(language)
            task = target_task if isinstance(target_task, dict) else {}
            title = str(task.get("title") or "Task").strip()[:60]
            start = str(task.get("start") or "").strip()
            task_date = str(task.get("date") or "").strip()
            timing = (time_until_text or "").strip() or (
                f"{minutes_until} min" if minutes_until else "soon"
            )

            week_list = [
                t for t in (week_tasks or []) if isinstance(t, dict)
            ][:40]
            month_list = [
                t for t in (month_tasks or []) if isinstance(t, dict)
            ][:30]

            week_by_date: Dict[str, List[Dict[str, Any]]] = {}
            for t in _sorted_today_tasks(week_list):
                day_key = str(t.get("date") or "").strip() or "?"
                week_by_date.setdefault(day_key, []).append(t)

            week_lines: List[str] = []
            for day_key in sorted(week_by_date.keys()):
                day_tasks = week_by_date[day_key]
                is_today = bool(today_date and day_key == today_date)
                day_label = f"{day_key} ({len(day_tasks)})"
                if is_today:
                    day_label += " — today"
                week_lines.append(day_label)
                for t in day_tasks[:5]:
                    t_title = str(t.get("title") or "Task").strip()[:40]
                    t_start = str(t.get("start") or "").strip()
                    mark = "✓ " if t.get("completed") else ""
                    week_lines.append(f"  • {mark}{t_start} {t_title}".strip())
                if len(day_tasks) > 5:
                    week_lines.append(f"  • … +{len(day_tasks) - 5} more")

            week_block = (
                "\n".join(week_lines) if week_lines else "(light week)"
            )

            month_titles = [
                str(t.get("title") or "").strip()[:35]
                for t in month_list
                if str(t.get("title") or "").strip()
            ]
            month_hint = (
                f"{len(month_list)} tasks this month"
                + (
                    f" — e.g. {', '.join(month_titles[:4])}"
                    if month_titles
                    else ""
                )
            )

            system_prompt = (
                "You are Evo, a warm behavioral coach sending a mobile push reminder. "
                "Write ONLY the notification body (no title). "
                "Must mention how soon the event starts using the provided timing text. "
                "Reference the target task by name. "
                "Use the calendar week context to personalize — note if today is busy "
                "or if the week ahead is lighter/heavier. "
                "You may nod to identity/streak if helpful — agency-forward, never guilt. "
                "Maximum 160 characters. At most one emoji."
            )

            target_line = (
                f"Target: {start} {title} ({task_date})".strip()
                if start
                else f"Target: {title} ({task_date})".strip()
            )
            user_prompt = (
                f"{target_line}\n"
                f"Starts in: {timing}\n\n"
                f"Calendar week ({len(week_list)} tasks):\n{week_block}\n\n"
                f"Month context: {month_hint}\n\n"
                f"Write the push body in {normalized_language}."
            )

            if user_context:
                context_block = "\n".join(f"• {c}" for c in user_context[:6])
                user_prompt = (
                    f"Relevant user context:\n{context_block}\n\n{user_prompt}"
                )

            earned = normalize_earned_runes_for_llm(earned_runes)
            grounding_lines = []
            if isinstance(behavior_stats, dict) and behavior_stats:
                streak = behavior_stats.get("current_streak")
                if streak:
                    grounding_lines.append(f"Current streak: {streak}")
            if earned:
                grounding_lines.append(
                    "Earned rune identities: "
                    + "; ".join(f"{e['name']}" for e in earned[:4])
                )
            if isinstance(identity_context, dict) and identity_context:
                id_block = _format_identity_context(identity_context)
                if id_block.strip():
                    grounding_lines.append("Identity:\n" + id_block[:400])
            if grounding_lines:
                user_prompt = "\n".join(grounding_lines) + "\n\n" + user_prompt

            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=120,
                temperature=0.9,
                language=normalized_language,
            )
            if response and response.strip():
                return response.strip()
            return None
        except Exception as e:
            logger.error(f"Failed to generate reminder message: {str(e)}")
            return None

    def summarize_end_of_the_week_message(
        self,
        week_summary: List[Dict[str, Any]],
        language: str = 'thai',
        user_context: Optional[List[str]] = None,
        month_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate rest and recharge suggestions based on week summary.
        
        Args:
            week_summary: List of completed tasks/activities from the week
            language: Language for the response
            user_context: Optional RAG context (e.g. from user memory) to personalize suggestions
            month_context: Optional previous/current/next month data to improve relevance
            
        Returns:
            Personalized rest and recharge suggestions, or None if no data
        """
        try:
            # Validate inputs
            if not week_summary:
                logger.info("No week summary data found for end-of-week message")
                return None
            
            #normalized_language = self.validator.validate_language(language)
            total_activities = len(week_summary)
            
            # Build concise prompt
            system_prompt = (
                "You are Evo, a supportive AI assistant. "
                "Provide an encouraging weekly summary and suggest rest to recharge the user. "
                "Review the user's weekly accomplishments and provide personalized recharge suggestions. "
                "Include specific rest activities and preparation tips for the upcoming week. "
                "Remind users about maintaining work-life balance and celebrating their achievements. "
                "Keep responses uplifting, motivational, and actionable."
            )
            
            # Format week activities with minimal info
            activities_info = "\n".join([
                f"• {activity.get('title', 'Activity')} - {activity.get('typeOfTodo', '')} - {activity.get('start', '')}"
                for activity in week_summary  
            ])
            
            user_prompt = (
                f"Weekly summary of {total_activities} completed activities:\n{activities_info}\n"
                f"Summarize this week and suggest rest to recharge in {language}. "
                f"Include specific rest activities, preparation tips for next week, and work-life balance reminders. "
                f"Keep it encouraging with positive emojis. Keep it within 120 words. No filler words."
            )
            if user_context:
                context_block = "\n".join(f"• {c}" for c in user_context[:10])
                user_prompt = (
                    f"User's todo context (analyze to prevent overload, protect deep work, maintain goal momentum—be practical):\n{context_block}\n\n"
                    f"{user_prompt} "
                    "Using the context: suggest when to rest, when to batch tasks, and balance based on their habits. Tie suggestions to their actual todos."
                )
            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"
            
            # Make API call with optimized parameters
            # Note: 120 words needs ~180 tokens for English, ~250+ for Thai/CJK languages
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt, 
                max_completion_tokens=250,
                temperature=1.0,  # Balance creativity with consistency
                language=language
            )

            logger.info(f"End-of-week rest suggestions generated successfully for {total_activities} activities")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate end-of-week rest suggestions: {str(e)}")
            # Enhanced fallback suggestions with better formatting
            fallback_suggestions = {
                'thai': "สัปดาห์นี้คุณทำได้ดีมาก! พักผ่อนให้เต็มที่ นอนหลับให้เพียงพอ และทำกิจกรรมที่ชอบ 🌙✨",
                'english': "Great week! Take time to rest, sleep well, and do activities you enjoy 🌙✨",
                'chinese': "这周做得很好！好好休息，充足睡眠，做你喜欢的事情 🌙✨",
                'japanese': "今週はお疲れ様でした！十分に休んで、よく眠り、好きなことをしてください 🌙✨",
                'korean': "이번 주 정말 잘하셨어요! 충분히 휴식하고, 잘 자고, 좋아하는 활동을 하세요 🌙✨"
            }
            return fallback_suggestions.get(language, fallback_suggestions['english'])
    
    def summarize_next_week_message(
        self,
        week_data: List[Dict[str, Any]],
        language: str = "thai",
        user_context: Optional[List[str]] = None,
        month_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate next week summary based on week data.
        
        Args:
            week_data: List of upcoming tasks/events for next week
            language: Language for the response
            user_context: Optional RAG context (e.g. from user memory) to personalize suggestions
            month_context: Optional previous/current/next month data to improve relevance
            
        Returns:
            Personalized next week summary, or None if no data
        """
        try:
            # Validate inputs
            if not week_data:
                logger.info("No week data found for next week message")
                return None
            
            #normalized_language = self.validator.validate_language(language)
            total_tasks = len(week_data)
            
            # Build concise prompt
            system_prompt = (
                "You are Evo, creating next week previews. "
                "Provide motivating and actionable previews of upcoming tasks. "
                "Include brief preparation suggestions (e.g., gather resources, plan ahead, set priorities). "
                "Remind users about maintaining balance throughout the week (e.g., schedule breaks, stay hydrated, celebrate progress). "
                "Keep responses concise but helpful and encouraging."
            )
            
            # Format upcoming tasks with minimal info
            tasks_info = "\n".join([
                f"• {task.get('title', 'Task')} - {task.get('typeOfTodo', '')} - {task.get('start', '')}"
                for task in week_data  
            ])
            
            user_prompt = (
                f"Next week preview for {total_tasks} tasks:\n{tasks_info}\n"
                f"Highlight 2-3 priorities in {language}. Include preparation suggestions and balance reminders. "
                f"Keep it motivating with emojis. Keep it within 120 words. No filler words."
            )
            if user_context:
                context_block = "\n".join(f"• {c}" for c in user_context[:10])
                user_prompt = (
                    f"User's todo context (analyze to prevent overload, protect deep work, maintain goal momentum—be practical):\n{context_block}\n\n"
                    f"{user_prompt} "
                    "Using the context: suggest deep work vs meetings, when to rest, batching tasks, spacing heavy days. Tie suggestions to their actual todos."
                )
            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"
            
            # Make API call with optimized parameters
            # Note: 120 words needs ~180 tokens for English, ~250+ for Thai/CJK languages
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt, 
                max_completion_tokens=250,
                temperature=1.0,  # Balance creativity with consistency
                language=language
            )
            
            logger.info(f"Next week summary generated successfully for {total_tasks} tasks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate next week summary: {str(e)}")
            # Enhanced fallback messages with better formatting
            fallback_messages = {
                'thai': "สัปดาห์หน้ามีงานสำคัญรออยู่ เตรียมพร้อมและวางแผนให้ดีนะคะ 📅✨",
                'english': "Important tasks await next week. Stay prepared and plan well! 📅✨",
                'chinese': "下周有重要的任务等着你。请做好准备，好好规划！📅✨",
                'japanese': "来週は重要な仕事が待っています。準備を整え、計画を立てましょう！📅✨",
                'korean': "다음 주에 중요한 일이 기다리고 있습니다. 준비하고 계획을 잘 세우세요! 📅✨"
            }
            return fallback_messages.get(language, fallback_messages['english'])

    def suggest_schedule_optimizations(
        self,
        schedule_data: List[Dict[str, Any]],
        language: str = "thai",
        user_context: Optional[List[str]] = None,
        month_context: Optional[Dict[str, Any]] = None,
        scope: str = "day",
    ) -> Optional[str]:
        """
        Use RAG context + current schedule to suggest how to optimize the user's schedule.
        Scope can be 'day' (today) or 'week'.

        Args:
            schedule_data: List of todo/event items (title, start, typeOfTodo, etc.)
            language: Response language
            user_context: Optional RAG context (user habits, preferences, past patterns)
            month_context: Optional previous/current/next month data to improve relevance
            scope: 'day' or 'week'

        Returns:
            Text with concrete schedule optimization suggestions.
        """
        if not schedule_data:
            return None
        total = len(schedule_data)
        schedule_info = "\n".join([
            f"• {item.get('title', 'Task')} - {item.get('typeOfTodo', '')} - {item.get('start', '')} - {item.get('date', '')}"
            for item in schedule_data
        ])
        system_prompt = (
            "You are Evo, an AI schedule coach. Your job is to analyze the user's todo list and suggest how to optimize their schedule. "
            "Use the user's context (their actual todos, habits, past behavior) and any month context. "
            + RAG_TODO_ANALYSIS_AIMS +
            "Keep the response clear and in the requested language. Use bullet points. Aim for 80–150 words."
        )
        user_prompt = (
            f"Schedule to optimize ({scope}, {total} items):\n{schedule_info}\n\n"
            f"Analyze and provide suggestions in {language}."
        )
        if user_context:
            context_block = "\n".join(f"• {c}" for c in user_context[:10])
            user_prompt = (
                f"User's todo list context (analyze this to prevent overload, protect deep work, maintain goals—be practical):\n{context_block}\n\n"
                f"{user_prompt}"
            )
        month_block = _format_month_context(month_context)
        if month_block:
            user_prompt = f"{month_block}\n\n{user_prompt}"
        try:
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=400,
                temperature=0.7,
                language=language,
            )
            logger.info("Schedule optimization suggestions generated (scope=%s)", scope)
            return response
        except Exception as e:
            logger.error("Failed to generate schedule optimizations: %s", e)
            return None

    def analyze_todo_list(
        self,
        user_context: List[str],
        language: str = "thai",
        schedule_data: Optional[List[Dict[str, Any]]] = None,
        month_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Analyze the user's todo list (from RAG context) for: prevent overload, protect deep work,
        maintain goal momentum, and give practical advice. Optional current schedule_data for focus.
        """
        if not user_context or not isinstance(user_context, list):
            return None
        context_block = "\n".join(f"• {c}" for c in user_context[:15])
        system_prompt = (
            "You are Evo, an AI schedule and productivity coach. Analyze the user's todo list and give brief, practical advice. "
            + RAG_TODO_ANALYSIS_AIMS +
            "Output 2–4 short bullet points only. Use the requested language."
        )
        user_prompt = (
            f"User's todo list (from their history):\n{context_block}\n\n"
            f"Analyze for overload, deep work protection, goal momentum, and give practical suggestions. Respond in {language}."
        )
        if schedule_data:
            schedule_info = "\n".join([
                f"• {item.get('title', 'Task')} - {item.get('typeOfTodo', '')} - {item.get('start', '')} - {item.get('date', '')}"
                for item in schedule_data[:30]
            ])
            user_prompt = f"Current schedule to consider:\n{schedule_info}\n\n{user_prompt}"
        month_block = _format_month_context(month_context)
        if month_block:
            user_prompt = f"{month_block}\n\n{user_prompt}"
        try:
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=350,
                temperature=0.6,
                language=language,
            )
            logger.info("Todo list analysis generated")
            return response
        except Exception as e:
            logger.error("Failed to generate todo list analysis: %s", e)
            return None

    def summarize_this_month_todos_from_text(
        self,
        this_month_todos_text: str,
        language: str = "thai",
        month_context: Optional[Dict[str, Any]] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        last_week_completion_rate: Optional[float] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Generate a title and summary of this month's todos when the data is provided as a single string.
        Optional month_context (previous/current/next month data) improves relevance and continuity.

        Args:
            this_month_todos_text: Raw text describing this month's todos (any format)
            language: Language for the response (e.g. 'thai', 'english')
            month_context: Optional dict with keys "previous", "current", "next" (each str or list) for RAG-style context

        Returns:
            Tuple of (title, summary) where title is short and catchy, summary is motivating and actionable.
            Returns (None, None) if input is invalid or generation fails.
        """
        try:
            # Validate inputs
            if not this_month_todos_text or not this_month_todos_text.strip():
                logger.info("No this month's todos text provided for summary")
                return (None, None)

            # Truncate very long input to prevent timeout (limit to 5000 characters)
            original_length = len(this_month_todos_text)
            if original_length > 5000:
                logger.warning(f"Input text is very long ({original_length} chars), truncating to 5000 chars")
                this_month_todos_text = this_month_todos_text[:5000] + "..."

            normalized_language = self.validator.validate_language(language)
            
            # Build concise prompt for both title and summary (single API call, optimized for speed)
            system_prompt = """You are Evo, an inspiring AI coach.
                Return TITLE and SUMMARY in this exact format:
                TITLE: [5-10 words, 1-2 emojis]
                SUMMARY: [max 400 words, positive emojis]

                Format the output in a friendly, easy-to-read way:
                - TITLE: Use engaging, motivational language with appropriate emojis
                - SUMMARY: Use line breaks to separate key points
                - Include positive, encouraging emojis throughout
                - Use clear, concise language that's easy to scan
                - Make it inspiring and actionable
                - Keep paragraphs short for better readability
                
                In the SUMMARY, include helpful guidance:
                - Preparation tips for upcoming tasks (e.g., organize workspace, gather materials, review goals)
                - Work-life balance reminders (e.g., schedule downtime, take breaks, celebrate milestones, maintain healthy habits)
                - Encourage sustainable productivity and well-being
                - If month context is provided, reference continuity (e.g. building on last month, preparing for next)."""

            user_prompt = (
                f"This month's todos:\n{this_month_todos_text}\n\n"
                f"TITLE: [5-10 word catchy title with 1-2 emojis in {normalized_language}]\n"
                f"SUMMARY: [motivating summary, max 400 chars with emojis in {normalized_language}]"
            )
            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"
            identity_block = _format_identity_context(identity_context, last_week_completion_rate)
            if identity_block:
                user_prompt = f"{identity_block}\n\n{user_prompt}"

            # Generate both title and summary in a single API call (reduced tokens for faster response)
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=200,  # Reduced from 300 for faster generation
                temperature=1.0,
                language=normalized_language,
            )

            # Parse the response to extract title and summary
            title = None
            summary = None
            
            if response:
                response_text = response.strip()
                # Try to parse the structured response using regex for more robust parsing
                # Look for TITLE: and SUMMARY: markers (case-insensitive, flexible spacing)
                title_pattern = r'(?:^|\n)\s*TITLE\s*:\s*(.+?)(?=\n\s*SUMMARY\s*:|$)'
                summary_pattern = r'(?:^|\n)\s*SUMMARY\s*:\s*(.+?)(?=\n\s*TITLE\s*:|$)'
                
                # Try to find title and summary using regex
                title_match = re.search(title_pattern, response_text, re.IGNORECASE | re.DOTALL)
                summary_match = re.search(summary_pattern, response_text, re.IGNORECASE | re.DOTALL)
                
                if title_match:
                    title = title_match.group(1).strip()
                if summary_match:
                    summary = summary_match.group(1).strip()
                
                # If regex didn't work, try simpler line-by-line parsing
                if not title or not summary:
                    lines = response_text.split('\n')
                    for i, line in enumerate(lines):
                        line_upper = line.strip().upper()
                        if line_upper.startswith('TITLE:'):
                            title_text = line.split(':', 1)[1].strip() if ':' in line else ''
                            # Check if there's more on the same line or next lines before SUMMARY
                            if i + 1 < len(lines) and not lines[i + 1].strip().upper().startswith('SUMMARY:'):
                                # Collect until SUMMARY or end
                                j = i + 1
                                while j < len(lines) and not lines[j].strip().upper().startswith('SUMMARY:'):
                                    title_text += ' ' + lines[j].strip()
                                    j += 1
                            if title_text:
                                title = title_text.strip()
                        elif line_upper.startswith('SUMMARY:'):
                            summary_text = line.split(':', 1)[1].strip() if ':' in line else ''
                            # Collect remaining lines
                            if i + 1 < len(lines):
                                summary_text += ' ' + ' '.join([l.strip() for l in lines[i+1:]])
                            if summary_text:
                                summary = summary_text.strip()
                                break
                
                # Final fallback: if parsing completely failed, treat entire response as summary
                if not title or not summary:
                    logger.warning("Failed to parse title and summary from structured response, using fallback")
                    if not summary:
                        summary = response_text
                    # Use first few words as title if title parsing failed
                    if not title:
                        words = summary.split()[:8]  # First 8 words
                        title = ' '.join(words)
                        if len(title) > 60:
                            title = title[:57] + '...'

            logger.info("This month's todos title and summary (from text) generated successfully")
            return (title, summary)

        except Exception as e:
            logger.error(f"Failed to generate this month's todos title and summary from text: {str(e)}")
    def summarize_this_year_todos_from_text(
        self,
        this_year_todos_text: str,
        language: str = "thai",
        month_context: Optional[Dict[str, Any]] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        last_week_completion_rate: Optional[float] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Generate a title and summary of this year's todos when the data is provided as a single string.
        Optional month_context (previous/current/next month data) improves relevance and continuity.

        Args:
            this_year_todos_text: Raw text describing this year's todos (any format)
            language: Language for the response (e.g. 'thai', 'english')
            month_context: Optional dict with keys "previous", "current", "next" (each str or list) for RAG-style context

        Returns:
            Tuple of (title, summary). Returns (None, None) if input is invalid or generation fails.
        """
        try:
            # Validate inputs
            if not this_year_todos_text or not this_year_todos_text.strip():
                logger.info("No this year's todos text provided for summary")
                return (None, None)

            # Truncate very long input to prevent timeout (limit to 5000 characters)
            original_length = len(this_year_todos_text)
            if original_length > 5000:
                logger.warning(f"Input text is very long ({original_length} chars), truncating to 5000 chars")
                this_year_todos_text = this_year_todos_text[:5000] + "..."

            normalized_language = self.validator.validate_language(language)
            
            # Build concise prompt for both title and summary (single API call, optimized for speed)
            system_prompt = (
                "You are Evo, an inspiring AI coach. "
                "Return TITLE and SUMMARY in this exact format:\n"
                "TITLE: [5-10 words, 1-2 emojis]\n"
                "SUMMARY: [max 200 words, positive emojis]\n"
                "If month context is provided, reference continuity across months (e.g. last month's focus, next month's goals)."
            )

            user_prompt = (
                f"This year's todos:\n{this_year_todos_text}\n\n"
                f"TITLE: [5-10 word catchy title with 1-2 emojis in {normalized_language}]\n"
                f"SUMMARY: [motivating summary, max 200 chars with emojis in {normalized_language}]"
            )
            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"
            identity_block = _format_identity_context(identity_context, last_week_completion_rate)
            if identity_block:
                user_prompt = f"{identity_block}\n\n{user_prompt}"

            # Generate both title and summary in a single API call (reduced tokens for faster response)
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=200,  # Reduced from 300 for faster generation
                temperature=1.0,
                language=normalized_language,
            )

            # Parse the response to extract title and summary
            title = None
            summary = None
            
            if response:
                response_text = response.strip()
                # Try to parse the structured response using regex for more robust parsing
                # Look for TITLE: and SUMMARY: markers (case-insensitive, flexible spacing)
                title_pattern = r'(?:^|\n)\s*TITLE\s*:\s*(.+?)(?=\n\s*SUMMARY\s*:|$)'
                summary_pattern = r'(?:^|\n)\s*SUMMARY\s*:\s*(.+?)(?=\n\s*TITLE\s*:|$)'
                
                # Try to find title and summary using regex
                title_match = re.search(title_pattern, response_text, re.IGNORECASE | re.DOTALL)
                summary_match = re.search(summary_pattern, response_text, re.IGNORECASE | re.DOTALL)
                
                if title_match:
                    title = title_match.group(1).strip()
                if summary_match:
                    summary = summary_match.group(1).strip()
                
                # If regex didn't work, try simpler line-by-line parsing
                if not title or not summary:
                    lines = response_text.split('\n')
                    for i, line in enumerate(lines):
                        line_upper = line.strip().upper()
                        if line_upper.startswith('TITLE:'):
                            title_text = line.split(':', 1)[1].strip() if ':' in line else ''
                            # Check if there's more on the same line or next lines before SUMMARY
                            if i + 1 < len(lines) and not lines[i + 1].strip().upper().startswith('SUMMARY:'):
                                # Collect until SUMMARY or end
                                j = i + 1
                                while j < len(lines) and not lines[j].strip().upper().startswith('SUMMARY:'):
                                    title_text += ' ' + lines[j].strip()
                                    j += 1
                            if title_text:
                                title = title_text.strip()
                        elif line_upper.startswith('SUMMARY:'):
                            summary_text = line.split(':', 1)[1].strip() if ':' in line else ''
                            # Collect remaining lines
                            if i + 1 < len(lines):
                                summary_text += ' ' + ' '.join([l.strip() for l in lines[i+1:]])
                            if summary_text:
                                summary = summary_text.strip()
                                break
                
                # Final fallback: if parsing completely failed, treat entire response as summary
                if not title or not summary:
                    logger.warning("Failed to parse title and summary from structured response, using fallback")
                    if not summary:
                        summary = response_text
                    # Use first few words as title if title parsing failed
                    if not title:
                        words = summary.split()[:8]  # First 8 words
                        title = ' '.join(words)
                        if len(title) > 60:
                            title = title[:57] + '...'

            logger.info("This year's todos title and summary (from text) generated successfully")
            return (title, summary)

        except Exception as e:
            logger.error(f"Failed to generate this year's todos title and summary from text: {str(e)}")
            return (None, None)
    def predict_today_todo_fate_message(
        self,
        todo_data: List[Dict[str, Any]],
        language: str = "thai",
        model: str = "gpt-5.1",
        divination_system: str = "elder_futhark",
        earned_runes: Optional[List[Dict[str, Any]]] = None,
        behavior_stats: Optional[Dict[str, Any]] = None,
        output_style: str = "brief",
    ) -> str:
        """
        Predict today's todo fate based on the todo data.

        ``brief`` (default): legacy 1-rune "draw" + short line for push/notifications.

        ``share_card``: uses **earned** runes the user already unlocked (behavioral
        identity / plasticity), optional stats, and todos to write a longer
        fortune-style "path reading" that still frames outcomes as emerging from
        habits and context—not supernatural certainty.

        ``natural_power``: synthesizes **all** earned runes (meanings + identity
        lines) into a short personalized "natural power in you" status — capacities
        wired through practice and plasticity, not a random draw or dashboard tally.
        """
        try:
            normalized_language = self.validator.validate_language(language)
            normalized_output = (output_style or "brief").strip().lower()
            if normalized_output not in ("brief", "share_card", "natural_power"):
                normalized_output = "brief"

            earned = normalize_earned_runes_for_llm(
                earned_runes if isinstance(earned_runes, list) else None
            )
            stats = behavior_stats if isinstance(behavior_stats, dict) else {}

            if normalized_output == "share_card":
                rune_lines = []
                for r in earned[:24]:
                    if not isinstance(r, dict):
                        continue
                    key = r.get("key") or r.get("rune_key")
                    if not key:
                        continue
                    nm = r.get("name") or key
                    meaning = r.get("meaning") or ""
                    cat = r.get("category") or ""
                    becoming = r.get("becoming") or ""
                    rune_lines.append(
                        f"- {nm} ({key}): {meaning}"
                        + (f" [category: {cat}]" if cat else "")
                        + (f" — identity line: {becoming}" if becoming else "")
                    )
                rune_block = (
                    "Earned Elder Futhark identities (already unlocked by real behavior — use as vocabulary, not a new random draw):\n"
                    + ("\n".join(rune_lines) if rune_lines else "(No runes unlocked yet — speak to momentum and first steps.)")
                )
                stats_bits = []
                if stats:
                    for k in ("runes_unlocked", "runes_total", "current_streak", "completion_rate_7d"):
                        if k in stats and stats[k] is not None:
                            stats_bits.append(f"{k}: {stats[k]}")
                stats_block = (
                    "Behavior / progress signals:\n" + "\n".join(stats_bits)
                    if stats_bits
                    else "Behavior / progress signals: (none provided)"
                )
                todo_block = (
                    f"Today's todos (titles and completion where known):\n{todo_data}\n"
                    if todo_data
                    else "Today's todos: (none listed — still give a grounded reading from runes/stats.)\n"
                )
                system_prompt = (
                    "You are Evo, a warm behavioral coach. EVO uses Elder Futhark runes as "
                    "**recognized identities** earned through real actions — symbols of practice, "
                    "adaptation, and plasticity — not magical fate. "
                    "The user's life direction emerges from many small choices, constraints, and "
                    "context; never claim inevitability, curses, or supernatural knowledge. "
                    "You may use a poetic, fortune-teller *tone*, but every claim must stay "
                    "compatible with psychology and agency. "
                    "Do not mention lucky colors. No medical or legal advice. "
                    "Write entirely in the user's requested language. "
                    "Produce 2–4 sentences (about 350–650 characters for Thai or Latin scripts). "
                    "Optionally end with one short reflective question. At most 2 emojis."
                )
                user_prompt = (
                    f"{rune_block}\n\n{stats_block}\n\n{todo_block}\n"
                    f"Write the shareable path reading in {normalized_language}. "
                    "Weave at least one earned rune theme into how they are showing up with their todos; "
                    "if there are no todos, focus on identity momentum from runes/stats."
                )
                response = self._safe_chat_call(
                    system_prompt,
                    user_prompt,
                    max_completion_tokens=700,
                    temperature=0.95,
                    language=normalized_language,
                    model=model,
                )
                logger.info("Share-card style todo/rune reading generated successfully")
                return response

            if normalized_output == "natural_power":
                rune_lines = []
                for r in earned[:24]:
                    if not isinstance(r, dict):
                        continue
                    key = r.get("key") or r.get("rune_key")
                    if not key:
                        continue
                    nm = r.get("name") or key
                    meaning = r.get("meaning") or ""
                    cat = r.get("category") or ""
                    becoming = r.get("becoming") or ""
                    rune_lines.append(
                        f"- {nm} ({key}): {meaning}"
                        + (f" [category: {cat}]" if cat else "")
                        + (f" — earned identity: {becoming}" if becoming else "")
                    )
                rune_block = (
                    "All earned Elder Futhark identities (real behavior — the user's "
                    "natural-power vocabulary, not a new random draw):\n"
                    + ("\n".join(rune_lines) if rune_lines else "(No runes unlocked yet.)")
                )
                stats_bits = []
                if stats:
                    for k in ("runes_unlocked", "runes_total"):
                        if k in stats and stats[k] is not None:
                            stats_bits.append(f"{k}: {stats[k]}")
                stats_block = (
                    "Progress:\n" + "\n".join(stats_bits)
                    if stats_bits
                    else ""
                )
                system_prompt = (
                    "You are Evo, a warm behavioral coach. EVO uses Elder Futhark runes as "
                    "**recognized identities** earned through real actions — symbols of practice, "
                    "adaptation, and neuroplasticity — not magical fate or fortune-telling. "
                    "Write a compact 'natural power in you' status: inner capacities the user has "
                    "**already wired** through repetition, recovery, and return — like a living "
                    "profile of their nervous system's use-dependent change. "
                    "Weave themes from ALL listed runes into one coherent portrait; name at most "
                    "two rune symbols inline if it helps clarity. "
                    "Never claim supernatural knowledge, inevitability, or medical facts. "
                    "No lucky colors. Write entirely in the user's requested language. "
                    "Output ONLY the status text: 2–3 short sentences (~180–320 characters for "
                    "Latin scripts; Thai may run slightly longer). Warm, specific, agency-forward. "
                    "At most one emoji."
                )
                user_prompt = (
                    f"{rune_block}\n"
                    + (f"\n{stats_block}\n" if stats_block else "\n")
                    + f"Write the natural-power status in {normalized_language}. "
                    "If no runes are listed, encourage the first small rep that starts plasticity — "
                    "one sentence only."
                )
                response = self._safe_chat_call(
                    system_prompt,
                    user_prompt,
                    max_completion_tokens=400,
                    temperature=0.85,
                    language=normalized_language,
                    model=model,
                )
                logger.info("Natural-power rune status generated successfully")
                return response

            # --- brief (legacy) path ---
            normalized_divination_system = (divination_system or "elder_futhark").strip().lower()
            if normalized_divination_system == "elder_futhark":
                divination_instruction = (
                    "Use a 1-rune draw from the Elder Futhark (2nd to 8th century CE). "
                    "Name exactly one rune and give a practical, positive interpretation for today."
                )
            else:
                divination_instruction = (
                    f"Use the {normalized_divination_system} rune system for a 1-rune draw. "
                    "Name exactly one rune and give a practical, positive interpretation for today."
                )

            system_prompt = (
                "You are Evo, a warm rune guide texting the user at 8 AM. "
                "Sound friendly, lightly mystical, never doom-y. "
                f"{divination_instruction} "
                "Do not mention colors or lucky colors. "
                "Keep it lively and encouraging. Include 1-2 emojis. Maximum 180 characters."
            )
            ctx_bits = []
            if stats:
                for k in (
                    "runes_unlocked",
                    "runes_total",
                    "current_streak",
                    "longest_streak",
                    "completion_rate_7d",
                ):
                    if k in stats and stats[k] is not None:
                        ctx_bits.append(f"{k}: {stats[k]}")
            if earned:
                ctx_bits.append(
                    "Earned rune identities: "
                    + "; ".join(f"{e['name']} ({e['key']})" for e in earned[:10])
                    + (" …" if len(earned) > 10 else "")
                )
            extra_ctx = ""
            if ctx_bits:
                extra_ctx = (
                    "\n\nUser context from real activity (plasticity — not supernatural fate):\n"
                    + "\n".join(ctx_bits)
                )
                system_prompt += (
                    " You may nod briefly to one earned-identity theme when it fits; "
                    "still give today's one-rune draw as instructed. "
                    "If the user has structured plan todos, you may hint at doing one "
                    "practice step (drill material) before the day ends — no streak guilt."
                )
            if not todo_data:
                user_prompt = (
                    f"Generate today's rune guidance in {normalized_language}. "
                    f"{divination_instruction} "
                    "Do not mention colors or lucky colors. "
                    f"Keep it concise but helpful. Include 1-2 emojis. Maximum 150 characters."
                ) + extra_ctx
            else:
                user_prompt = (
                    f"Today's todos:\n{todo_data}\n\n"
                    f"Generate today's rune guidance in {normalized_language}. "
                    f"{divination_instruction} "
                    "Do not mention colors or lucky colors. "
                    f"Keep it concise but helpful. Include 1-2 emojis. Maximum 150 characters."
                ) + extra_ctx
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_completion_tokens=300,
                temperature=1.0,
                language=normalized_language,
                model=model,
            )
            logger.info("Today's todo fate prediction generated successfully")
            return response
        except RateLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate today's todo fate prediction: {str(e)}")
            return None
# Global instance for backward compatibility
_default_planner = None

def get_default_planner() -> PlannerUtils:
    """Get or create the default planner instance"""
    global _default_planner
    if _default_planner is None:
        _default_planner = PlannerUtils()
    return _default_planner

# Backward compatibility functions
def summarize_plan(planner_data: Dict[str, Any], plan_type: str = "general", language: str = "thai") -> str:
    """Backward compatibility function for plan summarization"""
    planner = get_default_planner()
    return planner.summarize_plan(planner_data, plan_type, language)

def motivate_user(summary: str) -> str:
    """Backward compatibility function for user motivation"""
    planner = get_default_planner()
    return planner.motivate_user(summary)

def track_progress(user_update: str, todo_data: Dict[str, Any], language: str) -> str:
    """Backward compatibility function for progress tracking"""
    planner = get_default_planner()
    summary = planner.summarize_plan(todo_data, "general", language)
    return planner.track_progress(user_update, summary, todo_data)

def respond_to_user_input(
    user_input: str,
    summary: str,
    identity_context: Optional[Dict[str, Any]] = None,
    last_week_completion_rate: Optional[float] = None,
) -> str:
    """Backward compatibility function for user input response"""
    planner = get_default_planner()
    return planner.respond_to_user_input(
        user_input,
        summary,
        identity_context=identity_context,
        last_week_completion_rate=last_week_completion_rate,
    )

def mood_boost(summary: str) -> str:
    """Backward compatibility function for mood boosting"""
    planner = get_default_planner()
    return planner.mood_boost(summary)


def evening_compliment_message(
    today_todo_list_data: Optional[List[Dict[str, Any]]] = None,
    language: str = "thai",
    first_name: Optional[str] = None,
    identity_context: Optional[Dict[str, Any]] = None,
    today_date: Optional[str] = None,
) -> str:
    """Backward compatibility wrapper for evening compliment pushes."""
    planner = get_default_planner()
    return planner.evening_compliment_message(
        today_todo_list_data=today_todo_list_data,
        language=language,
        first_name=first_name,
        identity_context=identity_context,
        today_date=today_date,
    )

def message_in_the_morning(
    today_todo_list_data: List[Dict[str, Any]],
    language: str = "thai",
    user_context: Optional[List[str]] = None,
    month_context: Optional[Dict[str, Any]] = None,
    earned_runes: Optional[List[Dict[str, Any]]] = None,
    behavior_stats: Optional[Dict[str, Any]] = None,
    identity_context: Optional[Dict[str, Any]] = None,
    morning_mode: Optional[str] = "todo_coach",
    week_tasks: Optional[List[Dict[str, Any]]] = None,
    today_date: Optional[str] = None,
) -> Optional[str]:
    """Backward compatibility function for message in the morning"""
    planner = get_default_planner()
    return planner.morning_message(
        today_todo_list_data,
        language,
        user_context=user_context,
        month_context=month_context,
        earned_runes=earned_runes,
        behavior_stats=behavior_stats,
        identity_context=identity_context,
        morning_mode=morning_mode,
        week_tasks=week_tasks,
        today_date=today_date,
    )


def todo_reminder_message(
    target_task: Dict[str, Any],
    week_tasks: Optional[List[Dict[str, Any]]] = None,
    month_tasks: Optional[List[Dict[str, Any]]] = None,
    language: str = "thai",
    minutes_until: int = 0,
    time_until_text: Optional[str] = None,
    identity_context: Optional[Dict[str, Any]] = None,
    earned_runes: Optional[List[Dict[str, Any]]] = None,
    behavior_stats: Optional[Dict[str, Any]] = None,
    today_date: Optional[str] = None,
    user_context: Optional[List[str]] = None,
) -> Optional[str]:
    """Backward compatibility function for personalized todo reminder push bodies."""
    planner = get_default_planner()
    return planner.reminder_message(
        target_task,
        week_tasks=week_tasks,
        month_tasks=month_tasks,
        language=language,
        minutes_until=minutes_until,
        time_until_text=time_until_text,
        identity_context=identity_context,
        earned_runes=earned_runes,
        behavior_stats=behavior_stats,
        today_date=today_date,
        user_context=user_context,
    )

def summarize_end_of_the_week_at_friday(
    week_data: List[Dict[str, Any]],
    language: str = "thai",
    user_context: Optional[List[str]] = None,
    month_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward compatibility function for summarize end of the week"""
    planner = get_default_planner()
    print(" == summarize_end_of_the_week_at_friday == ")
    print(week_data)
    print(language)
    return planner.summarize_end_of_the_week_message(week_data, language, user_context=user_context, month_context=month_context)

def summarize_next_week_at_sunday(
    week_data: List[Dict[str, Any]],
    language: str = "thai",
    user_context: Optional[List[str]] = None,
    month_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward compatibility function for summarize next week"""
    planner = get_default_planner()
    return planner.summarize_next_week_message(week_data, language, user_context=user_context, month_context=month_context)

def suggest_schedule_optimizations(
    schedule_data: List[Dict[str, Any]],
    language: str = "thai",
    user_context: Optional[List[str]] = None,
    month_context: Optional[Dict[str, Any]] = None,
    scope: str = "day",
) -> Optional[str]:
    """Suggest how to optimize the user's schedule using RAG context and current schedule."""
    planner = get_default_planner()
    return planner.suggest_schedule_optimizations(
        schedule_data=schedule_data,
        language=language,
        user_context=user_context,
        month_context=month_context,
        scope=scope,
    )


def analyze_todo_list(
    user_context: List[str],
    language: str = "thai",
    schedule_data: Optional[List[Dict[str, Any]]] = None,
    month_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Analyze user's todo list (RAG context) for overload, deep work, goal momentum; give practical advice."""
    planner = get_default_planner()
    return planner.analyze_todo_list(
        user_context=user_context,
        language=language,
        schedule_data=schedule_data,
        month_context=month_context,
    )


def get_todo_information(
    user_query: str,
    todo_data: Dict[str, Any],
    language: str = "thai",
    personalization_block: Optional[str] = None,
    general_coach: bool = False,
) -> str:
    """Backward compatibility function for getting todo information generator response"""
    planner = get_default_planner()
    return planner.get_todo_information_generator_response(
        user_query,
        todo_data,
        language,
        personalization_block=personalization_block,
        general_coach=general_coach,
    )

def summarize_this_year_todos_message(
    this_year_todos_data: str,
    language: str = "thai",
    month_context: Optional[Dict[str, Any]] = None,
    identity_context: Optional[Dict[str, Any]] = None,
    last_week_completion_rate: Optional[float] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Backward compatibility function for summarizing this year's todos"""
    planner = get_default_planner()
    return planner.summarize_this_year_todos_from_text(
        this_year_todos_data,
        language,
        month_context=month_context,
        identity_context=identity_context,
        last_week_completion_rate=last_week_completion_rate,
    )

def summarize_this_month_todos_message(
    this_month_todos_data: str,
    language: str = "thai",
    month_context: Optional[Dict[str, Any]] = None,
    identity_context: Optional[Dict[str, Any]] = None,
    last_week_completion_rate: Optional[float] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Backward compatibility function for summarizing this month's todos"""
    planner = get_default_planner()
    return planner.summarize_this_month_todos_from_text(
        this_month_todos_data,
        language,
        month_context=month_context,
        identity_context=identity_context,
        last_week_completion_rate=last_week_completion_rate,
    )

def predict_today_todo_fate(
    todo_data: List[Dict[str, Any]],
    language: str = "thai",
    divination_system: str = "elder_futhark",
    earned_runes: Optional[List[Dict[str, Any]]] = None,
    behavior_stats: Optional[Dict[str, Any]] = None,
    output_style: str = "brief",
) -> str:
    """Backward compatibility function for predicting today's todo fate"""
    planner = get_default_planner()
    return planner.predict_today_todo_fate_message(
        todo_data,
        language,
        model="gpt-4o-mini",
        divination_system=divination_system,
        earned_runes=earned_runes,
        behavior_stats=behavior_stats,
        output_style=output_style,
    )