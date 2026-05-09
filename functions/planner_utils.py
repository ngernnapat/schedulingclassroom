# app/planner_utils.py

import logging
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from chatgpt_wrapper import chat_with_gpt, ChatGPTWrapper, get_default_wrapper, RateLimitExceededError

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


# Analysis aims when using RAG todo context: what the model should do with the user's todo list
RAG_TODO_ANALYSIS_AIMS = """
When analyzing the user's todo list (from context above), always address these aims. Be practical, not theoretical—tie every point to their actual tasks and dates.

1. **Prevent overload**: Flag days or weeks with too many tasks, back-to-back meetings with no buffer, or unrealistic density. Suggest what to move, drop, or defer.
2. **Protect deep work time**: Identify focus-needed tasks (e.g. report, coding, study) and suggest when to block uninterrupted time; warn if they're squeezed between meetings.
3. **Maintain goal momentum**: Spot recurring or goal-related items (e.g. gym, learning, project milestones). Encourage consistency and suggest how to keep them visible and achievable.
4. **Be practical**: Give 2–4 concrete, actionable suggestions only. No generic advice—reference specific titles, dates, or patterns from their list. Use their language.
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
    max_completion_tokens: int = 200
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
    def build_todo_info_prompt(user_query: str, todo_data: Dict[str, Any]) -> tuple[str, str]:
        """Build prompt for providing information about todo_data"""
        system_prompt = (
            "You are Evo, a helpful AI assistant that provides practical information about todo lists. "
            "Be honest, concise, and actionable. If something doesn't make sense, say so clearly and helpfully."
        )
        
        # Format todo data for clarity
        todo_info = "\n".join([f"• {key}: {value}" for key, value in todo_data.items()])
        
        user_prompt = (
            f"User query: {user_query}\n\n"
            f"Todo data:\n{todo_info}\n\n"
            "Provide a CONCISE response (2-4 sentences maximum) focusing only on what matters:\n\n"
            "- If user asks about a specific point, give focused details\n"
            "- If user's query doesn't align with the todo data, point it out honestly\n"
            "- When relevant, add 1 practical suggestion for improvement\n"
            "- Only suggest links/resources if user explicitly asks\n\n"
            "Be honest: cheer up only when deserved, but be straightforward when something doesn't make sense. "
            "Use minimal emojis. Keep it short and practical."
        )
        
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
    
    def respond_to_user_input(self, user_input: str, summary: str) -> str:
        """
        Respond to user input naturally and supportively.
        
        Args:
            user_input: User's input message
            summary: Current planner summary for context
            
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
            
            # Make API call
            response = self._safe_chat_call(system_prompt, user_prompt)
            
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
    
    def get_todo_information_generator_response(self, user_query: str, todo_data: Dict[str, Any], language: str = "thai") -> str:
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
            
        Returns:
            Concise, honest information about the todo list
        """
        try:
            # Validate inputs
            if not user_query or not isinstance(user_query, str):
                user_query = "Tell me about this todo list"
            
            self.validator.validate_planner_data(todo_data)
            normalized_language = self.validator.validate_language(language)
            
            # Build prompt
            system_prompt, user_prompt = self.prompt_builder.build_todo_info_prompt(
                user_query, todo_data
            )
            
            # Make API call with moderate max_completion_tokens for concise responses
            response = self._safe_chat_call(
                system_prompt, user_prompt, language=normalized_language, max_completion_tokens=200
            )
            
            logger.info("Todo information generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get todo information: {str(e)}")
            return "I'm here to help! Could you tell me more about what you'd like to know? 😊"

    def morning_message(
        self,
        today_todo_list_data: List[Dict[str, Any]],
        language: str = "thai",
        user_context: Optional[List[str]] = None,
        month_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate a compact morning message summarizing today's tasks for notification.
        
        Args:
            today_todo_list_data: List of todo items for today
            language: Language for the response
            user_context: Optional RAG context (e.g. from user memory) to personalize the message
            month_context: Optional previous/current/next month data to improve relevance
        Returns:
            Compact notification message with task summary, or None if no tasks
        """
        try:
            if not today_todo_list_data:
                logger.info("No tasks found for morning message")
                return None

            total_tasks = len(today_todo_list_data)
            
            # Build concise prompt
            system_prompt = """You are Evo, a friendly AI assistant creating compact morning notifications for busy users.

CRITICAL CONSTRAINT: Your response MUST be exactly 150 characters or less (including all spaces, emojis, and punctuation).

OUTPUT FORMAT:
- Start with a warm greeting or brief motivational phrase (5-15 chars)
- Include the task count (e.g., "3 tasks today" or "5 todos")
- Add 1-2 relevant emojis for visual appeal
- Include today's auspicious color
- Use clear, scannable language - no filler words

PRIORITY ORDER (fit what you can within 150 chars):
1. Greeting/motivation + task count (required)
2. Task summary with schedule time (e.g., "Meeting 9AM, Report 2PM")
3. Today's fate/fortune prediction with auspicious color (brief positive outlook)

EXAMPLES OF GOOD OUTPUT:
- "Good morning! 🌅 4 tasks: Meeting 9AM, Report 2PM. Fate: Productive day! Lucky color: Blue 💪"
- "Hey! ☀️ 3 todos: Call 10AM, Review 3PM. Success awaits! Wear Green 🍀"
- "Morning! ✨ 5 tasks starting 8AM. Great things coming! Auspicious color: Gold 🎉"

Remember: Include task times, fate/fortune prediction, and auspicious color."""
            
            # Format tasks with minimal info
            tasks_info = "\n".join([
                f"• {task.get('title', 'Task')[:30]} - {task.get('detail', '')} - {task.get('start', '')}"
                for task in today_todo_list_data  # Show only 2 tasks
            ])

            remaining = total_tasks - 2 if total_tasks > 2 else 0
            if remaining:
                tasks_info += f"\n• +{remaining} more"
            
            user_prompt = (
                f"Morning notification for {total_tasks} tasks:\n{tasks_info}\n"
                "Include task summary with schedule times, today's fate/fortune prediction, and auspicious color. Keep it concise but helpful. Include count and 1-2 emojis. Maximum 150 characters."
            )
            if user_context:
                context_block = "\n".join(f"• {c}" for c in user_context[:10])
                user_prompt = (
                    f"Relevant user context (use to personalize and suggest schedule optimization):\n{context_block}\n\n"
                    f"TASK: {user_prompt} "
                    "If context suggests a quick schedule tip (e.g. buffer time, best focus time) and it fits within 150 characters, include it."
                )
            month_block = _format_month_context(month_context)
            if month_block:
                user_prompt = f"{month_block}\n\n{user_prompt}"
            
            # Make API call with optimized parameters
            # Note: max_completion_tokens is for token limit, not character limit
            # 150 characters needs ~100 tokens for English, ~200+ for Thai due to encoding
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt,
                max_completion_tokens=300,  # Enough tokens for 150-char response in any language
                temperature=1.0,  # Balance between creativity and consistency
                language=language
            )
            
            
            logger.info(f"Morning message generated successfully for {total_tasks} tasks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate morning message: {str(e)}")
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
    def predict_today_todo_fate_message(self, todo_data: List[Dict[str, Any]], language: str = "thai", model: str = "gpt-5.1") -> str:
        """
        Predict today's todo fate based on the todo data.
        
        Args:
            todo_data: List of todo data
            language: Language for the response
            model: Model to use for the response
        """
        try:
            # Validate inputs
            # if not todo_data:
            #     logger.info("No todo data provided for today's todo fate prediction")
            #     return None
            # Validate language
            normalized_language = self.validator.validate_language(language)
            # Build concise prompt
            system_prompt = (
                "You are Evo, a fortune teller. "
                "Predict today's todo fate based on the todo data. "
                "Include a brief positive outlook and suggest an auspicious color for today. "
                "Keep it concise but helpful. Include 1-2 emojis. Maximum 150 characters."
            )
            if not todo_data:
                user_prompt = (
                    f"Predict today's todo fate in {normalized_language}. "
                    f"Include a brief positive outlook and an auspicious color for today. "
                    f"Keep it concise but helpful. Include 1-2 emojis. Maximum 150 characters."
                )
            else:
                user_prompt = (
                    f"Today's todos:\n{todo_data}\n\n"
                    f"Predict today's todo fate in {normalized_language}. "
                    f"Include a brief positive outlook and an auspicious color for today. "
                    f"Keep it concise but helpful. Include 1-2 emojis. Maximum 150 characters."
                )
            # Make API call with optimized parameters
            # Note: 150 characters needs ~100 tokens for English, ~200+ for Thai/CJK languages
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

def respond_to_user_input(user_input: str, summary: str) -> str:
    """Backward compatibility function for user input response"""
    planner = get_default_planner()
    return planner.respond_to_user_input(user_input, summary)

def mood_boost(summary: str) -> str:
    """Backward compatibility function for mood boosting"""
    planner = get_default_planner()
    return planner.mood_boost(summary)

def message_in_the_morning(
    today_todo_list_data: List[Dict[str, Any]],
    language: str = "thai",
    user_context: Optional[List[str]] = None,
    month_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward compatibility function for message in the morning"""
    planner = get_default_planner()
    return planner.morning_message(today_todo_list_data, language, user_context=user_context, month_context=month_context)

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


def get_todo_information(user_query: str, todo_data: Dict[str, Any], language: str = "thai") -> str:
    """Backward compatibility function for getting todo information generator response"""
    planner = get_default_planner()
    return planner.get_todo_information_generator_response(user_query, todo_data, language)

def summarize_this_year_todos_message(
    this_year_todos_data: str,
    language: str = "thai",
    month_context: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Backward compatibility function for summarizing this year's todos"""
    planner = get_default_planner()
    return planner.summarize_this_year_todos_from_text(this_year_todos_data, language, month_context=month_context)

def summarize_this_month_todos_message(
    this_month_todos_data: str,
    language: str = "thai",
    month_context: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Backward compatibility function for summarizing this month's todos"""
    planner = get_default_planner()
    return planner.summarize_this_month_todos_from_text(this_month_todos_data, language, month_context=month_context)

def predict_today_todo_fate(todo_data: List[Dict[str, Any]], language: str = "thai") -> str:
    """Backward compatibility function for predicting today's todo fate"""
    planner = get_default_planner()
    return planner.predict_today_todo_fate_message(todo_data, language, model="gpt-4o-mini")