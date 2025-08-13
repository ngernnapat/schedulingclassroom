# app/planner_utils.py

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from chatgpt_wrapper import chat_with_gpt, ChatGPTWrapper, get_default_wrapper

# Configure logging
logger = logging.getLogger(__name__)

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
    max_tokens: int = 200
    temperature: float = 0.7
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

class PlannerUtils:
    """Enhanced planner utilities with better error handling and validation"""
    
    def __init__(self, config: Optional[PlannerConfig] = None, wrapper: Optional[ChatGPTWrapper] = None):
        """Initialize planner utilities"""
        self.config = config or PlannerConfig()
        self.wrapper = wrapper or get_default_wrapper()
        self.validator = PlannerValidator()
        self.prompt_builder = PromptBuilder()
        
        logger.info("PlannerUtils initialized")
    
    def _safe_chat_call(self, system_prompt: str, user_prompt: str, language: str = "thai", **kwargs) -> str:
        """Make a safe chat call with error handling"""
        try:
            # Extract specific parameters from kwargs to avoid conflicts
            max_tokens = kwargs.pop('max_tokens', self.config.max_tokens)
            temperature = kwargs.pop('temperature', self.config.temperature)
            top_p = kwargs.pop('top_p', self.config.top_p)
            
            return self.wrapper.chat_with_gpt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                language=language,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Chat call failed: {str(e)}")
            return f"I'm having trouble processing your request right now. Please try again later. (Error: {str(e)})"
    
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

    def morning_message(self, today_todo_list_data: List[Dict[str, Any]], language: str = "thai") -> str:
        """
        Generate a compact morning message summarizing today's tasks for notification.
        
        Args:
            today_todo_list_data: List of todo items for today
            language: Language for the response
        Returns:
            Compact notification message with task summary, or None if no tasks
        """
        try:
            if not today_todo_list_data:
                logger.info("No tasks found for morning message")
                return None

            total_tasks = len(today_todo_list_data)
            
            # Build prompt
            system_prompt = (
                "You are Evo, a concise AI assistant creating brief morning notifications. "
                "Keep responses very short, motivating and actionable. "
                "Focus on the most important information."
            )
            
            # Format tasks with better structure and only essential info
            tasks_info = "\n".join([
                f"• {task.get('title', 'Untitled')} - {task.get('start', '')}" +
                (f" ({task.get('location')})" if task.get('location') else "")
                for task in today_todo_list_data[:3]
            ])

            remaining = total_tasks - 3 if total_tasks > 3 else 0
            if remaining:
                tasks_info += f"\n• ...and {remaining} more tasks"
            
            user_prompt = (
                f"Create an energizing morning notification (max 100 chars) for {total_tasks} tasks:\n{tasks_info}\n\n"
                "Include task count and 1-2 relevant emojis. Be motivating but extremely concise."
            )
            
            # Make API call with optimized parameters
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt,
                max_tokens=50,
                temperature=0.7,  # Balance between creativity and consistency
                language=language
            )
            
            logger.info(f"Morning message generated successfully for {total_tasks} tasks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate morning message: {str(e)}")
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

def message_in_the_morning(today_todo_list_data: List[Dict[str, Any]], language: str = "thai") -> str:
    """Backward compatibility function for message in the morning"""
    planner = get_default_planner()
    return planner.morning_message(today_todo_list_data, language)