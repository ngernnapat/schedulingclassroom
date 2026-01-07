# app/planner_utils.py

import logging
import json
import re
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
    
    def _safe_chat_call(self, system_prompt: str, user_prompt: str, language: str = "thai", **kwargs) -> str:
        """Make a safe chat call with error handling and graceful degradation"""
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
            
            # Make API call with moderate max_tokens for concise responses
            response = self._safe_chat_call(
                system_prompt, user_prompt, language=normalized_language, max_tokens=200
            )
            
            logger.info("Todo information generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get todo information: {str(e)}")
            return "I'm here to help! Could you tell me more about what you'd like to know? 😊"

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
            
            # Build concise prompt
            system_prompt = """You are Evo, creating brief morning notifications.
                Keep responses motivating and actionable.

                Format the output in a friendly, easy-to-read way:
                - Use line breaks to separate key information
                - Include 1-2 relevant emojis for visual appeal
                - Use clear, concise language
                - Make it scannable and engaging
                - Start with a warm greeting or motivational phrase
                
                Include helpful suggestions:
                - Brief preparation tips for the tasks (e.g., gather materials, set up workspace, review notes)
                - Reminders about what to do after finishing tasks to maintain work-life balance (e.g., take breaks, hydrate, stretch, celebrate small wins, plan downtime)
                - Keep suggestions practical and encouraging"""
            
            # Format tasks with minimal info
            tasks_info = "\n".join([
                f"• {task.get('title', 'Task')[:30]} - {task.get('start', '')}"
                for task in today_todo_list_data[:2]  # Show only 2 tasks
            ])

            remaining = total_tasks - 2 if total_tasks > 2 else 0
            if remaining:
                tasks_info += f"\n• +{remaining} more"
            
            user_prompt = (
                f"Morning notification for {total_tasks} tasks:\n{tasks_info}\n"
                "Include task summary, brief preparation tips, and post-task balance reminders. Keep it concise but helpful. Include count and 1-2 emojis."
            )
            
            # Make API call with optimized parameters
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt,
                max_tokens=150,
                temperature=0.7,  # Balance between creativity and consistency
                language=language
            )
            
            logger.info(f"Morning message generated successfully for {total_tasks} tasks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate morning message: {str(e)}")
            return None


    def summarize_end_of_the_week_message(self, week_summary: List[Dict[str, Any]], language: str = 'thai') -> str:
        """
        Generate rest and recharge suggestions based on week summary.
        
        Args:
            week_summary: List of completed tasks/activities from the week
            language: Language for the response
            
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
                f"• {activity.get('title', 'Activity')[:25]} - {activity.get('typeOfTodo', '')}"
                for activity in week_summary[:4]  # Show only 4 activities
            ])
            
            if total_activities > 4:
                activities_info += f"\n• +{total_activities - 4} more"
            
            user_prompt = (
                f"Weekly summary of {total_activities} completed activities:\n{activities_info}\n"
                f"Summarize this week and suggest rest to recharge in {language}. "
                f"Include specific rest activities, preparation tips for next week, and work-life balance reminders. "
                f"Keep it encouraging with positive emojis."
            )
            
            # Make API call with optimized parameters
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt, 
                max_tokens=150,
                temperature=0.7,  # Balance creativity with consistency
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
    
    def summarize_next_week_message(self, week_data: List[Dict[str, Any]], language: str = "thai") -> str:
        """
        Generate next week summary based on week data.
        
        Args:
            week_data: List of upcoming tasks/events for next week
            language: Language for the response
            
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
                f"• {task.get('title', 'Task')[:20]} - {task.get('typeOfTodo', '')}"
                for task in week_data[:5]  # Show only 5 tasks
            ])
            
            remaining = total_tasks - 5 if total_tasks > 5 else 0
            if remaining:
                tasks_info += f"\n• +{remaining} more"
            
            user_prompt = (
                f"Next week preview for {total_tasks} tasks:\n{tasks_info}\n"
                f"Highlight 2-3 priorities in {language}. Include preparation suggestions and balance reminders. "
                f"Keep it motivating with emojis."
            )
            
            # Make API call with optimized parameters
            response = self._safe_chat_call(
                system_prompt, 
                user_prompt, 
                max_tokens=120,
                temperature=0.7,  # Balance creativity with consistency
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

    def summarize_this_month_todos_from_text(self, this_month_todos_text: str, language: str = "thai") -> tuple[str, str]:
        """
        Generate a title and summary of this month's todos when the data is provided as a single string.
        
        Args:
            this_month_todos_text: Raw text describing this month's todos (any format)
            language: Language for the response (e.g. 'thai', 'english')
            
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
                SUMMARY: [max 400 chars, positive emojis]

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
                - Encourage sustainable productivity and well-being"""

            user_prompt = (
                f"This month's todos:\n{this_month_todos_text}\n\n"
                f"TITLE: [5-10 word catchy title with 1-2 emojis in {normalized_language}]\n"
                f"SUMMARY: [motivating summary, max 400 chars with emojis in {normalized_language}]"
            )

            # Generate both title and summary in a single API call (reduced tokens for faster response)
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_tokens=200,  # Reduced from 300 for faster generation
                temperature=0.7,
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
    def summarize_this_year_todos_from_text(self, this_year_todos_text: str, language: str = "thai") -> tuple[str, str]:
        """
        Generate a title and summary of this year's todos when the data is provided as a single string.
        
        Args:
            this_year_todos_text: Raw text describing this year's todos (any format)
            language: Language for the response (e.g. 'thai', 'english')
            
        Returns:
            Tuple of (title, summary) where title is short and catchy, summary is motivating and actionable.
            Returns (None, None) if input is invalid or generation fails.
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
                "SUMMARY: [max 200 chars, positive emojis]"
            )

            user_prompt = (
                f"This year's todos:\n{this_year_todos_text}\n\n"
                f"TITLE: [5-10 word catchy title with 1-2 emojis in {normalized_language}]\n"
                f"SUMMARY: [motivating summary, max 200 chars with emojis in {normalized_language}]"
            )

            # Generate both title and summary in a single API call (reduced tokens for faster response)
            response = self._safe_chat_call(
                system_prompt,
                user_prompt,
                max_tokens=200,  # Reduced from 300 for faster generation
                temperature=0.7,
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

def summarize_end_of_the_week_at_friday(week_data: List[Dict[str, Any]], language: str = "thai") -> str:
    """Backward compatibility function for summarize end of the week"""
    planner = get_default_planner()
    print(" == summarize_end_of_the_week_at_friday == ")
    print(week_data)
    print(language)
    return planner.summarize_end_of_the_week_message(week_data, language)

def summarize_next_week_at_sunday(week_data: List[Dict[str, Any]], language: str = "thai") -> str:
    """Backward compatibility function for summarize next week"""
    planner = get_default_planner()
    return planner.summarize_next_week_message(week_data, language)

def get_todo_information(user_query: str, todo_data: Dict[str, Any], language: str = "thai") -> str:
    """Backward compatibility function for getting todo information generator response"""
    planner = get_default_planner()
    return planner.get_todo_information_generator_response(user_query, todo_data, language)

def summarize_this_year_todos_message(this_year_todos_data: str, language: str = "thai") -> tuple[str, str]:
    """Backward compatibility function for summarizing this year's todos"""
    planner = get_default_planner()
    title, summary = planner.summarize_this_year_todos_from_text(this_year_todos_data, language)
    return title, summary

def summarize_this_month_todos_message(this_month_todos_data: str, language: str = "thai") -> tuple[str, str]:
    """Backward compatibility function for summarizing this month's todos"""
    planner = get_default_planner()
    title, summary = planner.summarize_this_month_todos_from_text(this_month_todos_data, language)
    return title, summary