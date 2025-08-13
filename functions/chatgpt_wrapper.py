# app/chatgpt_wrapper.py

import os
import time
import logging
import json
from typing import Optional, Dict, Any, List
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ModelType(Enum):
    """Available OpenAI models"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

@dataclass
class ChatConfig:
    """Configuration for chat requests"""
    model: str = ModelType.GPT_4O.value
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 300
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class LanguageDetector:
    """Language detection utility with caching"""
    
    # Language mapping (for better reuse)
    LANGUAGE_MAP = {
        'en': 'English', 'th': 'Thai', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
        'zh-cn': 'Chinese', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 
        'ru': 'Russian', 'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
        'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish', 'fi': 'Finnish',
        'pl': 'Polish', 'cs': 'Czech', 'sk': 'Slovak', 'hu': 'Hungarian',
        'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sl': 'Slovenian',
        'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'mt': 'Maltese',
        'el': 'Greek', 'tr': 'Turkish', 'he': 'Hebrew', 'ar': 'Arabic',
        'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'fa': 'Persian',
        'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino'
    }
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def detect_language(text: str) -> str:
        """Detect language with caching for performance"""
        try:
            if not text or len(text.strip()) < 3:
                return 'en'  # Default to English for very short texts
            return detect(text)
        except LangDetectException:
            logger.warning(f"Could not detect language for text: {text[:50]}...")
            return 'en'
    
    @staticmethod
    def get_language_name(language_code: str) -> str:
        """Get full language name from code"""
        return LanguageDetector.LANGUAGE_MAP.get(language_code.lower(), language_code)

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def can_proceed(self) -> bool:
        """Check if we can make another API call"""
        now = time.time()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record an API call"""
        self.calls.append(time.time())

class ChatGPTWrapper:
    """Enhanced ChatGPT wrapper with error handling, retries, and monitoring"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ChatConfig] = None):
        """Initialize the ChatGPT wrapper"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        
        self.config = config or ChatConfig()
        self.client = OpenAI(api_key=self.api_key, timeout=self.config.timeout)
        self.rate_limiter = RateLimiter()
        self.language_detector = LanguageDetector()
        
        logger.info(f"ChatGPT wrapper initialized with model: {self.config.model}")
    
    def _validate_inputs(self, system_prompt: str, user_prompt: str) -> None:
        """Validate input parameters"""
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")
        
        # Check for potential injection attempts
        suspicious_patterns = ['<script>', 'javascript:', 'data:text/html']
        for pattern in suspicious_patterns:
            if pattern.lower() in user_prompt.lower():
                logger.warning(f"Potential injection attempt detected: {pattern}")
                raise ValueError("Invalid input detected")
    
    def _prepare_messages(self, system_prompt: str, user_prompt: str, 
                         language_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for the API call"""
        # Append language instruction if specified
        if language_name:
            system_prompt = f"{system_prompt}\nPlease reply in {language_name}."
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _handle_api_error(self, error: Exception, attempt: int) -> str:
        """Handle API errors with appropriate logging and fallback"""
        error_msg = str(error)
        
        if "rate limit" in error_msg.lower():
            logger.warning(f"Rate limit hit on attempt {attempt}")
            return "I'm currently experiencing high demand. Please try again in a moment."
        
        elif "quota" in error_msg.lower():
            logger.error("OpenAI quota exceeded")
            return "Service temporarily unavailable due to quota limits."
        
        elif "timeout" in error_msg.lower():
            logger.warning(f"Timeout on attempt {attempt}")
            return "Request timed out. Please try again."
        
        elif "authentication" in error_msg.lower():
            logger.error("Authentication failed")
            return "Service configuration error. Please contact support."
        
        else:
            logger.error(f"Unexpected API error on attempt {attempt}: {error_msg}")
            return "I encountered an unexpected error. Please try again later."
    
    def _make_api_call(self, messages: List[Dict[str, str]], config: Optional[ChatConfig] = None, attempt: int = 1) -> str:
        """Make API call with retry logic"""
        try:
            # Use provided config or fall back to default
            current_config = config or self.config
            
            if not self.rate_limiter.can_proceed():
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(current_config.retry_delay * 2)
            
            self.rate_limiter.record_call()
            
            response: ChatCompletion = self.client.chat.completions.create(
                model=current_config.model,
                messages=messages,
                temperature=current_config.temperature,
                top_p=current_config.top_p,
                max_tokens=current_config.max_tokens,
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")
            
            logger.info(f"API call successful (attempt {attempt})")
            return content.strip()
            
        except Exception as e:
            if attempt < current_config.max_retries:
                logger.warning(f"API call failed on attempt {attempt}: {str(e)}")
                time.sleep(current_config.retry_delay * attempt)  # Exponential backoff
                return self._make_api_call(messages, config, attempt + 1)
            else:
                return self._handle_api_error(e, attempt)
    
    def chat_with_gpt(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        auto_detect_language: bool = True,
        reply_language: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Send a chat completion request to OpenAI API with enhanced error handling.

        Args:
            system_prompt: System message to set assistant behavior
            user_prompt: Message to send to the assistant
            model: OpenAI model name (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling probability (overrides config)
            max_tokens: Maximum tokens for response (overrides config)
            auto_detect_language: Whether to detect user prompt language
            reply_language: Force a reply language (overrides detection)
            language: Explicit language code (overrides detection)

        Returns:
            Assistant's reply or error message

        Raises:
            ValueError: If input validation fails
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(system_prompt, user_prompt)
            
            # Determine language
            language_name = None
            if language:
                language_name = self.language_detector.get_language_name(language)
            elif reply_language:
                language_name = self.language_detector.get_language_name(reply_language)
            elif auto_detect_language:
                user_language_code = self.language_detector.detect_language(user_prompt)
                logger.debug(f"Detected language: {user_language_code}")
                language_name = self.language_detector.get_language_name(user_language_code)
            
            # Prepare messages
            messages = self._prepare_messages(system_prompt, user_prompt, language_name)
            
            # Override config if parameters provided
            current_config = ChatConfig(
                model=model or self.config.model,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            
            # Make API call
            response = self._make_api_call(messages, current_config)
            
            # Log performance metrics
            duration = time.time() - start_time
            logger.info(f"Chat completion completed in {duration:.2f}s")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Chat completion failed after {duration:.2f}s: {str(e)}")
            raise RuntimeError(f"Failed to communicate with OpenAI API: {str(e)}") from e

# Global instance for backward compatibility
_default_wrapper = None

def get_default_wrapper() -> ChatGPTWrapper:
    """Get or create the default ChatGPT wrapper instance"""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = ChatGPTWrapper()
    return _default_wrapper

def chat_with_gpt(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 300,
    auto_detect_language: bool = True,
    reply_language: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Backward compatibility function for existing code.
    
    This function maintains the same interface as the original chat_with_gpt
    but uses the enhanced ChatGPTWrapper internally.
    """
    wrapper = get_default_wrapper()
    return wrapper.chat_with_gpt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        auto_detect_language=auto_detect_language,
        reply_language=reply_language,
        language=language
    )