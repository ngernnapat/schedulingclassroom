# app/chatgpt_wrapper.py

import os
import time
import logging
import json
from typing import Optional, Dict, Any, List
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
from threading import Lock

import requests
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RateLimitExceededError(Exception):
    """Raised when OpenAI API rate limit is exceeded after all retries."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelType(Enum):
    """Available OpenAI models"""
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"
    GPT_5_2 = "gpt-5.2"
    
    GPT_4o_mini = "gpt-4o-mini"
    GPT_4o_turbo = "gpt-4o-turbo"
    GPT_4o_turbo_mini = "gpt-4o-turbo-mini"
    GPT_41_mini ="gpt-4.1-mini"
   

@dataclass
class ChatConfig:
    """Configuration for chat requests"""
    model: str = ModelType.GPT_5_1.value
    temperature: float = 1.0
    top_p: float = 0.9
    max_completion_tokens: int = 1024  # Increased from 300 to allow fuller responses
    timeout: int = 60  # Increased from 30 to 60 seconds
    max_retries: int = 5  # Increased for better rate limit handling
    retry_delay: float = 1.0
    connection_timeout: int = 10  # New: connection timeout
    read_timeout: int = 60  # New: read timeout

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

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = Lock()
    
    def can_proceed(self) -> bool:
        """Check if requests can proceed through the circuit breaker"""
        with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record a successful request"""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed request"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
            logger.info("Circuit breaker manually reset")

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 8, time_window: float = 60.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self.lock = Lock()  # Thread-safe operations
    
    def can_proceed(self) -> bool:
        """Check if we can make another API call"""
        with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record an API call"""
        with self.lock:
            self.calls.append(time.time())
    
    def get_wait_time(self) -> float:
        """Get the time to wait before the next call is allowed"""
        with self.lock:
            if not self.calls:
                return 0.0
            now = time.time()
            # Find the oldest call still in the window
            valid_calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            if len(valid_calls) < self.max_calls:
                return 0.0
            # Calculate when the oldest call will expire
            oldest_call = min(valid_calls)
            return (oldest_call + self.time_window) - now

class ChatGPTWrapper:
    """Enhanced ChatGPT wrapper with error handling, retries, and monitoring"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ChatConfig] = None):
        """Initialize the ChatGPT wrapper"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        
        self.config = config or ChatConfig()
        # Create client with more granular timeout control
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=(self.config.connection_timeout, self.config.read_timeout)
        )
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker()
        self.language_detector = LanguageDetector()
        
        logger.info(f"ChatGPT wrapper initialized with model: {self.config.model}")
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker if it's stuck open"""
        self.circuit_breaker.reset()
    
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
        
        # Handle specific OpenAI exceptions
        if isinstance(error, APITimeoutError):
            logger.warning(f"API timeout on attempt {attempt}: {error_msg}")
            return "Request timed out. Please try again in a moment."
        
        elif isinstance(error, APIConnectionError):
            logger.warning(f"API connection error on attempt {attempt}: {error_msg}")
            return "Connection issue. Please check your internet connection and try again."
        
        elif isinstance(error, RateLimitError):
            logger.warning(f"Rate limit exceeded on attempt {attempt}")
            return "I'm currently experiencing high demand. Please try again in a moment."
        
        elif isinstance(error, APIError):
            logger.error(f"OpenAI API error on attempt {attempt}: {error_msg}")
            if "quota" in error_msg.lower():
                return "Service temporarily unavailable due to quota limits."
            elif "authentication" in error_msg.lower():
                return "Service configuration error. Please contact support."
            else:
                return "Service temporarily unavailable. Please try again later."
        
        # Handle general error patterns
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            logger.warning(f"Timeout on attempt {attempt}: {error_msg}")
            return "Request timed out. Please try again."
        
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            logger.warning(f"Network error on attempt {attempt}: {error_msg}")
            return "Network connection issue. Please try again."
        
        elif "rate limit" in error_msg.lower():
            logger.warning(f"Rate limit hit on attempt {attempt}")
            return "I'm currently experiencing high demand. Please try again in a moment."
        
        else:
            logger.error(f"Unexpected API error on attempt {attempt}: {error_msg}")
            return "I encountered an unexpected error. Please try again later."
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after time from error response if available"""
        try:
            if hasattr(error, 'response') and error.response is not None:
                headers = error.response.headers
                if 'retry-after' in headers:
                    retry_after = float(headers['retry-after'])
                    logger.info(f"Extracted retry-after: {retry_after} seconds")
                    return retry_after
        except (AttributeError, ValueError, TypeError):
            pass
        return None
    
    def _make_api_call(self, messages: List[Dict[str, str]], config: Optional[ChatConfig] = None, attempt: int = 1) -> str:
        """Make API call with retry logic and circuit breaker"""
        try:
            # Check circuit breaker first
            if not self.circuit_breaker.can_proceed():
                logger.warning("Circuit breaker is OPEN, request blocked")
                return "Service is temporarily unavailable due to recent failures. Please try again later."
            
            # Use provided config or fall back to default
            current_config = config or self.config
            
            # Check rate limiting
            if not self.rate_limiter.can_proceed():
                wait_time = max(self.rate_limiter.get_wait_time(), current_config.retry_delay * 2)
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            self.rate_limiter.record_call()
            
            # Models that use newer API parameters (max_completion_tokens, no temp/top_p)
            reasoning_models = ["o1", "o1-mini", "o1-preview", "o3-mini", "o3"]
            # Models that only support temperature=1.0 but use standard max_completion_tokens
            temp_restricted_models = ["gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5"]
            
            model_lower = current_config.model.lower()
            is_reasoning_model = any(rm in model_lower for rm in reasoning_models)
            is_temp_restricted = any(rm in model_lower for rm in temp_restricted_models)
            
            # Build API call parameters based on model type
            api_params = {
                "model": current_config.model,
                "messages": messages,
            }
            
            # Use appropriate max tokens parameter based on model
            if is_reasoning_model:
                # Reasoning models use max_completion_tokens
                api_params["max_completion_tokens"] = current_config.max_completion_tokens
            else:
                # Standard models use max_completion_tokens
                api_params["max_completion_tokens"] = current_config.max_completion_tokens
            
            # Only include temperature/top_p if the model supports them
            if not is_reasoning_model and not is_temp_restricted:
                api_params["temperature"] = current_config.temperature
                api_params["top_p"] = current_config.top_p
            elif current_config.temperature != 1.0:
                logger.info(f"Model {current_config.model} only supports temperature=1.0, ignoring temperature={current_config.temperature}")
            
            logger.info(f"API params: model={current_config.model}, max_completion_tokens={current_config.max_completion_tokens}")
            
            # Make the API call
            response: ChatCompletion = self.client.chat.completions.create(**api_params)
            
            # Log response details for debugging
            logger.debug(f"Response: {response}")
            # Extract content from response, handling various response formats
            content = None
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                logger.debug(f"Message: {message}, finish_reason: {finish_reason}")
            
                content = message.content
                # Check for refusal (some models return refusal instead of content)
                if not content and hasattr(message, 'refusal') and message.refusal:
                    logger.warning(f"Model refused to respond: {message.refusal}")
                    content = f"Unable to process request: {message.refusal}"
                
                # Log finish reason for debugging
                finish_reason = response.choices[0].finish_reason
                if finish_reason and finish_reason != "stop":
                    logger.info(f"Response finish_reason: {finish_reason}")
                
                # Log if content is empty (regardless of finish_reason)
                if not content:
                    logger.warning(f"Empty content received. finish_reason={finish_reason}, usage={response.usage}")
            
            if not content:
                # Log response structure for debugging
                finish_reason = response.choices[0].finish_reason if response.choices else None
                logger.warning(f"Empty response. Model: {current_config.model}, Choices: {len(response.choices) if response.choices else 0}, finish_reason: {finish_reason}")
                if response.choices and len(response.choices) > 0:
                    logger.warning(f"Message object: {response.choices[0].message}")
                
                # Don't trip circuit breaker for empty responses - it's likely a prompt/token issue, not service failure
                # Return a user-friendly message instead of raising an exception
                if finish_reason == "length":
                    logger.warning(f"Response cut off due to token limit. max_completion_tokens={current_config.max_completion_tokens}")
                    return "Response was cut off due to token limit. Please try with a shorter prompt or increase max_completion_tokens."
                else:
                    return "The model returned an empty response. Please try rephrasing your request."
            
            # Handle partial response (content exists but was cut off)
            finish_reason = response.choices[0].finish_reason if response.choices else None
            if finish_reason == "length" and content:
                logger.warning(f"Response truncated at {len(content)} chars due to token limit ({current_config.max_completion_tokens} tokens)")
                # Return the partial content - it may still be useful
            
            # Record success in circuit breaker
            self.circuit_breaker.record_success()
            logger.info(f"API call successful (attempt {attempt})")
            return content.strip()
            
        except RateLimitError as e:
            # Handle rate limit errors with special retry logic
            self.circuit_breaker.record_failure()
            
            # Extract retry-after if available
            retry_after = self._extract_retry_after(e)
            
            # Check if we should retry
            if attempt < current_config.max_retries:
                # Use retry-after if available, otherwise use longer exponential backoff for rate limits
                if retry_after:
                    backoff_time = retry_after + (time.time() % 1)  # Add jitter
                else:
                    # Longer backoff for rate limits: start with 5 seconds, then exponential
                    backoff_time = 5.0 * (2 ** (attempt - 1)) + (time.time() % 1)
                
                logger.warning(f"Rate limit exceeded on attempt {attempt}, retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
                return self._make_api_call(messages, config, attempt + 1)
            else:
                logger.error(f"Rate limit exceeded after {current_config.max_retries} attempts")
                raise RateLimitExceededError(
                    "Rate limit exceeded. Please try again later.",
                    retry_after=retry_after
                )
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for temperature not supported error - don't retry with exponential backoff,
            # instead retry immediately without temperature parameter
            if "temperature" in error_msg.lower() and "unsupported" in error_msg.lower():
                logger.warning(f"Model doesn't support custom temperature, retrying without temperature parameter")
                # Create a new config with default temperature
                temp_config = ChatConfig(
                    model=current_config.model,
                    temperature=1.0,  # Use default temperature
                    top_p=1.0,  # Also reset top_p to default
                    max_completion_tokens=current_config.max_completion_tokens,
                    timeout=current_config.timeout,
                    max_retries=current_config.max_retries,
                    retry_delay=current_config.retry_delay
                )
                # Retry once with the corrected config
                if attempt == 1:  # Only retry once for this specific error
                    return self._make_api_call(messages, temp_config, attempt + 1)
                else:
                    self.circuit_breaker.record_failure()
                    return self._handle_api_error(e, attempt)
            
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure()
            
            # Check if we should retry
            if attempt < current_config.max_retries:
                # Calculate exponential backoff with jitter
                backoff_time = current_config.retry_delay * (2 ** (attempt - 1)) + (time.time() % 1)
                logger.warning(f"API call failed on attempt {attempt}: {error_msg}, retrying in {backoff_time:.1f}s")
                time.sleep(backoff_time)
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
        max_completion_tokens: Optional[int] = None,
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
            max_completion_tokens: Maximum tokens for response (overrides config)
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
                max_completion_tokens=max_completion_tokens or self.config.max_completion_tokens,
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

def reset_circuit_breaker():
    """Reset the circuit breaker on the default wrapper"""
    wrapper = get_default_wrapper()
    wrapper.reset_circuit_breaker()

def chat_with_gpt(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5-mini",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_completion_tokens: int = 1024,
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
        max_completion_tokens=max_completion_tokens,
        auto_detect_language=auto_detect_language,
        reply_language=reply_language,
        language=language
    )