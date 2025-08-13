#!/usr/bin/env python3
"""
Test script for improved ChatGPT wrapper and planner utilities
"""

import json
import time
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the improved modules
from chatgpt_wrapper import ChatGPTWrapper, ChatConfig, LanguageDetector, RateLimiter
from planner_utils import PlannerUtils, PlannerConfig, PlannerValidator, PromptBuilder
from config import get_config, AppConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestChatGPTWrapper(unittest.TestCase):
    """Test cases for improved ChatGPT wrapper"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = ChatConfig(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=100,
            timeout=10,
            max_retries=2
        )
        
        # Mock API key for testing
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            self.wrapper = ChatGPTWrapper(config=self.config)
    
    def test_initialization(self):
        """Test wrapper initialization"""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.config.model, "gpt-4o")
        self.assertEqual(self.wrapper.config.temperature, 0.7)
    
    def test_input_validation(self):
        """Test input validation"""
        # Test empty system prompt
        with self.assertRaises(ValueError):
            self.wrapper._validate_inputs("", "test user prompt")
        
        # Test empty user prompt
        with self.assertRaises(ValueError):
            self.wrapper._validate_inputs("test system prompt", "")
        
        # Test injection attempt
        with self.assertRaises(ValueError):
            self.wrapper._validate_inputs("test", "<script>alert('xss')</script>")
    
    def test_language_detection(self):
        """Test language detection functionality"""
        detector = LanguageDetector()
        
        # Test English detection
        self.assertEqual(detector.detect_language("Hello world"), "en")
        
        # Test Thai detection
        self.assertEqual(detector.detect_language("สวัสดี"), "th")
        
        # Test language name mapping
        self.assertEqual(detector.get_language_name("en"), "English")
        self.assertEqual(detector.get_language_name("th"), "Thai")
        self.assertEqual(detector.get_language_name("unknown"), "unknown")
    
    def test_rate_limiter(self):
        """Test rate limiting functionality"""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # Should allow first two calls
        self.assertTrue(limiter.can_proceed())
        limiter.record_call()
        
        self.assertTrue(limiter.can_proceed())
        limiter.record_call()
        
        # Third call should be blocked
        self.assertFalse(limiter.can_proceed())
        
        # Wait and try again
        time.sleep(1.1)
        self.assertTrue(limiter.can_proceed())
    
    @patch('openai.OpenAI')
    def test_successful_api_call(self, mock_openai):
        """Test successful API call"""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            wrapper = ChatGPTWrapper()
            response = wrapper.chat_with_gpt("test system", "test user")
            
            self.assertEqual(response, "Test response")
    
    @patch('openai.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """Test API error handling"""
        # Mock API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            wrapper = ChatGPTWrapper(config=ChatConfig(max_retries=1))
            response = wrapper.chat_with_gpt("test system", "test user")
            
            # Should return error message instead of raising exception
            self.assertIn("unexpected error", response.lower())

class TestPlannerUtils(unittest.TestCase):
    """Test cases for improved planner utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = PlannerConfig(
            max_tokens=150,
            temperature=0.8,
            enable_emojis=True,
            language="thai"
        )
        
        # Mock ChatGPT wrapper
        self.mock_wrapper = Mock()
        self.mock_wrapper.chat_with_gpt.return_value = "Test response"
        
        self.planner = PlannerUtils(config=self.config, wrapper=self.mock_wrapper)
    
    def test_planner_initialization(self):
        """Test planner initialization"""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.config.max_tokens, 150)
        self.assertEqual(self.planner.config.temperature, 0.8)
    
    def test_planner_validator(self):
        """Test planner validation"""
        validator = PlannerValidator()
        
        # Test valid planner data
        valid_data = {"tasks": ["task1", "task2"], "goals": ["goal1"]}
        self.assertTrue(validator.validate_planner_data(valid_data))
        
        # Test invalid planner data
        with self.assertRaises(ValueError):
            validator.validate_planner_data({})
        
        with self.assertRaises(ValueError):
            validator.validate_planner_data("not a dict")
        
        # Test language validation
        self.assertEqual(validator.validate_language("thai"), "thai")
        self.assertEqual(validator.validate_language("en"), "english")
        self.assertEqual(validator.validate_language(""), "thai")
        
        # Test user input validation
        self.assertEqual(validator.validate_user_input("valid input"), "valid input")
        
        with self.assertRaises(ValueError):
            validator.validate_user_input("")
        
        with self.assertRaises(ValueError):
            validator.validate_user_input("<script>alert('xss')</script>")
    
    def test_prompt_builder(self):
        """Test prompt building functionality"""
        builder = PromptBuilder()
        
        # Test summarize prompt
        planner_data = {"tasks": ["task1"], "goals": ["goal1"]}
        system_prompt, user_prompt = builder.build_summarize_prompt(planner_data, "thai")
        
        self.assertIn("Evo", system_prompt)
        self.assertIn("thai", user_prompt)
        self.assertIn("task1", user_prompt)
        
        # Test motivate prompt
        system_prompt, user_prompt = builder.build_motivate_prompt("test summary")
        self.assertIn("motivational advice", user_prompt)
        
        # Test progress prompt
        todo_data = {"task1": "completed", "task2": "pending"}
        system_prompt, user_prompt = builder.build_progress_prompt("update", "summary", todo_data)
        self.assertIn("task1: completed", user_prompt)
    
    def test_summarize_plan(self):
        """Test plan summarization"""
        planner_data = {"tasks": ["task1", "task2"], "goals": ["goal1"]}
        
        response = self.planner.summarize_plan(planner_data, "general", "thai")
        
        self.assertEqual(response, "Test response")
        self.mock_wrapper.chat_with_gpt.assert_called_once()
        
        # Test with invalid data
        with self.assertRaises(ValueError):
            self.planner.summarize_plan({}, "general", "thai")
    
    def test_motivate_user(self):
        """Test user motivation"""
        response = self.planner.motivate_user("test summary")
        
        self.assertEqual(response, "Test response")
        self.mock_wrapper.chat_with_gpt.assert_called_once()
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            self.planner.motivate_user("")
    
    def test_track_progress(self):
        """Test progress tracking"""
        todo_data = {"task1": "completed", "task2": "pending"}
        
        response = self.planner.track_progress("update", "summary", todo_data)
        
        self.assertEqual(response, "Test response")
        self.mock_wrapper.chat_with_gpt.assert_called_once()
    
    def test_respond_to_user_input(self):
        """Test user input response"""
        response = self.planner.respond_to_user_input("user message", "summary")
        
        self.assertEqual(response, "Test response")
        self.mock_wrapper.chat_with_gpt.assert_called_once()
    
    def test_mood_boost(self):
        """Test mood boosting"""
        response = self.planner.mood_boost("summary")
        
        self.assertEqual(response, "Test response")
        self.mock_wrapper.chat_with_gpt.assert_called_once()

class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear any existing config
        import config
        config._config = None
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_config_loading(self):
        """Test configuration loading"""
        config = get_config()
        
        self.assertEqual(config.environment.value, "development")
        self.assertTrue(config.debug)
        self.assertEqual(config.openai.api_key, "test-key")
        self.assertEqual(config.monitoring.log_level.value, "DEBUG")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test missing API key
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                get_config()
        
        # Test invalid temperature
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_TEMPERATURE': '3.0'  # Invalid value
        }):
            with self.assertRaises(ValueError):
                get_config()
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            config = get_config()
            config_dict = config.to_dict()
            
            self.assertIn("environment", config_dict)
            self.assertIn("openai", config_dict)
            self.assertIn("planner", config_dict)
            self.assertIn("firebase", config_dict)
            self.assertIn("security", config_dict)
            self.assertIn("monitoring", config_dict)

class TestIntegration(unittest.TestCase):
    """Integration tests for the improved system"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock environment for testing
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            self.config = get_config()
    
    def test_backward_compatibility(self):
        """Test backward compatibility functions"""
        # Test that old function signatures still work
        from chatgpt_wrapper import chat_with_gpt
        from planner_utils import summarize_plan, motivate_user, track_progress
        
        # These should not raise import errors
        self.assertIsNotNone(chat_with_gpt)
        self.assertIsNotNone(summarize_plan)
        self.assertIsNotNone(motivate_user)
        self.assertIsNotNone(track_progress)
    
    def test_error_handling_integration(self):
        """Test error handling across the system"""
        # Test with invalid inputs
        from planner_utils import get_default_planner
        
        planner = get_default_planner()
        
        # Should handle invalid data gracefully
        response = planner.summarize_plan({"invalid": "data"}, "general", "invalid_lang")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

def run_performance_test():
    """Run performance tests"""
    print("🚀 Running performance tests...")
    
    start_time = time.time()
    
    # Test language detection performance
    detector = LanguageDetector()
    for i in range(100):
        detector.detect_language(f"Test text {i}")
    
    detection_time = time.time() - start_time
    print(f"✅ Language detection: {detection_time:.3f}s for 100 calls")
    
    # Test rate limiter performance
    start_time = time.time()
    limiter = RateLimiter(max_calls=1000, time_window=60.0)
    for i in range(1000):
        limiter.can_proceed()
        limiter.record_call()
    
    rate_limit_time = time.time() - start_time
    print(f"✅ Rate limiter: {rate_limit_time:.3f}s for 1000 calls")
    
    # Test prompt building performance
    start_time = time.time()
    builder = PromptBuilder()
    planner_data = {"tasks": ["task1", "task2"], "goals": ["goal1"]}
    for i in range(100):
        builder.build_summarize_prompt(planner_data, "thai")
    
    prompt_time = time.time() - start_time
    print(f"✅ Prompt building: {prompt_time:.3f}s for 100 calls")

def main():
    """Main test runner"""
    print("🧪 Running comprehensive tests for improved ChatGPT wrapper and planner utilities")
    print("=" * 80)
    
    # Run unit tests
    print("\n📋 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n📊 Running performance tests...")
    run_performance_test()
    
    # Test configuration
    print("\n⚙️  Testing configuration...")
    try:
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            config = get_config()
            print(f"✅ Configuration loaded successfully for environment: {config.environment.value}")
            print(f"✅ OpenAI model: {config.openai.model}")
            print(f"✅ Default language: {config.planner.default_language}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 