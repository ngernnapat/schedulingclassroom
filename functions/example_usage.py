#!/usr/bin/env python3
"""
Example usage of improved ChatGPT wrapper and planner utilities
"""

import os
import logging
from typing import Dict, Any

# Import the improved modules
from chatgpt_wrapper import ChatGPTWrapper, ChatConfig, get_default_wrapper
from planner_utils import PlannerUtils, PlannerConfig, get_default_planner
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Example of basic usage (backward compatible)"""
    print("🔧 Example 1: Basic Usage (Backward Compatible)")
    print("-" * 50)
    
    # Set a mock API key for demonstration
    os.environ["OPENAI_API_KEY"] = "demo-key"
    
    try:
        from chatgpt_wrapper import chat_with_gpt
        from planner_utils import summarize_plan, motivate_user
        
        # These work exactly as before
        response = chat_with_gpt("You are a helpful assistant.", "Hello!")
        print(f"ChatGPT Response: {response}")
        
        # Example planner data
        planner_data = {
            "tasks": ["Complete project", "Exercise", "Read book"],
            "goals": ["Improve productivity", "Stay healthy"],
            "schedule": {"morning": "Exercise", "afternoon": "Work", "evening": "Relax"}
        }
        
        summary = summarize_plan(planner_data, "general", "thai")
        print(f"Plan Summary: {summary}")
        
        motivation = motivate_user("Working on improving productivity and health")
        print(f"Motivation: {motivation}")
        
    except Exception as e:
        print(f"❌ Error in basic usage: {e}")
        print("(This is expected without a real API key)")

def example_advanced_usage():
    """Example of advanced usage with custom configuration"""
    print("\n🚀 Example 2: Advanced Usage with Custom Configuration")
    print("-" * 50)
    
    try:
        # Custom ChatGPT configuration
        chat_config = ChatConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=150,
            max_retries=3,
            timeout=20
        )
        
        # Custom planner configuration
        planner_config = PlannerConfig(
            max_tokens=200,
            temperature=0.8,
            enable_emojis=True,
            language="english"
        )
        
        # Create instances with custom configs
        wrapper = ChatGPTWrapper(config=chat_config)
        planner = PlannerUtils(config=planner_config, wrapper=wrapper)
        
        print(f"✅ ChatGPT wrapper initialized with model: {chat_config.model}")
        print(f"✅ Planner initialized with language: {planner_config.language}")
        
        # Example usage
        planner_data = {
            "daily_tasks": ["Meditation", "Workout", "Learning"],
            "weekly_goals": ["Complete 3 projects", "Exercise 5 times"],
            "monthly_targets": ["Read 4 books", "Save $1000"]
        }
        
        # Use the planner with custom settings
        summary = planner.summarize_plan(planner_data, "detailed", "english")
        print(f"Custom Summary: {summary}")
        
    except Exception as e:
        print(f"❌ Error in advanced usage: {e}")
        print("(This is expected without a real API key)")

def example_error_handling():
    """Example of error handling improvements"""
    print("\n🛡️ Example 3: Error Handling Improvements")
    print("-" * 50)
    
    try:
        # Test with invalid inputs
        planner = get_default_planner()
        
        # This should handle gracefully
        response = planner.summarize_plan({"invalid": "data"}, "general", "invalid_lang")
        print(f"✅ Graceful handling of invalid data: {response}")
        
        # Test input validation
        from planner_utils import PlannerValidator
        validator = PlannerValidator()
        
        # Valid input
        valid_lang = validator.validate_language("thai")
        print(f"✅ Language validation: 'thai' -> '{valid_lang}'")
        
        # Invalid input handling
        try:
            validator.validate_user_input("")
            print("❌ Should have raised ValueError for empty input")
        except ValueError:
            print("✅ Correctly rejected empty input")
        
        # Security validation
        try:
            validator.validate_user_input("<script>alert('xss')</script>")
            print("❌ Should have raised ValueError for suspicious input")
        except ValueError:
            print("✅ Correctly rejected suspicious input")
            
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")

def example_configuration():
    """Example of configuration management"""
    print("\n⚙️ Example 4: Configuration Management")
    print("-" * 50)
    
    try:
        # Load configuration
        config = get_config()
        
        print(f"✅ Environment: {config.environment.value}")
        print(f"✅ Debug mode: {config.debug}")
        print(f"✅ OpenAI model: {config.openai.model}")
        print(f"✅ Default language: {config.planner.default_language}")
        print(f"✅ Max retries: {config.openai.max_retries}")
        print(f"✅ Rate limit calls: {config.openai.rate_limit_calls}")
        
        # Show configuration as dictionary
        config_dict = config.to_dict()
        print(f"✅ Configuration loaded successfully with {len(config_dict)} sections")
        
    except Exception as e:
        print(f"❌ Error in configuration example: {e}")

def example_performance_features():
    """Example of performance features"""
    print("\n⚡ Example 5: Performance Features")
    print("-" * 50)
    
    try:
        from chatgpt_wrapper import LanguageDetector, RateLimiter
        
        # Test language detection with caching
        detector = LanguageDetector()
        
        # First call (will detect)
        start_time = time.time()
        lang1 = detector.detect_language("Hello world")
        first_call_time = time.time() - start_time
        
        # Second call (will use cache)
        start_time = time.time()
        lang2 = detector.detect_language("Hello world")
        second_call_time = time.time() - start_time
        
        print(f"✅ Language detection: {lang1}")
        print(f"✅ First call: {first_call_time:.4f}s")
        print(f"✅ Cached call: {second_call_time:.4f}s")
        print(f"✅ Cache speedup: {first_call_time/second_call_time:.1f}x faster")
        
        # Test rate limiter
        limiter = RateLimiter(max_calls=3, time_window=5.0)
        
        print(f"✅ Rate limiter allows call: {limiter.can_proceed()}")
        limiter.record_call()
        print(f"✅ Rate limiter allows call: {limiter.can_proceed()}")
        limiter.record_call()
        print(f"✅ Rate limiter allows call: {limiter.can_proceed()}")
        limiter.record_call()
        print(f"✅ Rate limiter allows call: {limiter.can_proceed()}")
        
    except Exception as e:
        print(f"❌ Error in performance example: {e}")

def main():
    """Main example runner"""
    print("🎯 ChatGPT Wrapper and Planner Utilities - Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_basic_usage()
    example_advanced_usage()
    example_error_handling()
    example_configuration()
    example_performance_features()
    
    print("\n" + "=" * 70)
    print("🎉 All examples completed!")
    print("\n📚 For more information, see IMPROVEMENTS.md")
    print("🧪 To run tests, use: python test_improvements.py")

if __name__ == "__main__":
    import time
    main() 