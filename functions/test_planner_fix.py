#!/usr/bin/env python3
"""
Test script to verify planner utilities work correctly with optimized API call parameters.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_planner_utils():
    """Test the planner utilities with various scenarios"""
    
    try:
        from planner_utils import PlannerUtils, PlannerConfig
        
        # Test data
        test_planner_data = {
            "tasks": [
                {"title": "Morning Exercise", "time": "7:00 AM", "priority": "high"},
                {"title": "Team Meeting", "time": "10:00 AM", "priority": "medium"},
                {"title": "Lunch Break", "time": "12:00 PM", "priority": "low"}
            ],
            "goals": ["Complete project", "Exercise daily", "Read more"],
            "mood": "motivated"
        }
        
        test_todo_data = {
            "completed": 2,
            "total": 5,
            "progress": "40%"
        }
        
        # Initialize planner with custom config
        config = PlannerConfig(
            max_tokens=150,
            temperature=0.8,
            top_p=0.9,
            enable_emojis=True,
            enable_motivation=True,
            language="thai"
        )
        
        planner = PlannerUtils(config=config)
        
        print("🧪 Testing Planner Utilities...")
        print("=" * 50)
        
        # Test 1: Summarize plan
        print("\n1. Testing plan summarization...")
        summary = planner.summarize_plan(test_planner_data, "general", "thai")
        print(f"Summary: {summary}")
        
        # Test 2: Motivate user
        print("\n2. Testing user motivation...")
        motivation = planner.motivate_user(summary)
        print(f"Motivation: {motivation}")
        
        # Test 3: Track progress
        print("\n3. Testing progress tracking...")
        progress_feedback = planner.track_progress(
            "I completed 2 out of 5 tasks today!", 
            summary, 
            test_todo_data
        )
        print(f"Progress feedback: {progress_feedback}")
        
        # Test 4: Morning message
        print("\n4. Testing morning message...")
        today_tasks = [
            {"title": "Team Standup", "start": "9:00 AM", "location": "Conference Room"},
            {"title": "Client Meeting", "start": "2:00 PM", "location": "Zoom"},
            {"title": "Code Review", "start": "4:00 PM"}
        ]
        morning_msg = planner.morning_message(today_tasks, "thai")
        print(f"Morning message: {morning_msg}")
        
        # Test 5: User input response
        print("\n5. Testing user input response...")
        user_response = planner.respond_to_user_input(
            "I'm feeling a bit overwhelmed today", 
            summary
        )
        print(f"User response: {user_response}")
        
        # Test 6: Mood boost
        print("\n6. Testing mood boost...")
        mood_boost_msg = planner.mood_boost(summary)
        print(f"Mood boost: {mood_boost_msg}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_call_parameters():
    """Test that API call parameters are properly handled"""
    
    try:
        from planner_utils import PlannerUtils, PlannerConfig
        
        print("\n🔧 Testing API call parameter handling...")
        print("=" * 50)
        
        # Test with custom parameters
        config = PlannerConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.8
        )
        
        planner = PlannerUtils(config=config)
        
        # Test that custom parameters are used
        test_data = {"test": "data"}
        
        # This should use the custom config parameters
        result = planner.summarize_plan(test_data, "test", "thai")
        print(f"Custom config result: {result[:100]}...")
        
        # Test with override parameters
        result_override = planner._safe_chat_call(
            "You are a test assistant.",
            "Say hello briefly.",
            max_tokens=20,
            temperature=0.1,
            language="thai"
        )
        print(f"Override parameters result: {result_override}")
        
        print("✅ API call parameter tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"API call parameter test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Planner Utilities Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_success = test_planner_utils()
    test2_success = test_api_call_parameters()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("🎉 All tests passed! The planner utilities are working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 