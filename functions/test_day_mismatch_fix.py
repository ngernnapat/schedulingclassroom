#!/usr/bin/env python3
"""
Test script to verify the day mismatch handling fix
"""

import json
import sys
import os

# Mock the environment variable to avoid OpenAI initialization
os.environ["OPENAI_API_KEY"] = "test-key"

# Add the current directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_planner_content import PlannerGenerationError, _process_and_validate_response

def test_day_mismatch_handling():
    """Test that day count mismatches are handled gracefully"""
    print("🧪 Testing day mismatch handling...")
    
    # Test case 1: Fewer days than requested
    print("\n1. Testing fewer days than requested...")
    request_data = {
        "planName": "Test Plan",
        "category": "study",
        "totalDays": 5,
        "minutesPerDay": 60
    }
    
    # Mock response with only 3 days instead of 5
    mock_response_data = {
        "planName": "Test Plan",
        "category": "study",
        "totalDays": 5,
        "minutesPerDay": 60,
        "createdAt": "2024-01-01T00:00:00Z",
        "days": [
            {
                "title": "Day 1",
                "summary": "First day tasks",
                "tasks": [
                    {"title": "Task 1", "duration_min": 30, "done": False},
                    {"title": "Task 2", "duration_min": 30, "done": False}
                ]
            },
            {
                "title": "Day 2", 
                "summary": "Second day tasks",
                "tasks": [
                    {"title": "Task 1", "duration_min": 60, "done": False}
                ]
            },
            {
                "title": "Day 3",
                "summary": "Third day tasks", 
                "tasks": [
                    {"title": "Task 1", "duration_min": 60, "done": False}
                ]
            }
        ]
    }
    
    try:
        result = _process_and_validate_response(mock_response_data, request_data)
        print(f"✅ Fewer days test passed!")
        print(f"   - Result has {result.totalDays} days (expected 3)")
        print(f"   - Warning: {result.warning if hasattr(result, 'warning') and result.warning else 'None'}")
        assert result.totalDays == 3, f"Expected 3 days, got {result.totalDays}"
        assert hasattr(result, 'warning') and result.warning is not None, "Expected warning message"
    except Exception as e:
        print(f"❌ Fewer days test failed: {e}")
        return False
    
    # Test case 2: More days than requested
    print("\n2. Testing more days than requested...")
    
    # Add 2 more days to the mock response
    mock_response_data["days"].extend([
        {
            "title": "Day 4",
            "summary": "Fourth day tasks",
            "tasks": [{"title": "Task 1", "duration_min": 60, "done": False}]
        },
        {
            "title": "Day 5",
            "summary": "Fifth day tasks", 
            "tasks": [{"title": "Task 1", "duration_min": 60, "done": False}]
        }
    ])
    
    try:
        result = _process_and_validate_response(mock_response_data, request_data)
        print(f"✅ More days test passed!")
        print(f"   - Result has {result.totalDays} days (expected 5)")
        print(f"   - Warning: {result.warning if hasattr(result, 'warning') and result.warning else 'None'}")
        assert result.totalDays == 5, f"Expected 5 days, got {result.totalDays}"
        assert hasattr(result, 'warning') and result.warning is not None, "Expected warning message"
    except Exception as e:
        print(f"❌ More days test failed: {e}")
        return False
    
    # Test case 3: Exact match (no warning expected)
    print("\n3. Testing exact day match...")
    
    # Reset to exactly 5 days
    mock_response_data["days"] = mock_response_data["days"][:5]
    
    try:
        result = _process_and_validate_response(mock_response_data, request_data)
        print(f"✅ Exact match test passed!")
        print(f"   - Result has {result.totalDays} days (expected 5)")
        print(f"   - Warning: {result.warning if hasattr(result, 'warning') and result.warning else 'None'}")
        assert result.totalDays == 5, f"Expected 5 days, got {result.totalDays}"
        assert not hasattr(result, 'warning') or result.warning is None, "No warning expected for exact match"
    except Exception as e:
        print(f"❌ Exact match test failed: {e}")
        return False
    
    print("\n🎉 All day mismatch handling tests passed!")
    return True

if __name__ == "__main__":
    success = test_day_mismatch_handling()
    sys.exit(0 if success else 1)
