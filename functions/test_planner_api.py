#!/usr/bin/env python3
"""
Test script for the Planner Content Generation API.
This script demonstrates how to use the local API to test planner generation.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_categories():
    """Test the categories endpoint"""
    print("📋 Testing categories endpoint...")
    response = requests.get(f"{BASE_URL}/categories")
    print(f"Status: {response.status_code}")
    categories = response.json()
    for cat in categories["categories"]:
        print(f"  - {cat['value']}: {cat['description']}")
    print()

def test_examples():
    """Test the examples endpoint"""
    print("📚 Testing examples endpoint...")
    response = requests.get(f"{BASE_URL}/examples")
    print(f"Status: {response.status_code}")
    examples = response.json()
    for i, example in enumerate(examples["examples"], 1):
        print(f"  Example {i}: {example['name']}")
        print(f"    Category: {example['request']['category']}")
        print(f"    Days: {example['request']['totalDays']}")
    print()

def test_quick_generation(category: str = "learning", days: int = 7):
    """Test quick generation using the test endpoint"""
    print(f"⚡ Testing quick generation for {category} ({days} days)...")
    response = requests.get(f"{BASE_URL}/test/{category}?days={days}&minutes=30")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Generated {data['days']} days")
        print(f"Plan name: {data['content']['planName']}")
        print(f"Category: {data['content']['category']}")
        print(f"First day title: {data['content']['days'][0]['title']}")
        print(f"First day tasks: {len(data['content']['days'][0]['tasks'])}")
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_full_generation():
    """Test full generation using the main endpoint"""
    print("🚀 Testing full generation...")
    
    request_data = {
        "planName": "Python Learning Journey",
        "category": "learning",
        "totalDays": 5,
        "detailPrompt": "Learn Python programming from basics to intermediate level",
        "minutesPerDay": 60,
        "intensity": "moderate",
        "language": "en"
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! {data['message']}")
        
        content = data['data']
        print(f"Plan: {content['planName']}")
        print(f"Category: {content['category']}")
        print(f"Total days: {content['totalDays']}")
        print(f"Minutes per day: {content.get('minutesPerDay', 'Not specified')}")
        
        # Show first day details
        if content['days']:
            first_day = content['days'][0]
            print(f"\nFirst day: {first_day['title']}")
            print(f"Summary: {first_day['summary']}")
            print("Tasks:")
            for i, task in enumerate(first_day['tasks'], 1):
                duration = f" ({task.get('duration_min', 'N/A')} min)" if task.get('duration_min') else ""
                print(f"  {i}. {task['text']}{duration}")
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_thai_generation():
    """Test Thai language generation"""
    print("🇹🇭 Testing Thai language generation...")
    
    request_data = {
        "planName": "เรียนภาษาไทย 7 วัน",
        "category": "learning",
        "totalDays": 7,
        "detailPrompt": "เรียนภาษาไทยพื้นฐานสำหรับคนต่างชาติ",
        "minutesPerDay": 30,
        "language": "th"
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! {data['message']}")
        
        content = data['data']
        print(f"Plan: {content['planName']}")
        print(f"Language: Thai")
        
        # Show first day details
        if content['days']:
            first_day = content['days'][0]
            print(f"\nFirst day: {first_day['title']}")
            print(f"Summary: {first_day['summary']}")
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_error_handling():
    """Test error handling with invalid requests"""
    print("🛡️ Testing error handling...")
    
    # Test invalid category
    print("Testing invalid category...")
    response = requests.get(f"{BASE_URL}/test/invalid_category")
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"✅ Correctly rejected: {response.json()}")
    
    # Test invalid days
    print("\nTesting invalid days...")
    response = requests.post(f"{BASE_URL}/generate", json={
        "planName": "Test",
        "category": "learning",
        "totalDays": 100,  # Too many days
        "language": "en"
    })
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"✅ Correctly rejected: {response.json()}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 Starting Planner Content Generation API Tests")
    print("=" * 50)
    
    try:
        # Basic endpoint tests
        test_health()
        test_categories()
        test_examples()
        
        # Generation tests
        test_quick_generation("learning", 3)
        test_quick_generation("exercise", 5)
        test_full_generation()
        test_thai_generation()
        
        # Error handling tests
        test_error_handling()
        
        print("🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
        print("Start it with: python generate_planner_content_api.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
