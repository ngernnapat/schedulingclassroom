#!/usr/bin/env python3
"""
Demo script that shows the API structure and example requests
without requiring an actual OpenAI API key.
This is useful for understanding the API before setting up the real key.
"""

import json
from typing import Dict, Any

def show_api_structure():
    """Show the API structure and endpoints"""
    print("🚀 Planner Content Generation API Structure")
    print("=" * 50)
    
    endpoints = {
        "GET /": "API information and available endpoints",
        "GET /health": "Health check endpoint",
        "GET /categories": "List available plan categories",
        "GET /examples": "Get example requests for testing",
        "POST /generate": "Generate planner content (main endpoint)",
        "POST /generate-raw": "Generate with raw JSON input",
        "GET /test/{category}": "Quick test generation"
    }
    
    for endpoint, description in endpoints.items():
        print(f"  {endpoint:<20} - {description}")
    
    print()

def show_categories():
    """Show available categories"""
    print("📋 Available Plan Categories")
    print("=" * 30)
    
    categories = [
        ("learning", "Skill acquisition and knowledge development"),
        ("exercise", "Physical fitness and health"),
        ("travel", "Trip planning and itinerary"),
        ("finance", "Financial management and literacy"),
        ("health", "Holistic wellness and healthy habits"),
        ("personal_development", "Self-improvement and growth"),
        ("other", "Custom plan based on specific needs")
    ]
    
    for category, description in categories:
        print(f"  {category:<20} - {description}")
    
    print()

def show_example_requests():
    """Show example requests"""
    print("📚 Example API Requests")
    print("=" * 25)
    
    examples = [
        {
            "name": "Basic Learning Plan",
            "method": "POST",
            "endpoint": "/generate",
            "data": {
                "planName": "Python Programming Basics",
                "category": "learning",
                "totalDays": 7,
                "detailPrompt": "Beginner Python programming with focus on data structures",
                "minutesPerDay": 60,
                "language": "en"
            }
        },
        {
            "name": "Exercise Plan",
            "method": "POST",
            "endpoint": "/generate",
            "data": {
                "planName": "30-Day Fitness Challenge",
                "category": "exercise",
                "totalDays": 30,
                "detailPrompt": "Home workout routine for beginners, no equipment needed",
                "minutesPerDay": 45,
                "intensity": "moderate",
                "language": "en"
            }
        },
        {
            "name": "Quick Test",
            "method": "GET",
            "endpoint": "/test/learning?days=7&minutes=30",
            "data": None
        },
        {
            "name": "Thai Language Learning",
            "method": "POST",
            "endpoint": "/generate",
            "data": {
                "planName": "เรียนภาษาไทย 30 วัน",
                "category": "learning",
                "totalDays": 30,
                "detailPrompt": "Basic Thai language learning for English speakers",
                "minutesPerDay": 30,
                "language": "th"
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Method: {example['method']}")
        print(f"   Endpoint: {example['endpoint']}")
        if example['data']:
            print(f"   Request Body:")
            print(f"   {json.dumps(example['data'], indent=6)}")
        print()

def show_curl_examples():
    """Show curl command examples"""
    print("🌐 cURL Command Examples")
    print("=" * 25)
    
    print("1. Health Check:")
    print("   curl http://localhost:8000/health")
    print()
    
    print("2. Get Categories:")
    print("   curl http://localhost:8000/categories")
    print()
    
    print("3. Quick Test Generation:")
    print("   curl \"http://localhost:8000/test/learning?days=7&minutes=30\"")
    print()
    
    print("4. Full Generation Request:")
    print("   curl -X POST \"http://localhost:8000/generate\" \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{")
    print("       \"planName\": \"Python Learning Journey\",")
    print("       \"category\": \"learning\",")
    print("       \"totalDays\": 7,")
    print("       \"detailPrompt\": \"Learn Python programming from basics\",")
    print("       \"minutesPerDay\": 60,")
    print("       \"language\": \"en\"")
    print("     }'")
    print()

def show_response_format():
    """Show expected response format"""
    print("📊 Expected Response Format")
    print("=" * 30)
    
    example_response = {
        "success": True,
        "data": {
            "planName": "Python Learning Journey",
            "category": "learning",
            "totalDays": 7,
            "minutesPerDay": 60,
            "createdAt": {
                "seconds": 1703123456,
                "nanoseconds": 0
            },
            "days": [
                {
                    "id": "abc12345",
                    "dayNumber": 1,
                    "title": "Python Basics Introduction",
                    "summary": "Get started with Python fundamentals",
                    "tasks": [
                        {
                            "id": "task123",
                            "text": "Install Python and set up development environment",
                            "done": False,
                            "duration_min": 20,
                            "note": "Use Python 3.8 or later"
                        },
                        {
                            "id": "task456",
                            "text": "Write your first Python program",
                            "done": False,
                            "duration_min": 25,
                            "note": "Start with 'Hello World'"
                        },
                        {
                            "id": "task789",
                            "text": "Learn about variables and data types",
                            "done": False,
                            "duration_min": 15,
                            "note": "Focus on strings, numbers, and booleans"
                        }
                    ],
                    "tips": "Take breaks every 20 minutes to avoid fatigue"
                }
            ]
        },
        "message": "Successfully generated 7-day learning plan: Python Learning Journey"
    }
    
    print("Successful Response:")
    print(json.dumps(example_response, indent=2))
    print()

def show_setup_instructions():
    """Show setup instructions"""
    print("🔧 Setup Instructions")
    print("=" * 20)
    
    print("1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("2. Set up OpenAI API Key:")
    print("   python setup_api_key.py")
    print("   # OR")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("   # OR create .env file with:")
    print("   echo 'OPENAI_API_KEY=your-api-key-here' > .env")
    print()
    
    print("3. Start the API Server:")
    print("   python generate_planner_content_api.py")
    print("   # OR")
    print("   ./start_planner_api.sh")
    print()
    
    print("4. Test the API:")
    print("   python test_planner_api.py")
    print()
    
    print("5. Access Documentation:")
    print("   http://localhost:8000/docs")
    print()

def main():
    """Main demo function"""
    print("🎯 Planner Content Generation API - Demo & Documentation")
    print("=" * 60)
    print()
    
    show_api_structure()
    show_categories()
    show_example_requests()
    show_curl_examples()
    show_response_format()
    show_setup_instructions()
    
    print("🎉 That's it! You now understand the API structure.")
    print("Set up your OpenAI API key and start the server to begin testing!")

if __name__ == "__main__":
    main()
