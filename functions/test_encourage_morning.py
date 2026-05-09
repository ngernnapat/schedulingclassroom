#!/usr/bin/env python3
"""
Test script for the encourage_in_the_morning API endpoint.
Supports both Firebase emulator testing and direct function testing.
"""

import requests
import json
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Base URL for Firebase emulator (default)
EMULATOR_BASE_URL = "http://127.0.0.1:5001/schedule-optimization-d83ea/us-central1"

# Sample test data
SAMPLE_TODO_DATA = [
    {
        "title": "Morning standup meeting",
        "detail": "Discuss project progress with team",
        "start": "09:00"
    },
    {
        "title": "Work on feature implementation",
        "detail": "Complete the user authentication module",
        "start": "10:00"
    },
    {
        "title": "Lunch break",
        "detail": "Take a break and eat healthy",
        "start": "12:00"
    },
    {
        "title": "Code review",
        "detail": "Review pull requests from teammates",
        "start": "14:00"
    },
    {
        "title": "Documentation update",
        "detail": "Update API documentation",
        "start": "16:00"
    }
]


def test_via_emulator(base_url: str = EMULATOR_BASE_URL, language: str = "english"):
    """
    Test the encourage_in_the_morning endpoint via Firebase emulator.
    
    Run the emulator first:
        firebase emulators:start --only functions
    """
    print("🧪 Testing encourage_in_the_morning via Firebase Emulator")
    print("=" * 60)
    print(f"📡 Endpoint: {base_url}/encourage_in_the_morning")
    print()
    
    # Prepare request data
    request_data = {
        "today_todo_list_data": SAMPLE_TODO_DATA,
        "languageSelected": language
    }
    
    print("📤 Request Data:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    print()
    
    try:
        # Make POST request
        response = requests.post(
            f"{base_url}/encourage_in_the_morning",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"Response ====>: {response.text}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print("-" * 40)
            print("📬 Response:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            if data.get('data', {}).get('response'):
                print()
                print("💬 Morning Message:")
                print(data['data']['response'])
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to Firebase emulator")
        print()
        print("💡 Make sure to start the emulator first:")
        print("   cd functions && firebase emulators:start --only functions")
        print()
        print("   Or run direct test instead:")
        print("   python test_encourage_morning.py --direct")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_direct(language: str = "english"):
    """
    Test the function directly without using the emulator.
    This calls the underlying planner_utils function directly.
    """
    print("🧪 Testing encourage_in_the_morning (Direct Function Call)")
    print("=" * 60)
    
    try:
        # Import the planner utils
        from planner_utils import message_in_the_morning
        
        print("📤 Input Data:")
        print(json.dumps(SAMPLE_TODO_DATA, indent=2, ensure_ascii=False))
        print()
        print(f"🌐 Language: {language}")
        print()
        
        # Call the function directly
        print("⏳ Generating morning message...")
        response = message_in_the_morning(
            today_todo_list_data=SAMPLE_TODO_DATA,
            language=language
        )
        
        print()
        print("✅ Success!")
        print("-" * 40)
        print("💬 Morning Message:")
        print(response)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("💡 Make sure you have the required dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("   And set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_with_thai_data():
    """Test with Thai language and Thai todo items"""
    print("🇹🇭 Testing with Thai Language")
    print("=" * 60)
    
    thai_todo_data = [
        {
            "title": "ประชุมทีมเช้า",
            "detail": "พูดคุยความคืบหน้าโปรเจค",
            "start": "09:00"
        },
        {
            "title": "พัฒนาฟีเจอร์ใหม่",
            "detail": "ทำระบบยืนยันตัวตน",
            "start": "10:30"
        },
        {
            "title": "พักเที่ยง",
            "detail": "ทานอาหารกลางวัน",
            "start": "12:00"
        },
        {
            "title": "ทบทวนโค้ด",
            "detail": "ตรวจสอบ Pull Request",
            "start": "14:00"
        }
    ]
    
    try:
        from planner_utils import message_in_the_morning
        
        print("📤 Input Data (Thai):")
        print(json.dumps(thai_todo_data, indent=2, ensure_ascii=False))
        print()
        
        print("⏳ Generating morning message in Thai...")
        response = message_in_the_morning(
            today_todo_list_data=thai_todo_data,
            language="thai"
        )
        
        print()
        print("✅ Success!")
        print("-" * 40)
        print("💬 ข้อความตอนเช้า:")
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_edge_cases():
    """Test edge cases like empty data"""
    print("🔬 Testing Edge Cases")
    print("=" * 60)
    
    try:
        from planner_utils import message_in_the_morning
        
        # Test 1: Empty list
        print("\n1️⃣ Testing empty todo list:")
        response = message_in_the_morning([], "english")
        print(f"   Result: {response}")
        
        # Test 2: Single todo
        print("\n2️⃣ Testing single todo:")
        single_todo = [{"title": "Single task", "detail": "Only one task today", "start": "10:00"}]
        response = message_in_the_morning(single_todo, "english")
        print(f"   Result: {response}")
        
        # Test 3: Many todos
        print("\n3️⃣ Testing many todos (10 items):")
        many_todos = [
            {"title": f"Task {i}", "detail": f"Description {i}", "start": f"{8+i}:00"}
            for i in range(10)
        ]
        response = message_in_the_morning(many_todos, "english")
        print(f"   Result: {response[:200]}..." if len(response or '') > 200 else f"   Result: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def print_usage():
    """Print usage instructions"""
    print("""
Usage: python test_encourage_morning.py [OPTIONS]

Options:
    --emulator, -e    Test via Firebase emulator (default)
    --direct, -d      Test by calling the function directly
    --thai, -t        Test with Thai language data
    --edge, -x        Test edge cases
    --help, -h        Show this help message

Examples:
    # Start Firebase emulator first
    firebase emulators:start --only functions
    
    # Then run tests
    python test_encourage_morning.py              # Test via emulator
    python test_encourage_morning.py --direct     # Test function directly
    python test_encourage_morning.py --thai       # Test with Thai data
    python test_encourage_morning.py --edge       # Test edge cases

Environment:
    OPENAI_API_KEY    Required for direct function testing
    """)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test encourage_in_the_morning API")
    parser.add_argument("--emulator", "-e", action="store_true", help="Test via Firebase emulator")
    parser.add_argument("--direct", "-d", action="store_true", help="Test function directly")
    parser.add_argument("--thai", "-t", action="store_true", help="Test with Thai language")
    parser.add_argument("--edge", "-x", action="store_true", help="Test edge cases")
    parser.add_argument("--url", type=str, default=EMULATOR_BASE_URL, help="Custom emulator URL")
    parser.add_argument("--language", "-l", type=str, default="english", help="Language for testing")
    
    args = parser.parse_args()
    
    # Default to emulator test if no specific test is selected
    if not any([args.direct, args.thai, args.edge, args.emulator]):
        args.emulator = True
    
    if args.emulator:
        test_via_emulator(args.url, args.language)
    
    if args.direct:
        test_direct(args.language)
    
    if args.thai:
        test_with_thai_data()
    
    if args.edge:
        test_edge_cases()


if __name__ == "__main__":
    main()
