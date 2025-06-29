#!/usr/bin/env python3
"""
Test script for School Schedule Optimization Firebase Functions
"""

import json
import requests
import sys

def test_health_check(base_url: str) -> bool:
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health_check", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_schedule_info(base_url: str) -> bool:
    """Test the schedule info endpoint"""
    print("🔍 Testing schedule info endpoint...")
    try:
        response = requests.get(f"{base_url}/get_schedule_info", timeout=30)
        if response.status_code == 200:
            print("✅ Schedule info retrieved successfully")
            return True
        else:
            print(f"❌ Schedule info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Schedule info error: {e}")
        return False

def test_generate_schedule(base_url: str) -> bool:
    """Test the generate schedule endpoint"""
    print("🔍 Testing generate schedule endpoint...")
    
    test_data = {
        "n_teachers": 5,
        "grades": ["P1", "P2", "P3"],
        "pe_teacher": "T5",
        "pe_grades": ["P2", "P3"],
        "pe_day": 3,
        "n_pe_periods": 2,
        "start_hour": 8,
        "n_hours": 6,
        "lunch_hour": 4,
        "days_per_week": 5,
        "enable_pe_constraints": False,
        "homeroom_mode": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate_schedule",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Schedule generated successfully!")
                return True
            else:
                print(f"❌ Schedule generation failed: {data.get('error')}")
                return False
        elif response.status_code == 422:
            print("⚠️  No feasible solution found (acceptable)")
            return True
        else:
            print(f"❌ Schedule generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Schedule generation error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 School Schedule Optimization - Function Tests")
    
    base_url = "https://schedule-optimization-d83ea.cloudfunctions.net"
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    
    print(f"📍 Testing functions at: {base_url}")
    
    tests = [
        ("Health Check", lambda: test_health_check(base_url)),
        ("Schedule Info", lambda: test_schedule_info(base_url)),
        ("Generate Schedule", lambda: test_generate_schedule(base_url))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 