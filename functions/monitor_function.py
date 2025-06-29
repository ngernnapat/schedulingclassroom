#!/usr/bin/env python3
"""
Monitoring script for School Schedule Optimization Firebase Functions
This script helps identify issues that might be causing ERROR logs.
"""

import json
import requests
import sys
import time
from typing import Dict, Any

def test_endpoint(base_url: str, endpoint: str, method: str = 'GET', data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test a specific endpoint and return detailed results"""
    url = f"{base_url}/{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=30)
        elif method == 'POST':
            response = requests.post(
                url, 
                json=data, 
                headers={'Content-Type': 'application/json'}, 
                timeout=60
            )
        else:
            return {'error': f'Unsupported method: {method}'}
        
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content_type': response.headers.get('content-type', 'unknown'),
            'response_size': len(response.content),
            'response_time': response.elapsed.total_seconds(),
            'success': 200 <= response.status_code < 300,
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:500]
        }
        
    except requests.exceptions.Timeout:
        return {'error': 'Request timeout', 'timeout': True}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection error', 'connection_error': True}
    except Exception as e:
        return {'error': f'Request failed: {str(e)}', 'exception': True}

def test_health_check(base_url: str) -> Dict[str, Any]:
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    result = test_endpoint(base_url, 'health_check')
    
    if result.get('success'):
        print(f"✅ Health check passed: {result['status_code']}")
        if 'data' in result and isinstance(result['data'], dict):
            print(f"   - Service: {result['data'].get('service', 'unknown')}")
            print(f"   - Scheduler available: {result['data'].get('scheduler_available', 'unknown')}")
            print(f"   - Python version: {result['data'].get('python_version', 'unknown')}")
    else:
        print(f"❌ Health check failed: {result}")
    
    return result

def test_debug_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the debug endpoint"""
    print("🔍 Testing debug endpoint...")
    result = test_endpoint(base_url, 'debug')
    
    if result.get('success') and 'data' in result:
        data = result['data']
        print(f"✅ Debug info retrieved: {result['status_code']}")
        
        # Check import status
        if 'import_status' in data:
            print("   📦 Import Status:")
            for lib, status in data['import_status'].items():
                if status.get('available'):
                    print(f"     ✅ {lib}: {status.get('version', 'unknown')}")
                else:
                    print(f"     ❌ {lib}: {status.get('error', 'unknown error')}")
        
        # Check scheduler status
        if 'scheduler_status' in data:
            print("   🎯 Scheduler Status:")
            scheduler_status = data['scheduler_status']
            if scheduler_status.get('import'):
                print("     ✅ Import: Success")
                if scheduler_status.get('instantiation'):
                    print("     ✅ Instantiation: Success")
                if scheduler_status.get('get_inputs'):
                    print("     ✅ get_inputs: Success")
                if scheduler_status.get('get_model'):
                    print("     ✅ get_model: Success")
                if 'error' in scheduler_status:
                    print(f"     ❌ Error: {scheduler_status['error']}")
            else:
                print("     ❌ Import: Failed")
        
        # Check memory info
        if 'memory_info' in data:
            memory_info = data['memory_info']
            if memory_info.get('available'):
                print(f"   💾 Memory: {memory_info.get('memory_percent', 'unknown')}% used")
                print(f"     Available: {memory_info.get('memory_available_mb', 'unknown')} MB")
            else:
                print("   💾 Memory: psutil not available")
    
    return result

def test_schedule_generation(base_url: str, test_cases: list) -> Dict[str, Any]:
    """Test schedule generation with various test cases"""
    print("🔍 Testing schedule generation...")
    
    results = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['description']}")
        result = test_endpoint(base_url, 'generate_schedule', 'POST', test_case['data'])
        
        if result.get('success'):
            print(f"     ✅ Success: {result['status_code']}")
            if 'data' in result and isinstance(result['data'], dict):
                if result['data'].get('success'):
                    print(f"       Schedule generated with {result['data'].get('metadata', {}).get('total_assignments', 0)} assignments")
                else:
                    print(f"       Error: {result['data'].get('error', 'unknown')}")
        elif result.get('status_code') == 422:
            print(f"     ⚠️  No feasible solution (acceptable)")
        else:
            print(f"     ❌ Failed: {result}")
        
        results[f"test_case_{i+1}"] = result
        time.sleep(1)  # Small delay between requests
    
    return results

def main():
    """Main monitoring function"""
    print("🔧 School Schedule Optimization - Function Monitor")
    print("=" * 60)
    
    # base_url = "https://schedule-optimization-d83ea.cloudfunctions.net"
    # if len(sys.argv) > 1:
    #     base_url = sys.argv[1].rstrip('/')
    
    # print(f"📍 Monitoring functions at: {base_url}")
    # print()
    
    # Test cases for schedule generation
    test_cases = [
        {
            'description': 'Simple case (3 teachers, 2 grades)',
            'data': {
                'n_teachers': 3,
                'grades': ['P1', 'P2'],
                'pe_teacher': 'T3',
                'pe_grades': ['P2'],
                'pe_day': 3,
                'n_pe_periods': 1,
                'start_hour': 8,
                'n_hours': 4,
                'lunch_hour': 3,
                'days_per_week': 3,
                'enable_pe_constraints': False,
                'homeroom_mode': 1
            }
        },
        {
            'description': 'Medium case (5 teachers, 3 grades)',
            'data': {
                'n_teachers': 5,
                'grades': ['P1', 'P2', 'P3'],
                'pe_teacher': 'T5',
                'pe_grades': ['P2', 'P3'],
                'pe_day': 3,
                'n_pe_periods': 2,
                'start_hour': 8,
                'n_hours': 6,
                'lunch_hour': 4,
                'days_per_week': 5,
                'enable_pe_constraints': False,
                'homeroom_mode': 1
            }
        }
    ]
    
    # Run tests
    results = {
        'health_check': test_health_check("https://health-check-rvnln6uc4a-uc.a.run.app/health_check"),
        'debug': test_debug_endpoint("https://debug-rvnln6uc4a-uc.a.run.app/debug"),
        'schedule_generation': test_schedule_generation("https://generate-schedule-rvnln6uc4a-uc.a.run.app/generate_schedule", test_cases)
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MONITORING SUMMARY")
    print("=" * 60)
    
    health_success = results['health_check'].get('success', False)
    debug_success = results['debug'].get('success', False)
    
    print(f"Health Check: {'✅ PASS' if health_success else '❌ FAIL'}")
    print(f"Debug Endpoint: {'✅ PASS' if debug_success else '❌ FAIL'}")
    
    # Check for potential issues
    issues = []
    
    if not health_success:
        issues.append("Health check failed - function may be down")
    
    if not debug_success:
        issues.append("Debug endpoint failed - may indicate import issues")
    
    if debug_success and 'data' in results['debug']:
        data = results['debug']['data']
        
        # Check import issues
        if 'import_status' in data:
            for lib, status in data['import_status'].items():
                if not status.get('available'):
                    issues.append(f"Import issue: {lib} not available")
        
        # Check scheduler issues
        if 'scheduler_status' in data:
            scheduler_status = data['scheduler_status']
            if not scheduler_status.get('import'):
                issues.append("SchoolScheduler import failed")
            elif 'error' in scheduler_status:
                issues.append(f"Scheduler error: {scheduler_status['error']}")
        
        # Check memory issues
        if 'memory_info' in data:
            memory_info = data['memory_info']
            if memory_info.get('available') and memory_info.get('memory_percent', 0) > 90:
                issues.append("High memory usage detected")
    
    if issues:
        print("\n⚠️  POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("\n✅ No obvious issues detected")
    
    # Save results to file
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # filename = f"monitoring_results_{timestamp}.json"
    
    # try:
    #     with open(filename, 'w') as f:
    #         json.dump(results, f, indent=2, default=str)
    #     print(f"\n📄 Results saved to: {filename}")
    # except Exception as e:
    #     print(f"\n❌ Failed to save results: {e}")
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main()) 