#!/usr/bin/env python3
"""
Client example for the School Schedule Optimization Google Cloud Function
This script demonstrates how to use the API from a Python client
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class ScheduleOptimizerClient:
    """Client for the School Schedule Optimization API"""
    
    def __init__(self, base_url: str):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the Google Cloud Function
                     e.g., 'https://your-project.cloudfunctions.net'
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ScheduleOptimizerClient/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        response = self.session.get(f"{self.base_url}/health_check")
        response.raise_for_status()
        return response.json()
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the API"""
        response = self.session.get(f"{self.base_url}/get_schedule_info")
        response.raise_for_status()
        return response.json()
    
    def generate_schedule(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a school schedule
        
        Args:
            parameters: Schedule generation parameters
            
        Returns:
            Generated schedule data
        """
        response = self.session.post(
            f"{self.base_url}/generate_schedule",
            json=parameters
        )
        response.raise_for_status()
        return response.json()

def print_schedule_summary(schedule_data: Dict[str, Any]):
    """Print a summary of the generated schedule"""
    if not schedule_data.get('success'):
        print(f"❌ Error: {schedule_data.get('error', 'Unknown error')}")
        return
    
    schedule = schedule_data.get('schedule', [])
    homeroom = schedule_data.get('homeroom', [])
    metadata = schedule_data.get('metadata', {})
    
    print("✅ Schedule generated successfully!")
    print(f"📊 Total assignments: {metadata.get('total_assignments', 0)}")
    print(f"🏠 Homeroom assignments: {metadata.get('homeroom_assignments', 0)}")
    
    # Group by teacher
    teacher_assignments = {}
    for assignment in schedule:
        teacher = assignment['Teacher']
        if teacher not in teacher_assignments:
            teacher_assignments[teacher] = []
        teacher_assignments[teacher].append(assignment)
    
    print(f"\n👨‍🏫 Teacher assignments:")
    for teacher, assignments in sorted(teacher_assignments.items()):
        print(f"  {teacher}: {len(assignments)} periods")
    
    # Group by grade
    grade_assignments = {}
    for assignment in schedule:
        grade = assignment['Grade']
        if grade not in grade_assignments:
            grade_assignments[grade] = []
        grade_assignments[grade].append(assignment)
    
    print(f"\n📚 Grade assignments:")
    for grade, assignments in sorted(grade_assignments.items()):
        print(f"  {grade}: {len(assignments)} periods")
    
    # Show homeroom assignments
    if homeroom:
        print(f"\n🏠 Homeroom assignments:")
        for assignment in homeroom:
            print(f"  {assignment['Teacher']} → {assignment['Grade']}")

def main():
    """Main function demonstrating API usage"""
    
    # Replace with your actual Google Cloud Function URL
    base_url = "https://schedule-optimization-436110.us-central1.run.app"
    
    # For local testing, you might use:
    # base_url = "http://localhost:5001/your-project/us-central1"
    
    client = ScheduleOptimizerClient(base_url)
    
    print("🏫 School Schedule Optimization Client")
    print("=" * 50)
    
    try:
        # 1. Health check
        print("1. Checking service health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Service: {health['service']}")
        print(f"   Version: {health['version']}")
        print(f"   Scheduler available: {health['scheduler_available']}")
        print()
        
        # 2. Get API information
        print("2. Getting API information...")
        info = client.get_schedule_info()
        print(f"   Description: {info['description']}")
        print(f"   Available endpoints: {len(info['endpoints'])}")
        print()
        
        # 3. Generate a sample schedule
        print("3. Generating sample schedule...")
        
        # Sample parameters for a small school
        sample_params = {
            'n_teachers': 8,
            'grades': ['P1', 'P2', 'P3', 'P4', 'P5'],
            'pe_teacher': 'T8',
            'pe_grades': ['P3', 'P4', 'P5'],
            'pe_day': 3,
            'n_pe_periods': 3,
            'start_hour': 8,
            'n_hours': 6,
            'lunch_hour': 4,
            'days_per_week': 5,
            'enable_pe_constraints': False,
            'homeroom_mode': 1
        }
        
        print("   Parameters:")
        for key, value in sample_params.items():
            print(f"     {key}: {value}")
        print()
        
        # Generate schedule with timing
        start_time = time.time()
        schedule_data = client.generate_schedule(sample_params)
        end_time = time.time()
        
        print(f"   Generation time: {end_time - start_time:.2f} seconds")
        print()
        
        # 4. Display results
        print("4. Schedule summary:")
        print_schedule_summary(schedule_data)
        
        # 5. Save to file (optional)
        output_file = "generated_schedule.json"
        with open(output_file, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        print(f"\n💾 Schedule saved to {output_file}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Could not connect to the API")
        print("   Make sure the Google Cloud Function is deployed and running")
        print(f"   URL: {base_url}")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        if e.response is not None:
            try:
                error_data = e.response.json()
                print(f"   Error details: {error_data}")
            except:
                print(f"   Response: {e.response.text}")
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 