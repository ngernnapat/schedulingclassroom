"""
Local API Server for Testing (No Firebase)
Run with: python local_api.py
"""

import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy-loaded modules
_planner_utils = None
_generate_planner_content = None
_school_scheduler = None

def get_planner_utils():
    global _planner_utils
    if _planner_utils is None:
        import planner_utils as pu
        _planner_utils = pu
    return _planner_utils

def get_generate_planner_content():
    global _generate_planner_content
    if _generate_planner_content is None:
        import generate_planner_content as gpc
        _generate_planner_content = gpc
    return _generate_planner_content

def get_school_scheduler():
    global _school_scheduler
    if _school_scheduler is None:
        from school_scheduler import SchoolScheduler
        _school_scheduler = SchoolScheduler
    return _school_scheduler

# Constants
DEFAULT_SCHEDULE_PARAMS = {
    'pe_teacher': 'T13',
    'pe_grades': ['P4', 'P5', 'P6', 'M1', 'M2', 'M3'],
    'pe_day': 3,
    'n_pe_periods': 6,
    'start_hour': 8,
    'n_hours': 8,
    'lunch_hour': 5,
    'days_per_week': 5,
    'enable_pe_constraints': False,
    'homeroom_mode': 1
}

def create_response(data=None, success=True, message="Success", error=None, status_code=200, metadata=None):
    """Create standardized response"""
    response = {
        'success': success,
        'message': message,
        'data': data,
        'error': error,
        'metadata': metadata
    }
    return jsonify(response), status_code


# ==================== HEALTH CHECK ====================
@app.route('/health', methods=['GET'])
def health_check():
    return create_response(
        data={'status': 'healthy', 'timestamp': time.time()},
        message='API is running'
    )


# ==================== SCHEDULE ENDPOINTS ====================
@app.route('/generate_schedule', methods=['POST'])
def generate_schedule():
    """Generate a school schedule"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return create_response(success=False, message='No JSON data provided', status_code=400)
        
        # Set defaults
        for field, default_value in DEFAULT_SCHEDULE_PARAMS.items():
            if field not in data:
                data[field] = default_value
        
        # Get scheduler
        SchoolScheduler = get_school_scheduler()
        scheduler = SchoolScheduler()
        scheduler.set_pe_constraints_enabled(data.get('enable_pe_constraints', False))
        scheduler.set_homeroom_mode(data.get('homeroom_mode', 1))
        
        # Initialize and run
        logger.info(f"Generating schedule with: {data}")
        
        if not scheduler.get_inputs(
            n_teachers=data['n_teachers'],
            grades=data['grades'],
            pe_teacher=data.get('pe_teacher', 'T13'),
            pe_grades=data.get('pe_grades', ['P4', 'P5', 'P6', 'M1', 'M2', 'M3']),
            pe_day=data.get('pe_day', 3),
            n_pe_periods=data.get('n_pe_periods', 6),
            start_hour=data.get('start_hour', 8),
            n_hours=data.get('n_hours', 8),
            lunch_hour=data.get('lunch_hour', 5),
            days_per_week=data.get('days_per_week', 5),
            enable_pe_constraints=data.get('enable_pe_constraints', False),
            homeroom_mode=data.get('homeroom_mode', 1)
        ):
            return create_response(success=False, message='Failed to initialize scheduler', status_code=500)
        
        scheduler.get_model()
        
        if not scheduler.get_solution():
            return create_response(success=False, message='No feasible solution found', status_code=422)
        
        # Format response
        schedule_data = scheduler.schedule_df.to_dict('records') if scheduler.schedule_df is not None else []
        homeroom_data = scheduler.homeroom_df.to_dict('records') if scheduler.homeroom_df is not None else []
        
        reformatted_schedule = []
        for item in schedule_data:
            converted_start_time = item["TimeSlot"].split("-")[0]
            reformatted_schedule.append({
                "subject": item["Grade"],
                "grade": item["Grade"],
                "teacher": item["Teacher"],
                "day": item["DayName"],
                "period": item["Hour"],
                "time": converted_start_time,
                "timeslot": item["TimeSlot"],
                "duration": 1
            })
        
        processing_time = round(time.time() - start_time, 2)
        
        return create_response(
            data={'schedule': reformatted_schedule, 'homeroom': homeroom_data, 'parameters': data},
            message='Schedule generated successfully',
            metadata={'processing_time_seconds': processing_time}
        )
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Schedule generation failed', error=str(e), status_code=500)


# ==================== PLANNER ENDPOINTS ====================
@app.route('/generate_planner_content', methods=['POST'])
def generate_planner_content():
    """Generate planner content using ChatGPT"""
    try:
        gpc = get_generate_planner_content()
        payload = request.get_json()
        
        logger.info(f"Received payload: {payload}")
        parsed = gpc.GeneratePlannerRequest(**payload)
        
        content = gpc.chat.generate(parsed)
        logger.info(f"Generated: {content.planName} with {len(content.days)} days")
        
        return jsonify(content.model_dump())
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Generation failed', error=str(e), status_code=500)


@app.route('/summarize_planner', methods=['POST'])
def summarize_planner():
    """Summarize planner data"""
    try:
        data = request.get_json()
        if not data or 'planner_data' not in data:
            return create_response(success=False, message='planner_data is required', status_code=400)
        
        language = data.get('language', 'thai')
        pu = get_planner_utils()
        summary = pu.summarize_plan(data['planner_data'], language)
        
        return create_response(data={'summary': summary}, message='Planner summarized successfully')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Summarization failed', error=str(e), status_code=500)


@app.route('/progress', methods=['POST'])
def progress():
    """Track user progress"""
    try:
        data = request.get_json()
        if not data or 'todo_data' not in data:
            return create_response(success=False, message='todo_data is required', status_code=400)
        
        user_query = data.get('user_update', 'Tell me about this todo list')
        language = data.get('language', 'thai')
        
        pu = get_planner_utils()
        information = pu.get_todo_information(user_query, data['todo_data'], language)
        
        return create_response(data={'feedback': information}, message='Todo information provided')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/coach', methods=['POST'])
def coach():
    """Respond to user input"""
    try:
        data = request.get_json()
        if not data or 'user_input' not in data or 'summary' not in data:
            return create_response(success=False, message='user_input and summary are required', status_code=400)
        
        pu = get_planner_utils()
        response = pu.respond_to_user_input(data['user_input'], data['summary'])
        
        return create_response(data={'response': response}, message='Response generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/encourage_in_the_morning', methods=['POST'])
def encourage_in_the_morning():
    """Encourage user to start the day"""
    try:
        data = request.get_json()
        if not data or 'today_todo_list_data' not in data:
            return create_response(success=False, message='today_todo_list_data is required', status_code=400)
        
        pu = get_planner_utils()
        response = pu.message_in_the_morning(
            today_todo_list_data=data['today_todo_list_data'],
            language=data.get('languageSelected', 'thai')
        )
        
        return create_response(data={'response': response}, message='Response generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/summarize_end_of_the_week', methods=['POST'])
def summarize_end_of_the_week():
    """Summarize end of week"""
    try:
        data = request.get_json()
        if not data or 'week_data' not in data:
            return create_response(success=False, message='week_data is required', status_code=400)
        
        language = data.get('language', 'thai')
        pu = get_planner_utils()
        rest_suggestions = pu.summarize_end_of_the_week_at_friday(week_data=data['week_data'], language=language)
        
        return create_response(data={'response': rest_suggestions}, message='Response generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/summarize_next_week', methods=['POST'])
def summarize_next_week():
    """Summarize next week's plan"""
    try:
        data = request.get_json()
        if not data or 'week_data' not in data:
            return create_response(success=False, message='week_data is required', status_code=400)
        
        language = data.get('language', 'thai')
        pu = get_planner_utils()
        preparation = pu.summarize_next_week_at_sunday(week_data=data['week_data'], language=language)
        
        return create_response(data={'response': preparation}, message='Response generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/summary_this_year_todos', methods=['POST'])
def summary_this_year_todos():
    """Summarize this year's todos"""
    try:
        data = request.get_json()
        if not data or 'this_year_todos_data' not in data:
            return create_response(success=False, message='this_year_todos_data is required', status_code=400)
        
        language = data.get('languageSelected', 'thai')
        pu = get_planner_utils()
        title, summary = pu.summarize_this_year_todos_message(
            this_year_todos_data=data['this_year_todos_data'],
            language=language
        )
        
        return create_response(data={'title': title, 'summary': summary}, message='Summary generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/summary_this_month_todos', methods=['POST'])
def summary_this_month_todos():
    """Summarize this month's todos"""
    try:
        data = request.get_json()
        if not data or 'this_month_todos_data' not in data:
            return create_response(success=False, message='this_month_todos_data is required', status_code=400)
        
        language = data.get('languageSelected', 'thai')
        pu = get_planner_utils()
        title, summary = pu.summarize_this_month_todos_message(
            this_month_todos_data=data['this_month_todos_data'],
            language=language
        )
        
        return create_response(data={'title': title, 'summary': summary}, message='Summary generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


@app.route('/todo_fate_prediction', methods=['POST'])
def todo_fate_prediction():
    """Predict fate of user's todos"""
    try:
        data = request.get_json()
        if not data or 'languageSelected' not in data:
            return create_response(success=False, message='languageSelected is required', status_code=400)
        
        todo_data = data.get('todo_data', [])
        language = data['languageSelected']
        
        pu = get_planner_utils()
        response = pu.predict_today_todo_fate(todo_data=todo_data, language=language)
        
        return create_response(data={'response': response}, message='Prediction generated')
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return create_response(success=False, message='Failed', error=str(e), status_code=500)


# ==================== API INFO ====================
@app.route('/get_schedule_info', methods=['GET'])
def get_schedule_info():
    """Get API information"""
    info = {
        'description': 'School Schedule Optimization API (Local)',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /get_schedule_info': 'API information',
            'POST /generate_schedule': 'Generate school schedule',
            'POST /generate_planner_content': 'Generate planner content',
            'POST /summarize_planner': 'Summarize planner data',
            'POST /progress': 'Track user progress',
            'POST /coach': 'Get coaching response',
            'POST /encourage_in_the_morning': 'Morning encouragement',
            'POST /summarize_end_of_the_week': 'End of week summary',
            'POST /summarize_next_week': 'Next week summary',
            'POST /summary_this_year_todos': 'Year todos summary',
            'POST /summary_this_month_todos': 'Month todos summary',
            'POST /todo_fate_prediction': 'Todo fate prediction'
        },
        'example_schedule_request': {
            "n_teachers": 13,
            "grades": ["P1", "P2", "P3", "P4", "P5", "P6", "M1", "M2", "M3"]
        }
    }
    return create_response(data=info, message='API information')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 1234))
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           LOCAL API SERVER - No Firebase Required            ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{port}                    ║
║                                                              ║
║  Endpoints:                                                  ║
║    GET  /health              - Health check                  ║
║    GET  /get_schedule_info   - API info                      ║
║    POST /generate_schedule   - Generate schedule             ║
║    POST /generate_planner_content - Generate planner         ║
║    POST /summarize_planner   - Summarize planner             ║
║    POST /coach               - Coaching response             ║
║    POST /progress            - Track progress                ║
║    ... and more                                              ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host='0.0.0.0', port=port, debug=True)
