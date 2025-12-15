# Google Cloud Function for School Schedule Optimization
# This function provides HTTP endpoints for generating and managing school schedules

import json
import logging
import os
import sys
import time
import traceback
import signal
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps

# Add current directory to Python path to ensure local modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Import local modules
from planner_utils import summarize_plan, motivate_user, track_progress, respond_to_user_input, message_in_the_morning, summarize_end_of_the_week_at_friday, summarize_next_week_at_sunday, get_todo_information, summarize_this_year_todos_message

# import the same models + chat wrapper from main.py
from generate_planner_content import GeneratePlannerRequest, chat
# Load environment variables
load_dotenv()

# Import school_scheduler from the same directory
try:
    from school_scheduler import SchoolScheduler
    SCHEDULER_AVAILABLE = True
    logging.info("SchoolScheduler imported successfully")
except ImportError as e:
    logging.error(f"Failed to import SchoolScheduler: {e}")
    logging.error(f"Import traceback: {traceback.format_exc()}")
    SchoolScheduler = None
    SCHEDULER_AVAILABLE = False

# Constants
MAX_INSTANCES = 5
MAX_TEACHERS = 50
MAX_GRADES = 20
MAX_HOURS_PER_DAY = 12
MAX_DAYS_PER_WEEK = 7
DEFAULT_TIMEOUT = 300  # 5 minutes
API_TIMEOUT = 90  # 90 seconds for API calls

# Default values for schedule parameters
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

# CORS headers
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin',
    'Access-Control-Max-Age': '3600',
    'Access-Control-Allow-Credentials': 'true',
    'Content-Type': 'application/json'
}

# Pydantic Models for request validation
class ScheduleRequest(BaseModel):
    """Model for schedule generation requests"""
    n_teachers: int = Field(..., gt=0, le=MAX_TEACHERS, description="Number of teachers")
    grades: List[str] = Field(..., min_items=1, max_items=MAX_GRADES, description="List of grade levels")
    pe_teacher: str = Field(default=DEFAULT_SCHEDULE_PARAMS['pe_teacher'], description="PE teacher ID")
    pe_grades: List[str] = Field(default=DEFAULT_SCHEDULE_PARAMS['pe_grades'], description="Grades with PE")
    pe_day: int = Field(default=DEFAULT_SCHEDULE_PARAMS['pe_day'], ge=1, le=7, description="Day for PE classes")
    n_pe_periods: int = Field(default=DEFAULT_SCHEDULE_PARAMS['n_pe_periods'], ge=0, description="Number of PE periods")
    start_hour: int = Field(default=DEFAULT_SCHEDULE_PARAMS['start_hour'], ge=0, le=23, description="Starting hour")
    n_hours: int = Field(default=DEFAULT_SCHEDULE_PARAMS['n_hours'], ge=1, le=MAX_HOURS_PER_DAY, description="Hours per day")
    lunch_hour: int = Field(default=DEFAULT_SCHEDULE_PARAMS['lunch_hour'], ge=1, description="Lunch hour")
    days_per_week: int = Field(default=DEFAULT_SCHEDULE_PARAMS['days_per_week'], ge=1, le=MAX_DAYS_PER_WEEK, description="Days per week")
    enable_pe_constraints: bool = Field(default=DEFAULT_SCHEDULE_PARAMS['enable_pe_constraints'], description="Enable PE constraints")
    homeroom_mode: int = Field(default=DEFAULT_SCHEDULE_PARAMS['homeroom_mode'], ge=0, le=2, description="Homeroom mode")

    class Config:
        json_schema_extra = {
            "example": {
                "n_teachers": 13,
                "grades": ["P1", "P2", "P3", "P4", "P5", "P6", "M1", "M2", "M3"],
                "pe_teacher": "T13",
                "pe_grades": ["P4", "P5", "P6", "M1", "M2", "M3"],
                "pe_day": 3,
                "n_pe_periods": 6,
                "start_hour": 8,
                "n_hours": 8,
                "lunch_hour": 5,
                "days_per_week": 5,
                "enable_pe_constraints": False,
                "homeroom_mode": 1
            }
        }

class PlannerDataRequest(BaseModel):
    """Model for planner data requests"""
    planner_data: Dict[str, Any] = Field(..., description="Planner data")
    language: str = Field(default="thai", description="Response language")

class ProgressUpdateRequest(BaseModel):
    """Model for progress update requests"""
    user_update: Optional[str] = Field(default=None, description="User update")
    summary: Optional[str] = Field(default=None, description="Summary")
    todo_data: Optional[Dict[str, Any]] = Field(default=None, description="Todo data")

class UserInputRequest(BaseModel):
    """Model for user input requests"""
    user_input: str = Field(..., description="User input")
    summary: str = Field(..., description="Summary")

# Response Models
class ApiResponse(BaseModel):
    """Base API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ScheduleResponse(ApiResponse):
    """Schedule generation response model"""
    schedule: Optional[List[Dict[str, Any]]] = None
    homeroom: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None

# Initialize Firebase app
try:
    initialize_app()
    logging.info("Firebase app initialized successfully")
except Exception as e:
    logging.warning(f"Firebase app already initialized or failed to initialize: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set global options for cost control
set_global_options(max_instances=MAX_INSTANCES)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def with_timeout(seconds: int):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                return create_response(
                    success=False,
                    message='Request timeout',
                    error=f'Operation timed out after {seconds} seconds',
                    status_code=408
                )
            finally:
                # Restore original handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

def create_response(
    data: Optional[Dict[str, Any]] = None,
    success: bool = True,
    message: str = "Success",
    error: Optional[str] = None,
    status_code: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> https_fn.Response:
    """Create a standardized HTTP response"""
    response_data = {
        'success': success,
        'message': message,
        'data': data,
        'error': error,
        'metadata': metadata
    }
    
    return https_fn.Response(
        json.dumps(response_data, default=str),
        status=status_code,
        headers=CORS_HEADERS
    )

def handle_preflight_request() -> https_fn.Response:
    """Handle CORS preflight requests"""
    return https_fn.Response('', status=200, headers=CORS_HEADERS)

def validate_schedule_request(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate the incoming schedule request data"""
    try:
        # Validate required fields
        required_fields = ['n_teachers', 'grades']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate n_teachers
        try:
            n_teachers = int(data['n_teachers'])
            if n_teachers <= 0 or n_teachers > MAX_TEACHERS:
                return False, f"n_teachers must be between 1 and {MAX_TEACHERS}"
        except (ValueError, TypeError):
            return False, "n_teachers must be a valid integer"
        
        # Validate grades
        grades = data['grades']
        if not isinstance(grades, list) or len(grades) == 0:
            return False, "grades must be a non-empty list"
        
        if len(grades) > MAX_GRADES:
            return False, f"grades list cannot exceed {MAX_GRADES} items"
        
        # Validate individual grades
        for grade in grades:
            if not isinstance(grade, str) or len(grade) == 0:
                return False, f"Invalid grade format: {grade}"
        
        # Set default values for optional fields
        for field, default_value in DEFAULT_SCHEDULE_PARAMS.items():
            if field not in data:
                data[field] = default_value
        
        # Validate lunch_hour against n_hours
        if 'lunch_hour' in data and 'n_hours' in data:
            if data['lunch_hour'] > data['n_hours']:
                return False, "lunch_hour must be between 1 and n_hours"
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error in validate_schedule_request: {e}")
        return False, f"Validation error: {str(e)}"

def format_schedule_data(schedule_df, homeroom_df) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Format schedule data for response"""
    schedule_data = []
    if schedule_df is not None:
        schedule_data = schedule_df.to_dict('records')
    
    homeroom_data = []
    if homeroom_df is not None:
        homeroom_data = homeroom_df.to_dict('records')
    
    # Reformat schedule data
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
    
    return reformatted_schedule, homeroom_data

# Generate a school schedule based on provided parameters
@https_fn.on_request(max_instances=3)
def generate_schedule(req: https_fn.Request) -> https_fn.Response:
    """Generate a school schedule based on provided parameters"""
    start_time = time.time()
    
    try:
        # Handle preflight requests
        if req.method == 'OPTIONS':
            return handle_preflight_request()
        
        # Validate HTTP method
        if req.method != 'POST':
            return create_response(
                success=False,
                message=f'Method {req.method} not allowed',
                error='This endpoint only accepts POST requests with JSON data',
                status_code=405,
                data={
                    'endpoints': {
                        'POST /generate_schedule': 'Generate a school schedule (requires JSON data)',
                        'GET /health_check': 'Check service health',
                        'GET /get_schedule_info': 'Get API information and examples',
                        'GET /debug': 'Get debug information'
                    }
                }
            )
        
        # Parse request data
        try:
            data = req.get_json()
            if data is None:
                return create_response(
                    success=False,
                    message='No JSON data provided',
                    error='This endpoint requires a POST request with JSON data in the body',
                    status_code=400,
                    data={'example': ScheduleRequest.Config.json_schema_extra['example']}
                )
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return create_response(
                success=False,
                message='Invalid JSON',
                error=f'This endpoint requires a POST request with valid JSON data in the body. Error: {str(e)}',
                status_code=400
            )
        
        # Validate request data
        is_valid, error_message = validate_schedule_request(data)
        if not is_valid:
            logger.warning(f"Invalid request data: {error_message}")
            return create_response(
                success=False,
                message='Validation failed',
                error=error_message,
                status_code=400
            )
        
        # Check if SchoolScheduler is available
        if not SCHEDULER_AVAILABLE:
            logger.error("SchoolScheduler module not available")
            return create_response(
                success=False,
                message='Service unavailable',
                error='SchoolScheduler module not available',
                status_code=500
            )
        
        # Generate schedule
        logger.info(f"Generating schedule with parameters: {data}")
        
        try:
            scheduler = SchoolScheduler()
            scheduler.set_pe_constraints_enabled(data.get('enable_pe_constraints', False))
            scheduler.set_homeroom_mode(data.get('homeroom_mode', 1))
            
            # Initialize scheduler inputs
            logger.info("Initializing scheduler inputs...")
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
                logger.error("Failed to initialize scheduler inputs")
                return create_response(
                    success=False,
                    message='Initialization failed',
                    error='Failed to initialize scheduler inputs',
                    status_code=500
                )
            
            # Build optimization model
            logger.info("Building optimization model...")
            scheduler.get_model()
            
            # Solve optimization problem
            logger.info("Solving optimization problem...")
            if not scheduler.get_solution():
                logger.warning("No feasible solution found for the given constraints")
                return create_response(
                    success=False,
                    message='No solution found',
                    error='No feasible solution found for the given constraints',
                    status_code=422
                )
            
            # Format response data
            logger.info("Preparing response data...")
            schedule_data, homeroom_data = format_schedule_data(scheduler.schedule_df, scheduler.homeroom_df)
            
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"Schedule generated successfully in {processing_time} seconds")
            
            return create_response(
                data={
                    'schedule': schedule_data,
                    'homeroom': homeroom_data,
                    'parameters': data
                },
                success=True,
                message='Schedule generated successfully',
                metadata={
                    'total_assignments': len(schedule_data),
                    'homeroom_assignments': len(homeroom_data),
                    'processing_time_seconds': processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error in schedule generation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return create_response(
                success=False,
                message='Schedule generation failed',
                error=f'Schedule generation failed: {str(e)}',
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_schedule: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_response(
            success=False,
            message='Internal server error',
            error=f'Internal server error: {str(e)}',
            status_code=500
        )


# Get information about available schedule parameters and constraints
@https_fn.on_request()
def get_schedule_info(req: https_fn.Request) -> https_fn.Response:
    """Get information about available schedule parameters and constraints"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'GET':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only GET method is allowed',
            status_code=405
        )
    
    info_data = {
        'description': 'School Schedule Optimization API',
        'endpoints': {
            'POST /generate_schedule': 'Generate a new school schedule',
            'GET /health_check': 'Check service health',
            'GET /get_schedule_info': 'Get API information',
            'GET /debug': 'Get debug information'
        },
        'required_parameters': {
            'n_teachers': f'Number of teachers (integer, 1-{MAX_TEACHERS})',
            'grades': f'List of grade levels (e.g., ["P1", "P2", "P3"], max {MAX_GRADES} items)'
        },
        'optional_parameters': {
            'pe_teacher': 'Physical education teacher ID (default: "T13")',
            'pe_grades': 'Grades that have PE (default: ["P4", "P5", "P6", "M1", "M2", "M3"])',
            'pe_day': 'Day for PE classes (default: 3)',
            'n_pe_periods': 'Number of PE periods (default: 6)',
            'start_hour': 'Starting hour (default: 8)',
            'n_hours': f'Number of hours per day (default: 8, max: {MAX_HOURS_PER_DAY})',
            'lunch_hour': 'Lunch hour (default: 5)',
            'days_per_week': f'Days per week (default: 5, max: {MAX_DAYS_PER_WEEK})',
            'enable_pe_constraints': 'Enable PE constraints (default: false)',
            'homeroom_mode': 'Homeroom mode: 0=none, 1=basic, 2=advanced (default: 1)'
        },
        'example_request': ScheduleRequest.Config.json_schema_extra['example'],
        'constraints': {
            'max_teachers': MAX_TEACHERS,
            'max_grades': MAX_GRADES,
            'max_hours_per_day': MAX_HOURS_PER_DAY,
            'max_days_per_week': MAX_DAYS_PER_WEEK
        }
    }
    
    return create_response(data=info_data, message='API information retrieved successfully')

########### Generate Planner Content API Endpoints #############
@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540)  # 9 minutes timeout
def generate_planner_content(req: https_fn.Request) -> https_fn.Response:
    """Generate planner content using ChatGPT"""
    try:
        payload = req.get_json()
        print(f"Received payload: {payload}")
        
        parsed = GeneratePlannerRequest(**payload)
        print(f"Parsed request: {parsed}")
        
        content = chat.generate(parsed)
        print(f"Generated content: {content.planName} with {len(content.days)} days")
        
        return content.model_dump()
    except ValidationError as ve:
        print(f"Validation error: {ve.errors()}")
        # Format validation errors in a user-friendly way
        errors = []
        for error in ve.errors():
            field = " → ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            # Make error messages more user-friendly
            if "type_error" in message:
                message = "Please provide a valid value"
            elif "value_error" in message:
                message = "The value provided is not valid"
            errors.append(f"{field}: {message}")
        
        user_friendly_detail = {
            "error": "Invalid request parameters",
            "message": "Please check the following fields and try again:",
            "details": errors
        }
        raise HTTPException(status_code=400, detail=user_friendly_detail)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide user-friendly error message without exposing internals
        error_str = str(e).lower()
        if "api" in error_str or "openai" in error_str:
            user_message = "We're having trouble generating your planner right now. Please try again in a moment."
        elif "timeout" in error_str:
            user_message = "The request took too long to process. Please try with fewer days or simpler requirements."
        elif "rate" in error_str or "quota" in error_str:
            user_message = "We've reached our service limit. Please try again in a few minutes."
        else:
            user_message = "We couldn't generate your planner. Please check your inputs and try again."
        
        raise HTTPException(status_code=500, detail={"error": "Generation failed", "message": user_message})


#################### EVO ChatGPT API Endpoints ######################

# Summarize planner data using ChatGPT
@https_fn.on_request(memory=1024, max_instances=3)
def summarize_planner(req: https_fn.Request) -> https_fn.Response:
    """Summarize planner data using ChatGPT"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        if 'planner_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='planner_data is required',
                status_code=400
            )
        
        language = data.get('language', 'thai')
        logger.info(f"Summarizing planner data in language: {language}")
        
        summary = summarize_plan(data['planner_data'], language)
        return create_response(
            data={'summary': summary},
            message='Planner summarized successfully'
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_planner: {str(e)}")
        return create_response(
            success=False,
            message='Summarization failed',
            error=f'Failed to summarize planner: {str(e)}',
            status_code=500
        )

# AI Assistant to provide information about todo_data
@https_fn.on_request(memory=1024, max_instances=3)
def progress(req: https_fn.Request) -> https_fn.Response:
    """Track user progress using ChatGPT"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        required_fields = ['todo_data']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    message='Missing required field',
                    error=f'{field} is required',
                    status_code=400
                )
        
        # Get optional user query about the todo_data
        user_query = data.get('user_update', 'Tell me about this todo list')
        language = data.get('language', 'thai')
        todo_data = data['todo_data']
        
        # Get information about todo_data using AI assistant
        information = get_todo_information(user_query, todo_data, language)
        return create_response(
            data={'feedback': information},
            message='Todo information provided successfully'
        )
        
    except Exception as e:
        logger.error(f"Error in todo_assistant: {str(e)}")
        return create_response(
            success=False,
            message='Todo assistant failed',
            error=f'Failed to provide information: {str(e)}',
            status_code=500
        )

# Respond to user input using ChatGPT
@https_fn.on_request(memory=1024, max_instances=3)
def coach(req: https_fn.Request) -> https_fn.Response:
    """Respond to user input using ChatGPT"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        required_fields = ['user_input', 'summary']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    message='Missing required field',
                    error=f'{field} is required',
                    status_code=400
                )
        
        response = respond_to_user_input(data['user_input'], data['summary'])
        return create_response(
            data={'response': response},
            message='Response generated successfully'
        )
        
    except Exception as e:
        logger.error(f"Error in coach: {str(e)}")
        return create_response(
            success=False,
            message='Response generation failed',
            error=f'Failed to generate response: {str(e)}',
            status_code=500
        )

# Encourage the user to start the day using ChatGPT
@https_fn.on_request(memory=1024, max_instances=3)
def encourage_in_the_morning(req: https_fn.Request) -> https_fn.Response:
    """encourage the user to start the day using ChatGPT"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        required_fields = ['today_todo_list_data']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    message='Missing required field',
                    error=f'{field} is required',
                    status_code=400
                )
        
        response = message_in_the_morning(today_todo_list_data=data['today_todo_list_data'], language=data['languageSelected'])
        return create_response(
            data={'response': response},
            message='Response generated successfully'
        )
        
    except Exception as e:
        logger.error(f"Error in encourage_in_the_morning: {str(e)}")
        return create_response(
            success=False,
            message='Response generation failed',
            error=f'Failed to generate response: {str(e)}',
            status_code=500
        )

# Summarize the end of the week using ChatGPT and suggest rest to recharge energy
@https_fn.on_request(memory=1024, max_instances=3)
def summarize_end_of_the_week(req: https_fn.Request) -> https_fn.Response:
    """Summarize the end of the week using ChatGPT and suggest rest to recharge energy"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        if 'week_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='week_data is required',
                status_code=400
            )
        
        language = data.get('language', 'thai')
        logger.info(f"Summarizing end of week data in language: {language}")
        
        # Get week summary using planner utilities
        #week_summary = summarize_plan(data['week_data'], 'week_summary', language)
        
        # Generate rest and recharge suggestions
        rest_suggestions = summarize_end_of_the_week_at_friday(week_data=data['week_data'], language=language)
        
        return create_response(
            data={'response': rest_suggestions},
            message='Response generated successfully',
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_end_of_the_week: {str(e)}")
        return create_response(
            success=False,
            message='Response generation failed',
            error=f'Failed to generate response: {str(e)}',
            status_code=500
        )
        
# Summarize next week's plan and provide encouraging preparation suggestions
@https_fn.on_request(memory=1024, max_instances=3)
def summarize_next_week(req: https_fn.Request) -> https_fn.Response:
    """Summarize next week's plan and provide encouraging preparation suggestions"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        if 'week_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='week_data is required',
                status_code=400
            )
        
        language = data.get('language', 'thai')
        logger.info(f"Summarizing next week data in language: {language}")
        
        # Get week summary using planner utilities
        #week_summary = summarize_plan(data['week_data'], 'week_summary', language)
        
        # Generate preparation suggestions and encouragement
        preparation_suggestions = summarize_next_week_at_sunday(week_data=data['week_data'], language = language)
        
        return create_response(
            data={'response': preparation_suggestions},
            message='Next week summary and preparation suggestions generated successfully'
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_next_week: {str(e)}")
        return create_response(
            success=False,
            message='Response generation failed',
            error=f'Failed to generate response: {str(e)}',
            status_code=500
        )

@https_fn.on_request(memory=1024, max_instances=3)
def summary_this_year_todos(req: https_fn.Request) -> https_fn.Response:
    """Summarize this year's todos using ChatGPT"""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    try:
        data = req.get_json()
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                error='Request body is required',
                status_code=400
            )
        
        # Validate required fields
        if 'this_year_todos_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='year_data is required',
                status_code=400
            )
        
        language = data.get('languageSelected', 'thai')
        logger.info(f"Summarizing this year's todos in language: {language}")
        
        # Summarize this year's todos using ChatGPT
        summary = summarize_this_year_todos_message(this_year_todos_data=data['this_year_todos_data'], language=language)
        return create_response(
            data={'response': summary},
            message='This year\'s todos summary generated successfully',
            status_code=200,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_this_year_todos: {str(e)}")
        return create_response(
            success=False,
            message='This year\'s todos summary generation failed',
            error=f'Failed to generate this year\'s todos summary: {str(e)}',
            status_code=500
        )