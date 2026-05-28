# Google Cloud Function for School Schedule Optimization
# This function provides HTTP endpoints for generating and managing school schedules

import json
import logging
import os
import sys
import time
import traceback
import signal
import uuid
import io
import requests
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps

# Add current directory to Python path to ensure local modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app, storage, firestore
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from datetime import datetime
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Lazy-loaded modules to reduce cold start time
_planner_utils = None
_generate_planner_content = None
_school_scheduler = None
_scheduler_available = None
_todo_generator = None

def get_planner_utils():
    """Lazy load planner_utils module"""
    global _planner_utils
    if _planner_utils is None:
        import planner_utils as pu
        _planner_utils = pu
    return _planner_utils

def get_generate_planner_content():
    """Lazy load generate_planner_content module"""
    global _generate_planner_content
    if _generate_planner_content is None:
        import generate_planner_content as gpc
        _generate_planner_content = gpc
    return _generate_planner_content

def get_todo_generator():
    """Lazy load todo_generator module"""
    global _todo_generator
    if _todo_generator is None:
        import todo_generator as tg
        _todo_generator = tg
    return _todo_generator

def get_school_scheduler():
    """Lazy load school_scheduler module"""
    global _school_scheduler, _scheduler_available
    if _scheduler_available is None:
        try:
            from school_scheduler import SchoolScheduler
            _school_scheduler = SchoolScheduler
            _scheduler_available = True
            logging.info("SchoolScheduler imported successfully")
        except ImportError as e:
            logging.error(f"Failed to import SchoolScheduler: {e}")
            logging.error(f"Import traceback: {traceback.format_exc()}")
            _school_scheduler = None
            _scheduler_available = False
    return _school_scheduler, _scheduler_available

#Lazy-loaded YOLO model
_yolo_model = None

def get_yolo_model():
    """Lazy load YOLO model"""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = os.path.join(current_dir, 'trained_model', 'best.pt')
            _yolo_model = YOLO(model_path)
            logging.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            logging.error(f"Load traceback: {traceback.format_exc()}")
            _yolo_model = None
    return _yolo_model

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
    metadata: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None
) -> https_fn.Response:
    """Create a standardized HTTP response."""
    response_data = {
        'success': success,
        'message': message,
        'data': data,
        'error': error,
        'metadata': metadata
    }
    headers = {**CORS_HEADERS, **(extra_headers or {})}
    return https_fn.Response(
        json.dumps(response_data, default=str),
        status=status_code,
        headers=headers
    )


def handle_preflight_request() -> https_fn.Response:
    """Handle CORS preflight requests."""
    return https_fn.Response('', status=200, headers=CORS_HEADERS)


def _month_context_from_request(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build month_context dict from request body (previous_month_data, current_month_data, next_month_data)."""
    month_context = {}
    for key, req_key in [
        ('previous', 'previous_month_data'),
        ('current', 'current_month_data'),
        ('next', 'next_month_data'),
    ]:
        val = data.get(req_key)
        if val is not None and (isinstance(val, str) and val.strip() or isinstance(val, list)):
            month_context[key] = val
    return month_context if month_context else None


def _month_context_for_user(
    user_id: Optional[str],
    data: Dict[str, Any],
    top_k_per_period: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Get month_context from RAG (user memory) when user_id is present, then merge request body.
    Request body (previous_month_data, current_month_data, next_month_data) can override or fill gaps.
    """
    month_context = None
    if user_id and isinstance(user_id, str) and user_id.strip():
        try:
            from user_memory import retrieve_month_context_from_rag
            month_context = retrieve_month_context_from_rag(user_id.strip(), top_k_per_period=top_k_per_period)
        except Exception as e:
            logger.warning("RAG month context retrieval failed: %s", e)
    body_month = _month_context_from_request(data)
    if body_month:
        merged = dict(month_context or {})
        merged.update(body_month)  # request body overrides or supplements RAG
        return merged if merged else None
    return month_context


def _get_intent_profile_for_user(user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetch computed intent profile from users/{uid}.intent_profile."""
    if not user_id or not isinstance(user_id, str) or not user_id.strip():
        return None
    try:
        db = firestore.client()
        snap = db.collection("users").document(user_id.strip()).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        profile = data.get("intent_profile")
        return profile if isinstance(profile, dict) else None
    except Exception as e:
        logger.warning("Failed to fetch intent profile for user %s: %s", user_id, e)
        return None


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

# # Generate a school schedule based on provided parameters
# @https_fn.on_request(max_instances=3)
# def generate_schedule(req: https_fn.Request) -> https_fn.Response:
#     """Generate a school schedule based on provided parameters"""
#     start_time = time.time()
    
#     try:
#         # Handle preflight requests
#         if req.method == 'OPTIONS':
#             return handle_preflight_request()
        
#         # Validate HTTP method
#         if req.method != 'POST':
#             return create_response(
#                 success=False,
#                 message=f'Method {req.method} not allowed',
#                 error='This endpoint only accepts POST requests with JSON data',
#                 status_code=405,
#                 data={
#                     'endpoints': {
#                         'POST /generate_schedule': 'Generate a school schedule (requires JSON data)',
#                         'GET /health_check': 'Check service health',
#                         'GET /get_schedule_info': 'Get API information and examples',
#                         'GET /debug': 'Get debug information'
#                     }
#                 }
#             )
        
#         # Parse request data
#         try:
#             data = req.get_json()
#             if data is None:
#                 return create_response(
#                     success=False,
#                     message='No JSON data provided',
#                     error='This endpoint requires a POST request with JSON data in the body',
#                     status_code=400,
#                     data={'example': ScheduleRequest.Config.json_schema_extra['example']}
#                 )
#         except Exception as e:
#             logger.error(f"JSON parsing error: {e}")
#             return create_response(
#                 success=False,
#                 message='Invalid JSON',
#                 error=f'This endpoint requires a POST request with valid JSON data in the body. Error: {str(e)}',
#                 status_code=400
#             )
        
#         # Validate request data
#         is_valid, error_message = validate_schedule_request(data)
#         if not is_valid:
#             logger.warning(f"Invalid request data: {error_message}")
#             return create_response(
#                 success=False,
#                 message='Validation failed',
#                 error=error_message,
#                 status_code=400
#             )
        
#         # Check if SchoolScheduler is available (lazy load)
#         SchoolScheduler, scheduler_available = get_school_scheduler()
#         if not scheduler_available:
#             logger.error("SchoolScheduler module not available")
#             return create_response(
#                 success=False,
#                 message='Service unavailable',
#                 error='SchoolScheduler module not available',
#                 status_code=500
#             )
        
#         # Generate schedule
#         logger.info(f"Generating schedule with parameters: {data}")
        
#         try:
#             scheduler = SchoolScheduler()
#             scheduler.set_pe_constraints_enabled(data.get('enable_pe_constraints', False))
#             scheduler.set_homeroom_mode(data.get('homeroom_mode', 1))
            
#             # Initialize scheduler inputs
#             logger.info("Initializing scheduler inputs...")
#             if not scheduler.get_inputs(
#                 n_teachers=data['n_teachers'],
#                 grades=data['grades'],
#                 pe_teacher=data.get('pe_teacher', 'T13'),
#                 pe_grades=data.get('pe_grades', ['P4', 'P5', 'P6', 'M1', 'M2', 'M3']),
#                 pe_day=data.get('pe_day', 3),
#                 n_pe_periods=data.get('n_pe_periods', 6),
#                 start_hour=data.get('start_hour', 8),
#                 n_hours=data.get('n_hours', 8),
#                 lunch_hour=data.get('lunch_hour', 5),
#                 days_per_week=data.get('days_per_week', 5),
#                 enable_pe_constraints=data.get('enable_pe_constraints', False),
#                 homeroom_mode=data.get('homeroom_mode', 1)
#             ):
#                 logger.error("Failed to initialize scheduler inputs")
#                 return create_response(
#                     success=False,
#                     message='Initialization failed',
#                     error='Failed to initialize scheduler inputs',
#                     status_code=500
#                 )
            
#             # Build optimization model
#             logger.info("Building optimization model...")
#             scheduler.get_model()
            
#             # Solve optimization problem
#             logger.info("Solving optimization problem...")
#             if not scheduler.get_solution():
#                 logger.warning("No feasible solution found for the given constraints")
#                 return create_response(
#                     success=False,
#                     message='No solution found',
#                     error='No feasible solution found for the given constraints',
#                     status_code=422
#                 )
            
#             # Format response data
#             logger.info("Preparing response data...")
#             schedule_data, homeroom_data = format_schedule_data(scheduler.schedule_df, scheduler.homeroom_df)
            
#             processing_time = round(time.time() - start_time, 2)
#             logger.info(f"Schedule generated successfully in {processing_time} seconds")
            
#             return create_response(
#                 data={
#                     'schedule': schedule_data,
#                     'homeroom': homeroom_data,
#                     'parameters': data
#                 },
#                 success=True,
#                 message='Schedule generated successfully',
#                 metadata={
#                     'total_assignments': len(schedule_data),
#                     'homeroom_assignments': len(homeroom_data),
#                     'processing_time_seconds': processing_time
#                 }
#             )
            
#         except Exception as e:
#             logger.error(f"Error in schedule generation: {str(e)}")
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             return create_response(
#                 success=False,
#                 message='Schedule generation failed',
#                 error=f'Schedule generation failed: {str(e)}',
#                 status_code=500
#             )
        
#     except Exception as e:
#         logger.error(f"Unexpected error in generate_schedule: {str(e)}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return create_response(
#             success=False,
#             message='Internal server error',
#             error=f'Internal server error: {str(e)}',
#             status_code=500
#         )


# # Get information about available schedule parameters and constraints
# @https_fn.on_request()
# def get_schedule_info(req: https_fn.Request) -> https_fn.Response:
#     """Get information about available schedule parameters and constraints"""
#     if req.method == 'OPTIONS':
#         return handle_preflight_request()
    
#     if req.method != 'GET':
#         return create_response(
#             success=False,
#             message='Method not allowed',
#             error='Only GET method is allowed',
#             status_code=405
#         )
    
#     info_data = {
#         'description': 'School Schedule Optimization API',
#         'endpoints': {
#             'POST /generate_schedule': 'Generate a new school schedule',
#             'GET /health_check': 'Check service health',
#             'GET /get_schedule_info': 'Get API information',
#             'GET /debug': 'Get debug information'
#         },
#         'required_parameters': {
#             'n_teachers': f'Number of teachers (integer, 1-{MAX_TEACHERS})',
#             'grades': f'List of grade levels (e.g., ["P1", "P2", "P3"], max {MAX_GRADES} items)'
#         },
#         'optional_parameters': {
#             'pe_teacher': 'Physical education teacher ID (default: "T13")',
#             'pe_grades': 'Grades that have PE (default: ["P4", "P5", "P6", "M1", "M2", "M3"])',
#             'pe_day': 'Day for PE classes (default: 3)',
#             'n_pe_periods': 'Number of PE periods (default: 6)',
#             'start_hour': 'Starting hour (default: 8)',
#             'n_hours': f'Number of hours per day (default: 8, max: {MAX_HOURS_PER_DAY})',
#             'lunch_hour': 'Lunch hour (default: 5)',
#             'days_per_week': f'Days per week (default: 5, max: {MAX_DAYS_PER_WEEK})',
#             'enable_pe_constraints': 'Enable PE constraints (default: false)',
#             'homeroom_mode': 'Homeroom mode: 0=none, 1=basic, 2=advanced (default: 1)'
#         },
#         'example_request': ScheduleRequest.Config.json_schema_extra['example'],
#         'constraints': {
#             'max_teachers': MAX_TEACHERS,
#             'max_grades': MAX_GRADES,
#             'max_hours_per_day': MAX_HOURS_PER_DAY,
#             'max_days_per_week': MAX_DAYS_PER_WEEK
#         }
#     }
    
#     return create_response(data=info_data, message='API information retrieved successfully')

########### Generate Planner Content API Endpoints #############
@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2)  # 9 minutes timeout
def generate_planner_content(req: https_fn.Request) -> https_fn.Response:
    """Generate planner content (sync). Supports chunked generation for totalDays > 7."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405,
        )

    try:
        gpc = get_generate_planner_content()
        payload = req.get_json() or {}
        logger.info("generate_planner_content: days=%s", payload.get("totalDays"))
        parsed = gpc.GeneratePlannerRequest(**payload)
        chat = gpc.ChatWrapper(gpc.ChatWrapperConfig())
        content = chat.generate(parsed)
        logger.info(
            "generate_planner_content: %s with %s days",
            content.planName,
            len(content.days),
        )
        return create_response(
            data=content.model_dump(),
            message="Plan generated successfully",
        )
    except ValidationError as ve:
        errors = []
        for error in ve.errors():
            field = " → ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            if "type_error" in message:
                message = "Please provide a valid value"
            elif "value_error" in message:
                message = "The value provided is not valid"
            errors.append(f"{field}: {message}")
        return create_response(
            success=False,
            message="Please check the following fields and try again",
            error="; ".join(errors),
            status_code=400,
        )
    except gpc.PlannerGenerationError as e:
        return create_response(
            success=False,
            message="Generation failed",
            error=e.user_message,
            status_code=500,
        )
    except Exception as e:
        logger.error("generate_planner_content error: %s", e)
        traceback.print_exc()
        error_str = str(e).lower()
        if "api" in error_str or "openai" in error_str:
            user_message = "We're having trouble generating your planner right now. Please try again in a moment."
        elif "timeout" in error_str:
            user_message = "The request took too long to process. Please try with fewer days or simpler requirements."
        elif "rate" in error_str or "quota" in error_str:
            user_message = "We've reached our service limit. Please try again in a few minutes."
        else:
            user_message = "We couldn't generate your planner. Please check your inputs and try again."
        return create_response(
            success=False,
            message="Generation failed",
            error=user_message,
            status_code=500,
        )


@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2)
def refine_planner_content(req: https_fn.Request) -> https_fn.Response:
    """Refine an existing AI-generated plan draft based on user feedback."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405,
        )

    try:
        gpc = get_generate_planner_content()
        payload = req.get_json() or {}
        parsed = gpc.RefinePlannerRequest(**payload)
        chat = gpc.ChatWrapper(gpc.ChatWrapperConfig())
        content = chat.refine_plan(parsed)
        return create_response(
            data=content.model_dump(),
            message="Plan refined successfully",
        )
    except ValidationError as ve:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
        return create_response(
            success=False,
            message='Invalid request parameters',
            error=str(errors),
            status_code=400,
        )
    except gpc.PlannerGenerationError as e:
        return create_response(
            success=False,
            message='Refinement failed',
            error=e.user_message,
            status_code=500,
        )
    except Exception as e:
        logger.error(f"refine_planner_content error: {e}")
        import traceback
        traceback.print_exc()
        return create_response(
            success=False,
            message='Refinement failed',
            error=str(e),
            status_code=500,
        )


# =========================
# Coach Review Endpoint — domain-aware AI coaching for EVO Coach Premium
# =========================
# Single HTTPS endpoint backing two features in the mobile app:
#   - screens/taskCoachReview.js  (per-task professional review)
#   - screens/planCoachReview.js  (plan-progress review)
#
# Tiering model (drives model selection):
#   tier == "free"     → light model (cheap, good-enough)
#   tier == "premium"  → high-quality model (premium subscriber, 199 THB/mo)
#
# Both tiers get UNLIMITED reviews. The differentiator is reasoning quality
# and richness, not artificial scarcity. This is intentional: a coach who
# stops talking to you on day 4 of the month isn't a coach.
#
# Request payload (POST):
#   {
#     "summary":        "<rich context block written by the frontend>",
#     "user_input":     "<the structured ask, usually requesting JSON output>",
#     "languageSelected": "thai" | "english",
#     "tier":            "free" | "premium"   (defaults to "free")
#   }
#
# Response payload:
#   {
#     "success": true,
#     "data": {
#       "response":    "<LLM text — frontend parses JSON inside>",
#       "model_used":  "gpt-4o-mini" | "gpt-5.4"
#     }
#   }

# Model choices. Free tier uses the cheapest competent model; premium uses
# a deep-reasoning model that produces meaningfully better coach reviews.
COACH_MODEL_FREE = "gpt-5.4-mini"
COACH_MODEL_PREMIUM = "gpt-5.4"

# Coach persona instructions wrapped around the frontend's user_input.
# The frontend's `summary` field already encodes the domain-specific persona;
# this system prompt just enforces output discipline so the JSON contract holds.
_COACH_SYSTEM_PROMPT = (
    "You are EVO Coach, an evidence-based personal development coach. You receive a "
    "structured context block from the user that defines the domain (diet, exercise, "
    "exam prep, sleep, meditation, or general) and the persona you should adopt. "
    "Honor that persona — speak like a real expert in that domain.\n\n"
    "Critical output rules:\n"
    "1. When the user asks for JSON, return ONLY valid JSON. No markdown fences, no preamble.\n"
    "2. Be specific. Reference the actual data the user provided.\n"
    "3. Never moralize. Never be toxic-positive. Never give generic advice.\n"
    "4. Tone: warm, direct, evidence-based. Short sentences over long ones."
)


# ---------------------------------------------------------------------------
# Coach review security helpers
# ---------------------------------------------------------------------------
# The client passes `tier: "premium"` to request the flagship model. We CANNOT
# trust that field — a free user could spoof it from the client. Instead, we
# verify the Firebase Auth ID token, look up the user's subscription doc in
# Firestore, and derive the tier from that. The client's claim is ignored.
#
# `Authorization: Bearer <id_token>` header is the standard. If the header is
# missing or invalid, the request is downgraded to free tier (light model)
# rather than rejected — anonymous / dev callers still get a useful response,
# they just don't get the flagship model for free.

# Simple in-memory per-UID rate limit. Cloud Functions instances are
# short-lived but this gives a soft cap that survives within one warm
# container. Free tier: ~20 reviews / 5 min. Premium: ~60 / 5 min.
# Hard limits should come from billing, but this prevents the obvious
# abuse case (a script hammering the endpoint).
_coach_rate_state = {}  # uid -> list[timestamp]
_COACH_RATE_WINDOW_SEC = 300
_COACH_RATE_FREE_MAX = 20
_COACH_RATE_PREMIUM_MAX = 60

def _coach_rate_allow(uid: str, is_paid: bool) -> bool:
    if not uid:
        # Anonymous callers share one bucket — strict cap.
        uid = "__anon__"
    now = time.time()
    cap = _COACH_RATE_PREMIUM_MAX if is_paid else _COACH_RATE_FREE_MAX
    bucket = [t for t in _coach_rate_state.get(uid, []) if now - t < _COACH_RATE_WINDOW_SEC]
    if len(bucket) >= cap:
        _coach_rate_state[uid] = bucket
        return False
    bucket.append(now)
    _coach_rate_state[uid] = bucket
    return True


def _verify_coach_tier(req: https_fn.Request) -> Tuple[Optional[str], str]:
    """
    Decode the Firebase Auth ID token from the request and check the user's
    subscription doc. Returns (uid, tier) where tier ∈ {"free","plus","premium"}.
    Anonymous / invalid token yields (None, "free") — caller served free tier.

    Backwards-compat note: callers that previously destructured (uid, is_premium)
    should be updated to read (uid, tier) and treat anything other than
    "premium" as non-premium. A small adapter is provided below.
    """
    auth_header = req.headers.get("Authorization") or req.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return (None, "free")
    id_token = auth_header.split(" ", 1)[1].strip()
    if not id_token:
        return (None, "free")

    try:
        from firebase_admin import auth as fb_auth
        decoded = fb_auth.verify_id_token(id_token)
        uid = decoded.get("uid")
    except Exception as e:
        logger.info("coach_review: invalid id_token (%s) — downgrading to free", type(e).__name__)
        return (None, "free")

    if not uid:
        return (None, "free")

    try:
        snap = firestore.client().collection("users").document(uid) \
            .collection("coachSubscription").document("current").get()
        if not snap.exists:
            return (uid, "free")
        data = snap.to_dict() or {}
        persisted = (data.get("tier") or "free").lower()
        if persisted not in ("plus", "premium"):
            return (uid, "free")
        # Expiry check applies to BOTH paid tiers.
        expires_at = data.get("expiresAt")
        if expires_at:
            try:
                if hasattr(expires_at, "timestamp") and expires_at.timestamp() < time.time():
                    return (uid, "free")
            except Exception:
                pass
        return (uid, persisted)
    except Exception as e:
        logger.warning("coach_review: subscription lookup failed for %s: %s", uid, e)
        return (uid, "free")


@https_fn.on_request(memory=1024, max_instances=20, timeout_sec=120, cpu=1)
def coach_review(req: https_fn.Request) -> https_fn.Response:
    """Generate an AI Coach review. Tier is verified server-side from the
    Firebase Auth token — the client's `tier` field is ignored."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405,
        )

    try:
        payload = req.get_json(silent=True) or {}
        summary = (payload.get("summary") or "").strip()
        user_input = (payload.get("user_input") or "").strip()
        language_selected = (payload.get("languageSelected") or payload.get("language") or "english").lower()

        if not summary or not user_input:
            return create_response(
                success=False,
                message='Missing required fields',
                error='`summary` and `user_input` are required',
                status_code=400,
            )

        # Length guards — generous but bounded. Prevents accidental or
        # adversarial payloads from blowing past the model's context window
        # or driving up token spend.
        if len(summary) > 12000 or len(user_input) > 4000:
            return create_response(
                success=False,
                message='Payload too large',
                error='Coach review payload exceeds size limits.',
                status_code=413,
            )

        # Server-derived tier (ignores client's claim). Returns one of
        # "free" | "plus" | "premium". Plus and Free both use the light
        # model — Plus gets unlimited use + chat, not a better model.
        uid, tier = _verify_coach_tier(req)
        is_premium = tier == "premium"
        is_paid = tier in ("plus", "premium")

        # Rate limit per UID. Paid tiers get the higher window.
        if not _coach_rate_allow(uid or "", is_paid):
            return create_response(
                success=False,
                message='Rate limit',
                error='Too many coach reviews in a short window. Please try again in a few minutes.',
                status_code=429,
            )

        if is_premium:
            model = COACH_MODEL_PREMIUM
            max_tokens = 1400
        else:
            # Free + Plus → light model.
            model = COACH_MODEL_FREE
            max_tokens = 900

        # Compose the user prompt: domain context + the structured ask.
        user_prompt = f"{summary}\n\n---\n\n{user_input}"
        reply_language = "Thai" if language_selected == "thai" else "English"

        logger.info(
            "coach_review: uid=%s tier=%s model=%s lang=%s summary_chars=%d",
            uid or "anon", tier, model, language_selected, len(summary)
        )

        # Use the shared ChatGPT wrapper. The wrapper already handles
        # retries, circuit breakers, and rate limiting.
        from chatgpt_wrapper import chat_with_gpt
        response_text = chat_with_gpt(
            system_prompt=_COACH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            max_completion_tokens=max_tokens,
            auto_detect_language=False,
            reply_language=reply_language,
        )

        if not response_text or not response_text.strip():
            return create_response(
                success=False,
                message='Empty model response',
                error='The coach is briefly unavailable. Please try again in a moment.',
                status_code=502,
            )

        return create_response(
            data={
                "response": response_text,
                "feedback": response_text,  # back-compat with existing client parse chain
                "model_used": model,
                "tier": tier,
            },
            message='Coach review generated',
        )

    except Exception as e:
        logger.error("coach_review error: %s", e)
        traceback.print_exc()
        return create_response(
            success=False,
            message='Coach review failed',
            error="We couldn't reach your coach right now. Please try again in a moment.",
            status_code=500,
        )


# =========================
# Coach Subscription Verification — IAP receipt validation
# =========================
# Production-grade receipt validation for the EVO Coach Premium subscription.
# Called by the mobile app immediately after a successful App Store / Google
# Play purchase or after a Restore Purchases. The endpoint:
#
#   1. Verifies the user's Firebase Auth ID token (same path as coach_review).
#   2. Calls Apple's verifyReceipt or Google Play's purchases.subscriptions.get
#      to authenticate the receipt against the store of record.
#   3. Extracts the expiry date from the platform response.
#   4. Writes users/{uid}/coachSubscription/current with tier=premium,
#      expiresAt, source, and originalTransactionId.
#
# Why server-side: the device cannot be trusted. A modified client could send
# a forged receipt. Apple/Google sign their receipts and only they can verify
# them. The Firestore write only happens after a successful platform call.
#
# Secrets required (set via Firebase Functions secrets manager):
#   - IAP_APPLE_SHARED_SECRET           App Store Connect → App-Specific Shared Secret
#   - IAP_GOOGLE_SERVICE_ACCOUNT_JSON   Service account JSON with
#                                       androidpublisher.read scope, attached
#                                       to Play Console as a Financial admin.
#
# Both can be empty in development — the endpoint will fail closed (never
# grant premium without verification).

import base64
import json as _json

_APPLE_PROD_URL = "https://buy.itunes.apple.com/verifyReceipt"
_APPLE_SANDBOX_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

def _apple_shared_secret() -> Optional[str]:
    return os.getenv("IAP_APPLE_SHARED_SECRET") or None

def _google_service_account_json() -> Optional[str]:
    return os.getenv("IAP_GOOGLE_SERVICE_ACCOUNT_JSON") or None

def _verify_apple_receipt(receipt_b64: str, sku: str) -> Dict[str, Any]:
    """Verify an iOS receipt with Apple. Returns a normalised dict:
       { ok, expires_ms, original_transaction_id, raw_status }.
       Apple's protocol: try production first, fall back to sandbox if Apple
       tells us 21007 (sandbox receipt sent to prod).
    """
    shared = _apple_shared_secret()
    if not shared:
        return {"ok": False, "error": "Apple shared secret not configured"}

    body = {
        "receipt-data": receipt_b64,
        "password": shared,
        "exclude-old-transactions": True,
    }

    def _post(url: str):
        return requests.post(url, json=body, timeout=20).json()

    try:
        res = _post(_APPLE_PROD_URL)
        if res.get("status") == 21007:
            res = _post(_APPLE_SANDBOX_URL)
    except Exception as e:
        return {"ok": False, "error": f"Apple verify network error: {e}"}

    status = res.get("status")
    if status != 0:
        return {"ok": False, "error": f"Apple status={status}", "raw_status": status}

    # latest_receipt_info is the authoritative list of transactions for
    # auto-renewing subscriptions. Pick the entry matching our SKU with the
    # furthest-out expiry.
    candidates = res.get("latest_receipt_info") or res.get("receipt", {}).get("in_app", []) or []
    matching = [
        c for c in candidates
        if (c.get("product_id") == sku or not sku)
    ]
    if not matching:
        return {"ok": False, "error": "Receipt has no matching subscription", "raw_status": status}

    matching.sort(key=lambda c: int(c.get("expires_date_ms") or 0), reverse=True)
    top = matching[0]
    expires_ms = int(top.get("expires_date_ms") or 0)
    if expires_ms <= 0:
        return {"ok": False, "error": "No expiry on receipt"}

    return {
        "ok": True,
        "expires_ms": expires_ms,
        "original_transaction_id": top.get("original_transaction_id"),
        "raw_status": status,
    }


def _verify_google_subscription(purchase_token: str, sku: str) -> Dict[str, Any]:
    """Verify a Google Play subscription via the Play Developer API.
       Requires a service account JSON in IAP_GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    sa_json = _google_service_account_json()
    if not sa_json:
        return {"ok": False, "error": "Google service account not configured"}

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return {
            "ok": False,
            "error": "google-api-python-client / google-auth not installed. "
                     "Add 'google-api-python-client' and 'google-auth' to requirements.txt.",
        }

    try:
        info = _json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/androidpublisher"]
        )
        service = build("androidpublisher", "v3", credentials=creds, cache_discovery=False)
        # Package name is encoded in app.json / build.gradle. Hardcode the
        # production package name here to keep the function self-contained.
        package_name = os.getenv("ANDROID_PACKAGE_NAME", "com.evoforluanching")
        result = service.purchases().subscriptions().get(
            packageName=package_name,
            subscriptionId=sku,
            token=purchase_token,
        ).execute()
    except Exception as e:
        return {"ok": False, "error": f"Google verify error: {e}"}

    expires_ms = int(result.get("expiryTimeMillis") or 0)
    if expires_ms <= 0:
        return {"ok": False, "error": "No expiry on Google subscription"}

    return {
        "ok": True,
        "expires_ms": expires_ms,
        "original_transaction_id": result.get("orderId"),
        "auto_renewing": bool(result.get("autoRenewing", False)),
    }


@https_fn.on_request(memory=512, max_instances=10, timeout_sec=60, cpu=1)
def verify_coach_subscription(req: https_fn.Request) -> https_fn.Response:
    """Verify an iOS/Android IAP receipt and grant premium in Firestore."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    if req.method != 'POST':
        return create_response(
            success=False, message='Method not allowed',
            error='Only POST method is allowed', status_code=405,
        )

    try:
        # Auth — required (we need a uid to write the subscription doc).
        uid, _ = _verify_coach_tier(req)
        if not uid:
            return create_response(
                success=False, message='Auth required',
                error='Sign in to confirm your purchase.', status_code=401,
            )

        payload = req.get_json(silent=True) or {}
        receipt = (payload.get("receipt") or "").strip()
        platform = (payload.get("platform") or "").lower()
        sku = (payload.get("sku") or "").strip()
        transaction_id = (payload.get("transactionId") or "").strip() or None

        if not receipt or platform not in ("ios", "android") or not sku:
            return create_response(
                success=False, message='Missing fields',
                error='`receipt`, `platform` (ios|android), and `sku` are required.',
                status_code=400,
            )

        if platform == "ios":
            v = _verify_apple_receipt(receipt, sku)
            source = "ios"
        else:
            v = _verify_google_subscription(receipt, sku)
            source = "android"

        if not v.get("ok"):
            logger.warning("verify_coach_subscription failed for %s: %s", uid, v.get("error"))
            return create_response(
                success=False, message='Receipt invalid',
                error=v.get("error") or "Receipt could not be verified.",
                status_code=402,
            )

        expires_ms = int(v["expires_ms"])
        if expires_ms < (time.time() * 1000):
            return create_response(
                success=False, message='Subscription expired',
                error='This subscription has already expired.',
                status_code=402,
            )

        # Map SKU → tier. Plus and Premium are distinct subscriptions in
        # both stores; the SKU is the source of truth for which tier the
        # user just paid for. If the SKU doesn't match any known product
        # we fail closed (don't grant premium for an unknown product).
        sku_lc = (sku or "").lower()
        if "plus" in sku_lc:
            granted_tier = "plus"
        elif "premium" in sku_lc:
            granted_tier = "premium"
        else:
            logger.warning("verify_coach_subscription: unknown sku %s — refusing", sku)
            return create_response(
                success=False, message='Unknown product',
                error='Unrecognised subscription product.',
                status_code=400,
            )

        # Write Firestore subscription doc. From this moment the user is
        # treated as `granted_tier` by useCoachSubscription on the client
        # and by coach_review on the server.
        from firebase_admin import firestore as _fs
        sub_ref = (
            firestore.client()
            .collection("users").document(uid)
            .collection("coachSubscription").document("current")
        )
        sub_ref.set({
            "tier": granted_tier,
            "source": source,
            "sku": sku,
            "originalTransactionId": v.get("original_transaction_id") or transaction_id,
            "expiresAt": _fs.Timestamp.from_seconds(expires_ms // 1000) if hasattr(_fs, "Timestamp") else datetime.utcfromtimestamp(expires_ms // 1000),
            "updatedAt": _fs.SERVER_TIMESTAMP if hasattr(_fs, "SERVER_TIMESTAMP") else datetime.utcnow().isoformat(),
        }, merge=True)

        logger.info(
            "verify_coach_subscription: granted %s uid=%s sku=%s exp=%s",
            granted_tier, uid, sku, expires_ms
        )

        return create_response(
            data={
                "tier": granted_tier,
                "expiresAt": expires_ms,
                "source": source,
            },
            message='Subscription verified',
        )

    except Exception as e:
        logger.error("verify_coach_subscription error: %s", e)
        traceback.print_exc()
        return create_response(
            success=False, message='Verification failed',
            error="We couldn't verify your purchase right now. Please try again.",
            status_code=500,
        )


# =========================
# Async Planner Generation with Job Queue
# =========================

# Firestore collection for persistent job storage
PLANNER_JOBS_COLLECTION = "planner_jobs"

# Local cache for faster reads (optional, reduces Firestore reads)
_planner_jobs_cache: Dict[str, Dict[str, Any]] = {}

def _get_firestore_client():
    """Get Firestore client (lazy initialization)"""
    return firestore.client()

def _create_planner_job(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new planner generation job in Firestore"""
    job_id = f"plan_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    # Estimate generation time
    total_days = request_data.get('totalDays', 30)
    fast_mode = request_data.get('fastMode', False)
    estimated_seconds = 30 + (total_days * (2 if fast_mode else 4))
    
    job = {
        "job_id": job_id,
        "status": "pending",  # pending, processing, completed, failed
        "progress": 0,
        "progress_message": "Job created, waiting to start...",
        "estimated_seconds_remaining": estimated_seconds,
        "created_at": now,
        "updated_at": now,
        "request": request_data,
        "current_stage": "initializing",
        "stages_completed": 0,
        "total_stages": 4,
        "result": None,
        "error": None
    }
    
    # Save to Firestore
    try:
        db = _get_firestore_client()
        db.collection(PLANNER_JOBS_COLLECTION).document(job_id).set(job)
        # Also cache locally for faster access within same instance
        _planner_jobs_cache[job_id] = job
    except Exception as e:
        logger.error(f"Failed to save job to Firestore: {e}")
        # Fallback to cache only if Firestore fails
        _planner_jobs_cache[job_id] = job
    
    return job


def _update_planner_job(job_id: str, updates: Dict[str, Any]):
    """Update a planner job in Firestore"""
    updates["updated_at"] = datetime.utcnow().isoformat()
    
    try:
        db = _get_firestore_client()
        db.collection(PLANNER_JOBS_COLLECTION).document(job_id).update(updates)
        # Update local cache
        if job_id in _planner_jobs_cache:
            _planner_jobs_cache[job_id].update(updates)
    except Exception as e:
        logger.error(f"Failed to update job in Firestore: {e}")
        # Fallback to cache only
        if job_id in _planner_jobs_cache:
            _planner_jobs_cache[job_id].update(updates)


def _get_planner_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a planner job by ID from Firestore"""
    # Check local cache first for faster reads
    if job_id in _planner_jobs_cache:
        return _planner_jobs_cache[job_id]
    
    # Fetch from Firestore
    try:
        db = _get_firestore_client()
        doc = db.collection(PLANNER_JOBS_COLLECTION).document(job_id).get()
        if doc.exists:
            job = doc.to_dict()
            # Cache for future reads
            _planner_jobs_cache[job_id] = job
            return job
    except Exception as e:
        logger.error(f"Failed to get job from Firestore: {e}")
    
    return None


def _planner_job_progress_callback(job_id: str):
    """Returns a callback that writes generation progress to Firestore."""
    def _callback(updates: Dict[str, Any]) -> None:
        payload = dict(updates)
        if "progress_message" in payload:
            payload.setdefault("status", "processing")
        _update_planner_job(job_id, payload)
    return _callback


def _run_planner_generation_background(job_id: str, request_data: Dict[str, Any]):
    """
    Background worker function to generate planner content.
    Called in a separate thread after async job creation returns.
    """
    try:
        gpc = get_generate_planner_content()
        parsed = gpc.GeneratePlannerRequest(**request_data)

        _update_planner_job(job_id, {
            "status": "processing",
            "progress": 5,
            "progress_message": "Starting generation...",
            "current_stage": "initializing",
            "stages_completed": 0,
        })

        chat = gpc.ChatWrapper(gpc.ChatWrapperConfig())
        content = chat.generate(parsed, progress_callback=_planner_job_progress_callback(job_id))

        _update_planner_job(job_id, {
            "status": "completed",
            "progress": 100,
            "progress_message": "Generation complete!",
            "current_stage": "completed",
            "stages_completed": 4,
            "estimated_seconds_remaining": 0,
            "result": content.model_dump(),
        })

        logger.info(f"✓ Background generation completed for job: {job_id}")

    except Exception as e:
        logger.error(f"✗ Background generation failed for job {job_id}: {e}")
        import traceback
        traceback.print_exc()

        error_msg = str(e)
        if hasattr(e, "user_message"):
            error_msg = e.user_message

        _update_planner_job(job_id, {
            "status": "failed",
            "progress": 0,
            "progress_message": "Generation failed",
            "current_stage": "failed",
            "error": error_msg,
        })


# @https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2)
# def start_planner_generation(req: https_fn.Request) -> https_fn.Response:
#     """
#     Start an async planner generation job.
#     Spawns background processing and returns immediately with job ID for polling.
    
#     POST /start_planner_generation
#     Body: Same as generate_planner_content
    
#     Response: {
#         "success": true,
#         "data": {
#             "jobId": "plan_abc123",
#             "status": "processing",
#             "estimatedSeconds": 120,
#             "message": "Your planner is being generated."
#         }
#     }
    
#     Then poll GET /get_planner_job_status?jobId=plan_abc123 for progress updates.
#     """
#     if req.method == 'OPTIONS':
#         return handle_preflight_request()
    
#     if req.method != 'POST':
#         return create_response(
#             success=False,
#             message='Method not allowed',
#             error='Only POST method is allowed',
#             status_code=405
#         )
    
#     try:
#         request_data = req.get_json() or {}
        
#         # Validate request using the GeneratePlannerRequest model
#         gpc = get_generate_planner_content()
#         parsed = gpc.GeneratePlannerRequest(**request_data)
        
#         # Create job
#         job = _create_planner_job(parsed.model_dump())
#         job_id = job["job_id"]
        
#         logger.info(f"Created planner job: {job_id} for {parsed.totalDays}-day {parsed.category} plan")
        
#         # Start background generation in a thread
#         import threading
#         worker_thread = threading.Thread(
#             target=_run_planner_generation_background,
#             args=(job_id, parsed.model_dump()),
#             daemon=True
#         )
#         worker_thread.start()
        
#         logger.info(f"Background generation thread started for job: {job_id}")
        
#         return create_response(
#             data={
#                 "jobId": job_id,
#                 "status": "processing",
#                 "estimatedSeconds": job["estimated_seconds_remaining"],
#                 "message": "Your planner is being generated. Poll the status endpoint for updates.",
#                 "pollEndpoint": f"/get_planner_job_status?jobId={job_id}",
#                 "resultEndpoint": f"/get_planner_job_result?jobId={job_id}"
#             },
#             message="Planner generation started"
#         )
        
#     except ValidationError as ve:
#         errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
#         return create_response(
#             success=False,
#             message='Invalid request parameters',
#             error=str(errors),
#             status_code=400
#         )
#     except Exception as e:
#         logger.error(f"Error starting planner job: {e}")
#         return create_response(
#             success=False,
#             message='Failed to start planner generation',
#             error=str(e),
#             status_code=500
#         )


@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2)
def process_planner_job(req: https_fn.Request) -> https_fn.Response:
    """
    Process a planner generation job (called internally or by scheduler).
    This is the actual generation endpoint that does the work.
    
    POST /process_planner_job
    Body: { "jobId": "plan_abc123" }
    """
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    job_id = None
    gpc = None
    try:
        data = req.get_json() or {}
        job_id = data.get('jobId')
        
        if not job_id:
            return create_response(
                success=False,
                message='Missing jobId',
                error='jobId is required',
                status_code=400
            )
        
        job = _get_planner_job(job_id)
        if not job:
            return create_response(
                success=False,
                message='Job not found',
                error=f'No job found with ID: {job_id}',
                status_code=404
            )
        
        if job.get("status") == "completed" and job.get("result"):
            return create_response(
                data={"jobId": job_id, "status": "completed"},
                message="Planner generation already completed",
            )

        if job.get("status") == "processing" and job.get("progress", 0) > 10:
            return create_response(
                data={"jobId": job_id, "status": "processing", "progress": job.get("progress")},
                message="Planner generation already in progress",
            )

        gpc = get_generate_planner_content()
        request_data = job["request"]

        _run_planner_generation_background(job_id, request_data)
        job_after = _get_planner_job(job_id)
        if not job_after:
            raise RuntimeError("Job record missing after generation")
        if job_after.get("status") == "failed":
            raise gpc.PlannerGenerationError(
                job_after.get("error") or "Generation failed",
                job_after.get("error") or "Generation failed",
            )
        if job_after.get("status") != "completed":
            raise RuntimeError(
                f"Generation finished in unexpected state: {job_after.get('status')}"
            )

        logger.info(f"Completed planner job: {job_id}")

        return create_response(
            data={"jobId": job_id, "status": "completed"},
            message="Planner generation completed",
        )
        
    except Exception as e:
        gpc_mod = gpc or get_generate_planner_content()
        user_msg = getattr(e, "user_message", None) or str(e)
        if isinstance(e, gpc_mod.PlannerGenerationError):
            if job_id:
                _update_planner_job(job_id, {
                    "status": "failed",
                    "progress_message": "Generation failed",
                    "error": user_msg,
                })
            return create_response(
                success=False,
                message='Generation failed',
                error=user_msg,
                status_code=500,
            )
        logger.error(f"Error processing planner job: {e}")
        if job_id:
            _update_planner_job(job_id, {
                "status": "failed",
                "progress_message": "Generation failed",
                "error": str(e),
            })
        return create_response(
            success=False,
            message='Generation failed',
            error=str(e),
            status_code=500,
        )


def _extract_planner_job_id(req: https_fn.Request) -> Optional[str]:
    job_id = req.args.get('jobId') or req.args.get('job_id')
    if not job_id:
        try:
            data = req.get_json(silent=True, force=True) or {}
            job_id = data.get('jobId') or data.get('job_id')
        except Exception:
            pass
    if not job_id:
        try:
            job_id = req.form.get('jobId') or req.form.get('job_id')
        except Exception:
            pass
    return job_id


@https_fn.on_request(memory=256, max_instances=20, timeout_sec=10, cpu=1)
def get_planner_job_status(req: https_fn.Request) -> https_fn.Response:
    """Get status/progress for an async planner generation job."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    job_id = _extract_planner_job_id(req)
    if not job_id:
        return create_response(
            success=False,
            message='Missing jobId parameter',
            error='jobId is required (as query parameter or in JSON body)',
            status_code=400,
        )

    job = _get_planner_job(job_id)
    if not job:
        return create_response(
            success=False,
            message='Job not found',
            error=f'No job found with ID: {job_id}',
            status_code=404,
        )

    response_data = {
        "jobId": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "progressMessage": job["progress_message"],
        "currentStage": job["current_stage"],
        "stagesCompleted": job["stages_completed"],
        "totalStages": job["total_stages"],
        "estimatedSecondsRemaining": job["estimated_seconds_remaining"],
        "createdAt": job["created_at"],
        "updatedAt": job["updated_at"],
    }

    if job["status"] == "failed":
        response_data["error"] = job.get("error")

    if job["status"] == "completed" and job.get("result"):
        result = job["result"]
        response_data["resultSummary"] = {
            "planName": result.get("planName"),
            "category": result.get("category"),
            "totalDays": result.get("totalDays"),
            "ready": True,
        }

    return create_response(data=response_data, message="Job status retrieved")


@https_fn.on_request(memory=512, max_instances=10, timeout_sec=30, cpu=1)
def get_planner_job_result(req: https_fn.Request) -> https_fn.Response:
    """Get full PlannerContent for a completed async job."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    job_id = _extract_planner_job_id(req)
    if not job_id:
        return create_response(
            success=False,
            message='Missing jobId parameter',
            error='jobId is required (as query parameter or in JSON body)',
            status_code=400,
        )

    job = _get_planner_job(job_id)
    if not job:
        return create_response(
            success=False,
            message='Job not found',
            error=f'No job found with ID: {job_id}',
            status_code=404,
        )

    if job["status"] != "completed":
        return create_response(
            success=False,
            message='Job not completed',
            error=f'Job is still {job["status"]}. Progress: {job["progress"]}%',
            status_code=400,
            data={
                "status": job["status"],
                "progress": job["progress"],
                "progressMessage": job.get("progress_message"),
            },
        )

    return create_response(
        data=job["result"],
        message="Planner content retrieved successfully",
    )


@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2)  # 9 minutes timeout
def generate_planner_content_async(req: https_fn.Request) -> https_fn.Response:
    """
    Starts planner generation in the background and returns a job ID immediately.
    Poll get_planner_job_status, then fetch get_planner_job_result when completed.
    """
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    
    if req.method != 'POST':
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only POST method is allowed',
            status_code=405
        )
    
    gpc = get_generate_planner_content()
    job_id = None
    try:
        request_data = req.get_json() or {}
        request_data.setdefault("skipContextExtraction", False)
        
        parsed = gpc.GeneratePlannerRequest(**request_data)
        job = _create_planner_job(parsed.model_dump())
        job_id = job["job_id"]
        
        logger.info(f"Starting async generation job: {job_id}")
        
        _update_planner_job(job_id, {
            "status": "pending",
            "progress": 5,
            "progress_message": "Job queued — starting shortly...",
            "current_stage": "initializing",
            "stages_completed": 0,
        })

        # Do not run generation in a daemon thread here — Cloud Run stops CPU after this
        # response returns and the job stays at 5% forever. The mobile app calls
        # process_planner_job (kickoffPlannerJobProcessing) to run the worker.

        return create_response(
            data={
                "jobId": job_id,
                "status": "processing",
                "estimatedSeconds": job["estimated_seconds_remaining"],
            },
            message="Planner generation started",
            metadata={
                "jobId": job_id,
                "status": "processing",
                "totalDays": parsed.totalDays,
                "category": parsed.category,
                "fastMode": parsed.fastMode,
            },
        )
        
    except ValidationError as ve:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
        return create_response(
            success=False,
            message='Invalid request parameters',
            error=str(errors),
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error queueing async generation job: {e}")
        import traceback
        traceback.print_exc()
        
        if job_id:
            _update_planner_job(job_id, {
                "status": "failed",
                "progress_message": "Generation failed",
                "error": str(e)
            })
        
        return create_response(
            success=False,
            message='Generation failed',
            error=str(e),
            status_code=500,
            metadata={"jobId": job_id}
        )


#################### EVO ChatGPT API Endpoints ######################

# Summarize planner data using ChatGPT
@https_fn.on_request(memory=1024, max_instances=3, cpu=1)
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
        
        pu = get_planner_utils()
        summary = pu.summarize_plan(data['planner_data'], language)
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
@https_fn.on_request(memory=1024, max_instances=3, cpu=1)
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
        if 'todo_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='todo_data is required',
                status_code=400
            )

        todo_data = data.get('todo_data')
        if not isinstance(todo_data, dict):
            return create_response(
                success=False,
                message='Invalid todo_data',
                error='todo_data must be an object',
                status_code=400
            )

        # Build a stronger, contextual query for more accurate responses.
        raw_user_query = data.get('user_update')
        user_query = raw_user_query.strip() if isinstance(raw_user_query, str) and raw_user_query.strip() else "Help me understand this todo and what to do next."

        # Optional short chat history from client to preserve intent.
        chat_history = data.get('chat_history', [])
        history_lines: List[str] = []
        if isinstance(chat_history, list):
            for message in chat_history[-8:]:
                if not isinstance(message, dict):
                    continue
                role = message.get('role')
                text = message.get('text')
                if role in {'user', 'assistant'} and isinstance(text, str) and text.strip():
                    history_lines.append(f"{role}: {text.strip()}")

        if history_lines:
            history_text = "\n".join(history_lines)
            enriched_query = (
                f"Latest user question: {user_query}\n"
                f"Recent chat context:\n{history_text}\n"
                "Use recent context only if relevant to the latest question."
            )
        else:
            enriched_query = user_query

        # Normalize language field to avoid accidental unsupported values.
        language = data.get('language', 'thai')
        if not isinstance(language, str) or not language.strip():
            language = 'thai'
        else:
            language = language.strip().lower()

        # Get information about todo_data using AI assistant
        pu = get_planner_utils()
        information = pu.get_todo_information(enriched_query, todo_data, language)
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
@https_fn.on_request(memory=1024, max_instances=3, cpu=1)
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
        
        pu = get_planner_utils()
        # Optional behavior-grounding context (see docs/ORCHESTRATION.md
        # "Relevance second" — coaching should reference real streaks /
        # last-week completion rather than generic motivational copy).
        identity_context = data.get('identity_context')
        last_week_completion_rate = data.get('last_week_completion_rate')
        response = pu.respond_to_user_input(
            data['user_input'],
            data['summary'],
            identity_context=identity_context,
            last_week_completion_rate=last_week_completion_rate,
        )
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

# Encourage the user to start the day using ChatGPT (when user_id present, RAG is consulted before generating)
@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)  # 5 min timeout for ChatGPT + RAG
def encourage_in_the_morning(req: https_fn.Request) -> https_fn.Response:
    """Encourage the user to start the day. If user_id is provided, look into RAG (user todo memory) before generating; if missing/empty, don't include RAG context."""
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
        
        if 'today_todo_list_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='today_todo_list_data is required',
                status_code=400
            )
        
        user_context = None
        month_context = None
        user_id = data.get('user_id')
        if user_id and isinstance(user_id, str) and user_id.strip():
            user_id = user_id.strip()
            try:
                from user_memory import retrieve_user_context
                user_context = retrieve_user_context(user_id, "morning encouragement today tasks", top_k=5)
            except Exception as e:
                logger.warning("RAG retrieval failed in encourage_in_the_morning: %s", e)
            month_context = _month_context_for_user(user_id, data)
        pu = get_planner_utils()
        response = pu.message_in_the_morning(
            today_todo_list_data=data['today_todo_list_data'],
            language=data.get('languageSelected', 'thai'),
            user_context=user_context,
            month_context=month_context,
            earned_runes=data.get('earned_runes'),
            behavior_stats=data.get('behavior_stats'),
            identity_context=data.get('identity_context'),
        )
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

# Summarize the end of the week using ChatGPT and suggest rest (when user_id present, RAG is consulted before generating)
@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def summarize_end_of_the_week(req: https_fn.Request) -> https_fn.Response:
    """Summarize the end of the week and suggest rest. If user_id is provided, look into RAG before generating; if missing/empty, don't include RAG context."""
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
        
        if 'week_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='week_data is required',
                status_code=400
            )
        
        language = data.get('language', 'thai')
        logger.info(f"Summarizing end of week data in language: {language}")
        
        user_context = None
        month_context = None
        user_id = data.get('user_id')
        if user_id and isinstance(user_id, str) and user_id.strip():
            user_id = user_id.strip()
            try:
                from user_memory import retrieve_user_context
                user_context = retrieve_user_context(user_id, "end of week rest recharge", top_k=5)
            except Exception as e:
                logger.warning("RAG retrieval failed in summarize_end_of_the_week: %s", e)
            month_context = _month_context_for_user(user_id, data)
        pu = get_planner_utils()
        rest_suggestions = pu.summarize_end_of_the_week_at_friday(
            week_data=data['week_data'],
            language=language,
            user_context=user_context,
            month_context=month_context
        )
        
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
        
# Summarize next week's plan and provide preparation suggestions (when user_id present, RAG is consulted before generating)
@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def summarize_next_week(req: https_fn.Request) -> https_fn.Response:
    """Summarize next week and provide preparation suggestions. If user_id is provided, look into RAG before generating; if missing/empty, don't include RAG context."""
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
        
        if 'week_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='week_data is required',
                status_code=400
            )
        
        language = data.get('language', 'thai')
        logger.info(f"Summarizing next week data in language: {language}")
        
        user_context = None
        month_context = None
        user_id = data.get('user_id')
        if user_id and isinstance(user_id, str) and user_id.strip():
            user_id = user_id.strip()
            try:
                from user_memory import retrieve_user_context
                user_context = retrieve_user_context(user_id, "next week preparation", top_k=5)
            except Exception as e:
                logger.warning("RAG retrieval failed in summarize_next_week: %s", e)
            month_context = _month_context_for_user(user_id, data)
        pu = get_planner_utils()
        preparation_suggestions = pu.summarize_next_week_at_sunday(
            week_data=data['week_data'],
            language=language,
            user_context=user_context,
            month_context=month_context
        )
        
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


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def suggest_schedule_optimization(req: https_fn.Request) -> https_fn.Response:
    """Use RAG (user memory) + current schedule to suggest how to optimize the user's schedule.
    When user_id is provided, retrieves relevant context and gives personalized optimization tips.

    Request body:
        - schedule_data: List of todo/event items (required), e.g. today's list or week_data
        - user_id: Optional; if set, RAG context is retrieved to personalize suggestions
        - language: Optional (default 'thai')
        - scope: Optional 'day' or 'week' (default 'day')

    Returns:
        - response: Schedule optimization suggestions text
    """
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
        if 'schedule_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='schedule_data is required',
                status_code=400
            )
        schedule_data = data['schedule_data']
        if not isinstance(schedule_data, list):
            return create_response(
                success=False,
                message='Invalid schedule_data',
                error='schedule_data must be a list of todo/event items',
                status_code=400
            )
        user_context = None
        user_id = data.get('user_id')
        if user_id and isinstance(user_id, str) and user_id.strip():
            try:
                from user_memory import retrieve_user_context
                user_context = retrieve_user_context(
                    user_id.strip(),
                    "schedule habits preferences workload optimization",
                    top_k=5
                )
            except Exception as e:
                logger.warning("RAG retrieval failed in suggest_schedule_optimization: %s", e)
        language = data.get('language', 'thai')
        scope = data.get('scope', 'day')
        if scope not in ('day', 'week'):
            scope = 'day'
        month_context = _month_context_for_user(data.get('user_id'), data)
        pu = get_planner_utils()
        suggestions = pu.suggest_schedule_optimizations(
            schedule_data=schedule_data,
            language=language,
            user_context=user_context,
            month_context=month_context,
            scope=scope,
        )
        if suggestions is None:
            return create_response(
                success=False,
                message='No suggestions generated',
                error='Schedule data may be empty or generation failed',
                status_code=500
            )
        return create_response(
            data={'response': suggestions},
            message='Schedule optimization suggestions generated successfully',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in suggest_schedule_optimization: %s", str(e))
        return create_response(
            success=False,
            message='Schedule optimization failed',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def analyze_user_todos(req: https_fn.Request) -> https_fn.Response:
    """Analyze a user's todo list (from RAG) for: prevent overload, protect deep work time,
    maintain goal momentum, and give practical advice. Uses embedded todo memories for the user_id.

    Request body:
        - user_id: User identifier (required)
        - schedule_data: Optional list of current todo/event items (focus analysis on this schedule)
        - language: Optional (default 'thai')

    Returns:
        - response: Analysis text (overload, deep work, goals, practical suggestions)
    """
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
        user_id = data.get('user_id')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        user_id = user_id.strip()
        intent_profile = _get_intent_profile_for_user(user_id)
        # Retrieve user's todo list context from RAG (broad query for analysis)
        user_context = []
        try:
            from user_memory import retrieve_user_context
            user_context = retrieve_user_context(
                user_id,
                "todos schedule workload deep work goals habits",
                top_k=10
            )
        except Exception as e:
            logger.warning("RAG retrieval failed in analyze_user_todos: %s", e)
        if intent_profile:
            user_context.append(
                "intent_profile_context: " + json.dumps(intent_profile, ensure_ascii=False, default=str)
            )
        if not user_context:
            return create_response(
                success=False,
                message='No todo context found',
                error='No embedded todo memories for this user. Embed todos first (e.g. embedUserTodos).',
                status_code=404
            )
        schedule_data = data.get('schedule_data')
        if schedule_data is not None and not isinstance(schedule_data, list):
            schedule_data = None
        language = data.get('language', 'thai')
        month_context = _month_context_for_user(user_id, data)
        pu = get_planner_utils()
        analysis = pu.analyze_todo_list(
            user_context=user_context,
            language=language,
            schedule_data=schedule_data,
            month_context=month_context,
        )
        if analysis is None:
            return create_response(
                success=False,
                message='Analysis failed',
                error='Could not generate analysis',
                status_code=500
            )
        return create_response(
            data={'response': analysis},
            message='Todo list analysis generated successfully',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in analyze_user_todos: %s", str(e))
        return create_response(
            success=False,
            message='Analysis failed',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
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
        
        # Validate input size to prevent timeout (limit to 10000 characters)
        todos_data = data['this_year_todos_data']
        if isinstance(todos_data, str) and len(todos_data) > 10000:
            return create_response(
                success=False,
                message='Input too large',
                error='Input data is too large. Please provide a shorter summary (max 10000 characters).',
                status_code=400
            )
        
        language = data.get('languageSelected', 'thai')
        logger.info(f"Summarizing this year's todos in language: {language}")
        month_context = _month_context_for_user(data.get('user_id'), data)
        identity_context = data.get('identity_context')
        last_week_completion_rate = data.get('last_week_completion_rate')
        pu = get_planner_utils()
        title, summary = pu.summarize_this_year_todos_message(
            this_year_todos_data=data['this_year_todos_data'],
            language=language,
            month_context=month_context,
            identity_context=identity_context,
            last_week_completion_rate=last_week_completion_rate,
        )
        return create_response(
            data={'title': title, 'summary': summary},
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


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def summary_this_month_todos(req: https_fn.Request) -> https_fn.Response:
    """Summarize this month's todos using ChatGPT"""
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
        if 'this_month_todos_data' not in data:
            return create_response(
                success=False,
                message='Missing required field',
                error='month_data is required',
                status_code=400
            )
        
        # Validate input size to prevent timeout (limit to 10000 characters)
        todos_data = data['this_month_todos_data']
        if isinstance(todos_data, str) and len(todos_data) > 10000:
            return create_response(
                success=False,
                message='Input too large',
                error='Input data is too large. Please provide a shorter summary (max 10000 characters).',
                status_code=400
            )
        
        language = data.get('languageSelected', 'thai')
        logger.info(f"Summarizing this month's todos in language: {language}")
        month_context = _month_context_for_user(data.get('user_id'), data)
        identity_context = data.get('identity_context')
        last_week_completion_rate = data.get('last_week_completion_rate')
        pu = get_planner_utils()
        title, summary = pu.summarize_this_month_todos_message(
            this_month_todos_data=data['this_month_todos_data'],
            language=language,
            month_context=month_context,
            identity_context=identity_context,
            last_week_completion_rate=last_week_completion_rate,
        )
        return create_response(
            data={'title': title, 'summary': summary},
            message='This month\'s todos summary generated successfully',
            status_code=200,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_this_month_todos: {str(e)}")
        return create_response(
            success=False,
            message='This month\'s todos summary generation failed',
            error=f'Failed to generate this month\'s todos summary: {str(e)}',
            status_code=500
        )

# Generate rune-based daily guidance using ChatGPT
@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def todo_fate_prediction(req: https_fn.Request) -> https_fn.Response:
    """Generate rune-based daily guidance (Elder Futhark by default) from the user's todos using ChatGPT."""
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
        required_fields = ['languageSelected']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    message='Missing required field',
                    error=f'{field} is required',
                    status_code=400
                )

        divination_system = data.get('divination_system', 'elder_futhark')
        output_style = data.get('output_style', 'brief')
        earned_runes = data.get('earned_runes')
        behavior_stats = data.get('behavior_stats')

        if 'todo_data' in data:
            todo_data = data['todo_data']
            language = data['languageSelected']
        else:
            todo_data = []
            language = 'english'
        pu = get_planner_utils()
        response = pu.predict_today_todo_fate(
            todo_data=todo_data,
            language=language,
            divination_system=divination_system,
            earned_runes=earned_runes if isinstance(earned_runes, list) else None,
            behavior_stats=behavior_stats if isinstance(behavior_stats, dict) else None,
            output_style=output_style,
        )
        return create_response(
            data={'response': response},
            message='Rune guidance generated successfully'
        )
    except Exception as e:
        from chatgpt_wrapper import RateLimitExceededError
        if isinstance(e, RateLimitExceededError):
            retry_after = getattr(e, 'retry_after', None)
            extra_headers = {'Retry-After': str(int(retry_after))} if retry_after else None
            return create_response(
                success=False,
                message='Rate limit exceeded',
                error='Too many requests. Please try again later.',
                status_code=429,
                extra_headers=extra_headers
            )
        logger.error(f"Error in todo_fate_prediction: {str(e)}")
        return create_response(
            success=False,
            message='Todo fate prediction generation failed',
            error=f'Failed to generate todo fate prediction: {str(e)}',
            status_code=500
        )

@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def generate_todo_data_from_user_input(req: https_fn.Request) -> https_fn.Response:
    """Convert natural language user input into structured todo data using AI.
    
    Supports extracting multiple todos from a single input message.
    
    Request body:
        - user_input: Natural language description of todo operations (required)
        - languageSelected: Language for processing (default: 'thai')
        - current_date: Current date in ISO format for context (optional)
        - timezone: User's timezone (optional, default: 'Asia/Bangkok')
        - existing_todos: Optional list of existing todos used to match update/delete targets
    
    Returns:
        - actions: List of operations (create/update/delete) with target ids for update/delete
        - todos: Backward compatible list of create/update todo payloads
        - count: Number of actions
    """
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
        if 'user_input' not in data or not data['user_input']:
            return create_response(
                success=False,
                message='Missing required field',
                error='user_input is required',
                status_code=400
            )
        
        user_input = data['user_input']
        user_id = data.get('user_id')
        language = data.get('languageSelected', 'thai')
        current_date = data.get('current_date', datetime.now().isoformat())
        timezone = data.get('timezone', 'Asia/Bangkok')
        
        existing_todos = data.get('existing_todos', [])
        chat_history = data.get('chat_history', [])
        intent_profile = _get_intent_profile_for_user(user_id)
        enriched_user_input = user_input
        history_lines = []
        if isinstance(chat_history, list):
            for message in chat_history[-10:]:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                text = message.get("text")
                if role in ("user", "assistant") and isinstance(text, str) and text.strip():
                    history_lines.append(f"{role}: {text.strip()}")
        if intent_profile:
            intent_summary = {
                "topIntentCategories": intent_profile.get("topIntentCategories", []),
                "preferredTimeBlocks": intent_profile.get("preferredTimeBlocks", []),
                "preferredReminderLeadTime": intent_profile.get("preferredReminderLeadTime", []),
                "socialVisibilityTendency": intent_profile.get("socialVisibilityTendency", []),
            }
            enriched_user_input = (
                f"{user_input}\n\n"
                f"[USER_INTENT_PROFILE]\n{json.dumps(intent_summary, ensure_ascii=False, default=str)}\n"
                "Use this profile to better align suggested todo actions with the user's preferences."
            )
        if history_lines:
            enriched_user_input = (
                f"{enriched_user_input}\n\n"
                f"[RECENT_CHAT_HISTORY]\n{chr(10).join(history_lines)}\n"
                "Use this conversation history to infer user intent, constraints, and whether they want rest vs challenge."
            )

        # Use todo_generator module for action extraction
        tg = get_todo_generator()
        action_result = tg.extract_todo_actions_from_text(
            user_input=enriched_user_input,
            language=language,
            current_date=current_date,
            timezone=timezone,
            existing_todos=existing_todos if isinstance(existing_todos, list) else []
        )
        actions = action_result.get('actions', [])
        todos = action_result.get('todos', [])

        def _to_date_safe(date_value):
            if not date_value:
                return None
            if isinstance(date_value, datetime):
                return date_value.date()
            if isinstance(date_value, str):
                try:
                    return datetime.fromisoformat(date_value.replace("Z", "+00:00")).date()
                except Exception:
                    try:
                        return datetime.strptime(date_value[:10], "%Y-%m-%d").date()
                    except Exception:
                        return None
            return None

        def _build_energy_analytics(existing, extracted_actions, lang):
            def _compact_energy_summary(text, language_code):
                cleaned = " ".join(str(text or "").replace("\n", " ").split())
                if not cleaned:
                    return ""
                first_sentence = cleaned
                sentence_separators = [". ", "! ", "? ", "。", "！", "？"]
                for separator in sentence_separators:
                    if separator in cleaned:
                        first_sentence = cleaned.split(separator)[0].strip()
                        break
                concise = first_sentence if len(first_sentence) >= 40 else cleaned
                max_chars = 120 if str(language_code).lower() == "thai" else 140
                if len(concise) > max_chars:
                    concise = f"{concise[:max_chars - 1].rstrip()}…"
                return concise

            safe_existing = existing if isinstance(existing, list) else []
            safe_actions = extracted_actions if isinstance(extracted_actions, list) else []
            today = datetime.now().date()

            creates = 0
            updates = 0
            deletes = 0
            for action in safe_actions:
                a = str((action or {}).get("action", "create")).lower()
                if a == "delete":
                    deletes += 1
                elif a == "update":
                    updates += 1
                else:
                    creates += 1

            upcoming_7d = 0
            no_time_count = 0
            for todo in safe_existing:
                if not isinstance(todo, dict):
                    continue
                todo_date = _to_date_safe(todo.get("date"))
                if todo_date and 0 <= (todo_date - today).days <= 7:
                    upcoming_7d += 1
                if not str(todo.get("start", "")).strip():
                    no_time_count += 1

            net_change = creates - deletes
            projected_7d = max(0, upcoming_7d + net_change)

            if projected_7d >= 18:
                level = "high"
            elif projected_7d <= 4:
                level = "low"
            else:
                level = "balanced"

            # Build dynamic analysis context for AI (instead of fixed hard-coded sentences).
            action_context = (
                f"User request: {user_input}\n"
                f"Planned action mix => create:{creates}, update:{updates}, delete:{deletes}\n"
                f"Upcoming 7d existing todos: {upcoming_7d}\n"
                f"Projected 7d todos after actions: {projected_7d}\n"
                f"Todos without explicit start time: {no_time_count}\n"
                "Return one practical energy suggestion in one short sentence."
            )
            conversation_context = history_lines[-6:] if isinstance(history_lines, list) else []
            analytics_context = [action_context, *conversation_context]

            summary = None
            try:
                pu = get_planner_utils()
                # Reuse planner AI stack to produce personalized energy guidance.
                summary = pu.analyze_todo_list(
                    user_context=analytics_context,
                    language=lang,
                    schedule_data=safe_existing
                )
            except Exception as ai_err:
                logger.warning("Dynamic energy analytics generation failed, using fallback: %s", ai_err)

            if not summary:
                if level == "high":
                    summary = (
                        "Your week looks heavy. Add short recovery breaks and trim low-priority tasks to keep energy stable."
                        if lang != "thai"
                        else "ภาระงานสัปดาห์นี้ค่อนข้างแน่น แนะนำเว้นช่วงพักและลดงานที่ไม่เร่งด่วนเพื่อรักษาพลังงาน"
                    )
                elif level == "low":
                    summary = (
                        "Your load is light. This is a good window to add one meaningful challenge."
                        if lang != "thai"
                        else "ภาระงานช่วงนี้ค่อนข้างเบา เหมาะกับการเพิ่มความท้าทายใหม่เล็ก ๆ"
                    )
                else:
                    summary = (
                        "Your workload is balanced. Keep the rhythm and protect focus time for key tasks."
                        if lang != "thai"
                        else "สมดุลงานโดยรวมกำลังดี รักษาจังหวะเดิมและกันเวลาโฟกัสสำหรับงานสำคัญ"
                    )
            summary = _compact_energy_summary(summary, lang)

            return {
                "level": level,
                "summary": summary,
                "stats": {
                    "existing_count": len(safe_existing),
                    "upcoming_7d": upcoming_7d,
                    "projected_7d": projected_7d,
                    "actions_create": creates,
                    "actions_update": updates,
                    "actions_delete": deletes,
                }
            }

        analytics = _build_energy_analytics(existing_todos, actions, language)
        
        return create_response(
            data={
                'actions': actions,
                'todos': todos,
                'count': len(actions),
                'analytics': analytics
            },
            message=f'Successfully extracted {len(actions)} todo action(s)',
            success=True,
            status_code=200
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in generate_todo_data_from_user_input: {str(e)}")
        return create_response(
            success=False,
            message='Invalid input',
            error=str(e),
            status_code=400
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in generate_todo_data_from_user_input: {str(e)}")
        return create_response(
            success=False,
            message='AI response parsing failed',
            error=f'Failed to parse AI response: {str(e)}',
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error in generate_todo_data_from_user_input: {str(e)}")
        return create_response(
            success=False,
            message='Todo data generation failed',
            error=f'Failed to generate todo data: {str(e)}',
            status_code=500
        )

@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def create_rag_todo_users(req: https_fn.Request) -> https_fn.Response:
    """RAG-augmented todo extraction: convert natural language to structured todos,
    optionally using context (e.g. existing user todos) to avoid duplicates and align output.

    Request body:
        - user_input: Natural language description of one or more todos (required)
        - user_id: Optional; if set, context is retrieved from FAISS user memory (RAG)
        - context: Optional list of existing todos or text chunks (overrides user_id retrieval)
        - memory_top_k: Optional; when using user_id, how many memories to retrieve (default 5)
        - embed_new_todos: Optional; if true and user_id is set, extracted todos are embedded into user memory
        - languageSelected: Language for processing (default: 'thai')
        - current_date: Current date in ISO format (optional)
        - timezone: User's timezone (optional, default: 'Asia/Bangkok')

    Returns:
        - todos: List of structured todo objects
        - count: Number of todos extracted
    """
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

        if 'user_input' not in data or not data['user_input']:
            return create_response(
                success=False,
                message='Missing required field',
                error='user_input is required',
                status_code=400
            )

        user_input = data['user_input']
        context = data.get('context')  # optional: list of strings or todo-like dicts
        user_id = data.get('user_id')  # optional: retrieve context from FAISS user memory
        memory_top_k = data.get('memory_top_k', 5)
        language = data.get('languageSelected', 'thai')
        current_date = data.get('current_date', datetime.now().isoformat())
        timezone = data.get('timezone', 'Asia/Bangkok')

        from rag_todo_users import extract_todos_with_rag
        todos = extract_todos_with_rag(
            user_input=user_input,
            context=context if isinstance(context, list) else None,
            user_id=user_id if isinstance(user_id, str) and user_id.strip() else None,
            memory_top_k=min(20, max(1, int(memory_top_k))) if isinstance(memory_top_k, (int, float)) else 5,
            language=language,
            current_date=current_date,
            timezone=timezone,
        )

        # Optionally embed new todos into user memory (RAG) so they are used in future retrieval
        if todos and user_id and isinstance(user_id, str) and user_id.strip() and data.get('embed_new_todos'):
            try:
                from user_memory import add_todos_as_memories
                add_todos_as_memories(user_id.strip(), todos, mode="per_todo")
            except Exception as e:
                logger.warning("Failed to embed new todos in create_rag_todo_users: %s", e)

        return create_response(
            data={'todos': todos, 'count': len(todos)},
            message=f'Successfully extracted {len(todos)} todo(s)',
            success=True,
            status_code=200
        )

    except ValueError as e:
        logger.warning(f"Validation error in create_rag_todo_users: {str(e)}")
        return create_response(
            success=False,
            message='Invalid input',
            error=str(e),
            status_code=400
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in create_rag_todo_users: {str(e)}")
        return create_response(
            success=False,
            message='AI response parsing failed',
            error=f'Failed to parse AI response: {str(e)}',
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error in create_rag_todo_users: {str(e)}")
        return create_response(
            success=False,
            message='RAG todo extraction failed',
            error=f'Failed to extract todos: {str(e)}',
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def add_user_memory(req: https_fn.Request) -> https_fn.Response:
    """Add a memory for a user (FAISS + OpenAI embeddings). Stored in-memory per instance.

    Request body:
        - user_id: User identifier (required)
        - text: Memory content to embed and store (required)
        - metadata: Optional dict of extra fields to store with the memory

    Returns:
        - success, message
    """
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
        user_id = data.get('user_id')
        text = data.get('text')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        if not text or not isinstance(text, str) or not text.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='text is required',
                status_code=400
            )
        metadata = data.get('metadata')
        if metadata is not None and not isinstance(metadata, dict):
            metadata = None
        from user_memory import add_memory
        add_memory(user_id=user_id.strip(), text=text, metadata=metadata or None)
        return create_response(
            data=None,
            message='Memory added successfully',
            success=True,
            status_code=200
        )
    except ValueError as e:
        return create_response(success=False, message='Invalid input', error=str(e), status_code=400)
    except Exception as e:
        logger.error("Error in add_user_memory: %s", str(e))
        return create_response(
            success=False,
            message='Failed to add memory',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def embed_user_todos(req: https_fn.Request) -> https_fn.Response:
    """Embed a user's todo list into RAG (FAISS user memory). Use for bulk import (e.g. last year) or new todos.

    Request body:
        - user_id: User identifier (required)
        - todos: List of todo objects (required). Each should include id or todoId (for update/delete), plus title, detail, date, start, typeOfTodo, etc.
        - replace_todo_ids: Optional list of todo ids to replace. If provided, those memories are marked deleted first, then todos are embedded (use this to update: same id in replace_todo_ids and in the updated todo).
        - mode: Optional "per_todo" or "per_month". Default "per_todo".

    To update (title, date, or any field changed): send replace_todo_ids: [id] and todos: [the full updated todo] — same id, but with new title, date, start, detail, etc. We remove the old memory and embed the new content so RAG sees the updated todo. To delete: use deleteUserTodoMemories.

    Returns:
        - added: Number of memories added
        - replaced: Number of old memories marked deleted (if replace_todo_ids was sent)
    """
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
        user_id = data.get('user_id')
        todos = data.get('todos')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        if not isinstance(todos, list):
            return create_response(
                success=False,
                message='Missing or invalid field',
                error='todos must be a list of todo objects',
                status_code=400
            )
        mode = data.get('mode', 'per_todo')
        if mode not in ('per_todo', 'per_month'):
            mode = 'per_todo'
        replace_todo_ids = data.get('replace_todo_ids')
        replaced = 0
        if isinstance(replace_todo_ids, list) and replace_todo_ids:
            from user_memory import mark_memories_deleted_by_todo_ids
            replaced = mark_memories_deleted_by_todo_ids(user_id.strip(), replace_todo_ids)
        from user_memory import add_todos_as_memories
        added, embedded_texts = add_todos_as_memories(user_id.strip(), todos, mode=mode)
        return create_response(
            data={
                'added': added,
                'replaced': replaced,
                'embedded_text': embedded_texts,
            },
            message=f'Embedded {added} todo memory/memories successfully' + (f' (replaced {replaced} old)' if replaced else ''),
            success=True,
            status_code=200
        )
    except ValueError as e:
        return create_response(success=False, message='Invalid input', error=str(e), status_code=400)
    except Exception as e:
        logger.error("Error in embed_user_todos: %s", str(e))
        return create_response(
            success=False,
            message='Failed to embed todos',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def delete_user_todo_memories(req: https_fn.Request) -> https_fn.Response:
    """Mark RAG memories for the given todos as deleted (soft delete). Call when user updates or deletes todos.
    Only affects memories stored with todo_id (per_todo mode). Retrieval will skip these.

    Request body:
        - user_id: User identifier (required)
        - todo_ids: List of todo ids to mark deleted (required), e.g. ["id1", "id2"] or [1, 2]

    Returns:
        - deleted: Number of memories marked deleted
    """
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
        user_id = data.get('user_id')
        todo_ids = data.get('todo_ids')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        if not isinstance(todo_ids, list):
            return create_response(
                success=False,
                message='Missing or invalid field',
                error='todo_ids must be a list (e.g. ids of updated or deleted todos)',
                status_code=400
            )
        from user_memory import mark_memories_deleted_by_todo_ids
        count = mark_memories_deleted_by_todo_ids(user_id.strip(), todo_ids)
        return create_response(
            data={'deleted': count},
            message=f'Marked {count} todo memory/memories as deleted',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in delete_user_todo_memories: %s", str(e))
        return create_response(
            success=False,
            message='Failed to mark memories deleted',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def generate_lifestyle_response(req: https_fn.Request) -> https_fn.Response:
    """Generate an AI lifestyle coach response using the user's stored memories (RAG).

    Request body:
        - user_id: User identifier (required)
        - question: User's question (required)
        - system_prompt: Optional override for system prompt
        - model: Optional model name (default: gpt-4o-mini)

    Returns:
        - response: Generated text
    """
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
        user_id = data.get('user_id')
        question = data.get('question')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        if not question or not isinstance(question, str) or not question.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='question is required',
                status_code=400
            )
        from user_memory import generate_response
        system_prompt = data.get('system_prompt')
        model = data.get('model', 'gpt-4o-mini')
        response_text = generate_response(
            user_id=user_id.strip(),
            question=question.strip(),
            system_prompt=system_prompt if isinstance(system_prompt, str) else None,
            model=model if isinstance(model, str) else 'gpt-4o-mini'
        )
        return create_response(
            data={'response': response_text},
            message='Response generated successfully',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in generate_lifestyle_response: %s", str(e))
        return create_response(
            success=False,
            message='Response generation failed',
            error=str(e),
            status_code=500
        )


def _sanitize_intent_key(value: Optional[str], default: str = "unknown") -> str:
    """Sanitize dynamic keys for nested Firestore counters."""
    if not value or not isinstance(value, str):
        return default
    safe = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in value.strip().lower())
    return safe or default


def _build_time_bucket(event_time: datetime) -> str:
    """Create a coarse time bucket for preference patterns."""
    hour = event_time.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _safe_signal_weight(raw_weight: Any) -> float:
    """Normalize signal weights into a safe bounded range."""
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        weight = 1.0
    if weight < 0.1:
        return 0.1
    if weight > 5.0:
        return 5.0
    return weight


def _top_items_from_counts(counts: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top-N sorted key/count rows from a counter dict."""
    if not isinstance(counts, dict):
        return []
    rows = []
    for key, value in counts.items():
        try:
            rows.append({'key': key, 'count': float(value)})
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda row: row['count'], reverse=True)
    return rows[:max(1, top_n)]


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def track_user_intent_signal(req: https_fn.Request) -> https_fn.Response:
    """Track user intent signals and aggregate them for personalization.

    Request body:
        - user_id: User identifier (required)
        - event_name: Intent event name (required)
        - source: Optional screen/source (calendar, feed, ai, etc.)
        - signal_weight: Optional numeric weight (default 1.0, range 0.1..5.0)
        - metadata: Optional free-form dict
        - event_time: Optional ISO datetime string (default now UTC)
    """
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

        user_id = data.get('user_id')
        event_name = data.get('event_name')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        if not event_name or not isinstance(event_name, str) or not event_name.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='event_name is required',
                status_code=400
            )

        uid = user_id.strip()
        event_key = _sanitize_intent_key(event_name, default='unknown_event')
        source_key = _sanitize_intent_key(data.get('source', 'unknown_source'), default='unknown_source')
        signal_weight = _safe_signal_weight(data.get('signal_weight', 1.0))
        metadata = data.get('metadata')
        if metadata is not None and not isinstance(metadata, dict):
            metadata = None

        raw_event_time = data.get('event_time')
        event_time = None
        if isinstance(raw_event_time, str) and raw_event_time.strip():
            try:
                event_time = datetime.fromisoformat(raw_event_time.replace('Z', '+00:00'))
            except Exception:
                event_time = None
        if event_time is None:
            event_time = datetime.utcnow()
        time_bucket = _build_time_bucket(event_time)

        db = _get_firestore_client()
        user_ref = db.collection('users').document(uid)
        signal_ref = user_ref.collection('intentSignals').document()
        profile_ref = user_ref.collection('ai_intent_profile').document('current')

        signal_ref.set({
            'user_id': uid,
            'event_name': event_name.strip(),
            'event_key': event_key,
            'source': source_key,
            'signal_weight': signal_weight,
            'time_bucket': time_bucket,
            'event_time': event_time.isoformat(),
            'created_at': firestore.SERVER_TIMESTAMP,
            'metadata': metadata or {},
        })

        profile_ref.set({
            'user_id': uid,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'total_signals': firestore.Increment(1),
            'weighted_total': firestore.Increment(signal_weight),
            f'event_counts.{event_key}': firestore.Increment(signal_weight),
            f'source_counts.{source_key}': firestore.Increment(signal_weight),
            f'time_bucket_counts.{time_bucket}': firestore.Increment(signal_weight),
            'last_event_name': event_name.strip(),
            'last_source': source_key,
            'last_event_time': event_time.isoformat(),
        }, merge=True)

        return create_response(
            data={
                'event_key': event_key,
                'source': source_key,
                'time_bucket': time_bucket,
                'signal_weight': signal_weight,
            },
            message='Intent signal tracked successfully',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in track_user_intent_signal: %s", str(e))
        return create_response(
            success=False,
            message='Intent signal tracking failed',
            error=str(e),
            status_code=500
        )


@https_fn.on_request(memory=1024, max_instances=3, timeout_sec=540, cpu=1)
def get_user_intent_profile(req: https_fn.Request) -> https_fn.Response:
    """Fetch aggregated user intent profile for personalization and ranking."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()
    if req.method not in ('POST', 'GET'):
        return create_response(
            success=False,
            message='Method not allowed',
            error='Only GET and POST methods are allowed',
            status_code=405
        )
    try:
        data = req.get_json(silent=True) or {}
        user_id = data.get('user_id') or req.args.get('user_id')
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return create_response(
                success=False,
                message='Missing required field',
                error='user_id is required',
                status_code=400
            )
        top_n_raw = data.get('top_n') if isinstance(data, dict) else None
        try:
            top_n = int(top_n_raw) if top_n_raw is not None else 5
        except (TypeError, ValueError):
            top_n = 5
        top_n = min(20, max(1, top_n))

        db = _get_firestore_client()
        profile_doc = db.collection('users').document(user_id.strip()).collection('ai_intent_profile').document('current').get()
        if not profile_doc.exists:
            return create_response(
                success=False,
                message='Intent profile not found',
                error='No intent profile yet. Track at least one signal first.',
                status_code=404
            )
        profile = profile_doc.to_dict() or {}

        event_counts = profile.get('event_counts', {})
        source_counts = profile.get('source_counts', {})
        time_bucket_counts = profile.get('time_bucket_counts', {})

        return create_response(
            data={
                'profile': profile,
                'top_events': _top_items_from_counts(event_counts, top_n=top_n),
                'top_sources': _top_items_from_counts(source_counts, top_n=top_n),
                'top_time_buckets': _top_items_from_counts(time_bucket_counts, top_n=top_n),
            },
            message='Intent profile fetched successfully',
            success=True,
            status_code=200
        )
    except Exception as e:
        logger.error("Error in get_user_intent_profile: %s", str(e))
        return create_response(
            success=False,
            message='Failed to fetch intent profile',
            error=str(e),
            status_code=500
        )


# @https_fn.on_request(memory=2048, max_instances=3, cpu=2, timeout_sec=300)
# def yolo_image_generation(req: https_fn.Request) -> https_fn.Response:
#     """Run YOLO object detection on an image and save the result to Firebase Storage.
    
#     Request body:
#         - url: URL of the image to process
#         - filename: Name for the output file (without extension)
#         - confidence (optional): Detection confidence threshold (default: 0.25)
        
#     Returns:
#         - predicted_url: Public URL of the annotated image in Firebase Storage
#         - detections: List of detected objects with class names, confidence scores, and bounding boxes
#     """
#     import tempfile
#     import shutil
#     import glob as glob_module
#     from PIL import Image
    
#     start_time = time.time()
#     temp_dir = None
    
#     if req.method == 'OPTIONS':
#         return handle_preflight_request()
    
#     if req.method != 'POST':
#         return create_response(
#             success=False,
#             message='Method not allowed',
#             error='Only POST method is allowed',
#             status_code=405
#         )
    
#     try:
#         # Parse request data
#         try:
#             data = req.get_json()
#             if data is None:
#                 return create_response(
#                     success=False,
#                     message='Invalid request',
#                     error='Request body must be valid JSON',
#                     status_code=400
#                 )
#         except Exception as e:
#             return create_response(
#                 success=False,
#                 message='Invalid request',
#                 error=f'Failed to parse JSON: {str(e)}',
#                 status_code=400
#             )
        
#         # Validate required fields
#         image_url = data.get('url')
#         filename = data.get('filename')
        
#         if not image_url:
#             return create_response(
#                 success=False,
#                 message='Missing required field',
#                 error='url is required',
#                 status_code=400
#             )
        
#         if not filename:
#             return create_response(
#                 success=False,
#                 message='Missing required field',
#                 error='filename is required',
#                 status_code=400
#             )
        
#         # Optional parameters
#         confidence = data.get('confidence', 0.25)
        
#         logger.info(f"Processing YOLO detection for image: {image_url}, filename: {filename}")
        
#         # Convert Google Drive sharing URLs to direct download URLs
#         import re
#         if 'drive.google.com' in image_url:
#             # Extract file ID from various Google Drive URL formats
#             # Format 1: https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing
#             # Format 2: https://drive.google.com/open?id={FILE_ID}
#             # Format 3: https://drive.google.com/uc?id={FILE_ID}
#             file_id = None
            
#             # Try to extract file ID from /file/d/ format
#             match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', image_url)
#             if match:
#                 file_id = match.group(1)
#             else:
#                 # Try to extract from id= parameter
#                 match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', image_url)
#                 if match:
#                     file_id = match.group(1)
            
#             if file_id:
#                 # Convert to direct download URL
#                 image_url = f"https://drive.google.com/uc?export=download&id={file_id}"
#                 logger.info(f"Converted Google Drive URL to direct download: {image_url}")
#             else:
#                 logger.warning(f"Could not extract file ID from Google Drive URL: {image_url}")
        
#         # Download image from URL
#         try:
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#                 'Accept': 'image/*,*/*'
#             }
            
#             # Use a session to handle cookies (needed for Google Drive)
#             session = requests.Session()
#             response = session.get(image_url, timeout=30, headers=headers, allow_redirects=True)
#             response.raise_for_status()
            
#             # Handle Google Drive virus scan warning for large files
#             if 'drive.google.com' in image_url or 'drive.usercontent.google.com' in response.url:
#                 # Check if we got a confirmation page instead of the file
#                 if b'download_warning' in response.content or b'confirm=' in response.content:
#                     # Extract confirmation token and retry
#                     confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
#                     if confirm_match:
#                         confirm_token = confirm_match.group(1)
#                         # Add confirm parameter and retry
#                         if '?' in image_url:
#                             confirmed_url = f"{image_url}&confirm={confirm_token}"
#                         else:
#                             confirmed_url = f"{image_url}?confirm={confirm_token}"
#                         logger.info(f"Retrying Google Drive download with confirmation token")
#                         response = session.get(confirmed_url, timeout=30, headers=headers, allow_redirects=True)
#                         response.raise_for_status()
            
#             image_bytes = response.content
            
#             # Log content type for debugging
#             content_type = response.headers.get('Content-Type', 'unknown')
#             logger.info(f"Downloaded content: {len(image_bytes)} bytes, Content-Type: {content_type}")
            
#             # Basic validation - check if content looks like an image
#             if len(image_bytes) < 100:
#                 logger.error(f"Downloaded content too small: {len(image_bytes)} bytes")
#                 return create_response(
#                     success=False,
#                     message='Invalid image',
#                     error='Downloaded content is too small to be a valid image',
#                     status_code=400
#                 )
            
#             # Check for common HTML/error responses
#             content_start = image_bytes[:50].lower()
#             if b'<!doctype' in content_start or b'<html' in content_start or b'{"error' in content_start:
#                 logger.error(f"Downloaded content appears to be HTML or error response, not an image")
#                 return create_response(
#                     success=False,
#                     message='Invalid image URL',
#                     error='URL did not return a valid image (received HTML or error response)',
#                     status_code=400
#                 )
                
#         except requests.RequestException as e:
#             logger.error(f"Failed to download image: {str(e)}")
#             return create_response(
#                 success=False,
#                 message='Failed to download image',
#                 error=f'Could not fetch image from URL: {str(e)}',
#                 status_code=400
#             )
        
#         # Validate and open the image with PIL
#         try:
#             image = Image.open(io.BytesIO(image_bytes))
#             image.verify()  # Verify it's a valid image
#             # Re-open after verify (verify() leaves file in unusable state)
#             image = Image.open(io.BytesIO(image_bytes))
#             logger.info(f"Image validated: format={image.format}, size={image.size}, mode={image.mode}")
#         except Exception as e:
#             logger.error(f"Failed to open/validate image: {str(e)}")
#             logger.error(f"First 100 bytes of content: {image_bytes[:100]}")
#             return create_response(
#                 success=False,
#                 message='Invalid image format',
#                 error=f'Could not process image: {str(e)}. The URL may not point to a valid image file.',
#                 status_code=400
#             )
        
#         # Load YOLO model
#         model = get_yolo_model()
#         if model is None:
#             return create_response(
#                 success=False,
#                 message='Model not available',
#                 error='Failed to load YOLO model',
#                 status_code=500
#             )
        
#         # Generate storage path using provided filename
#         clean_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
#         storage_path = f"yolo_predictions/{clean_filename}_predicted.jpg"
        
#         # Set up temp directory for YOLO to save predicted images
#         temp_dir = tempfile.mkdtemp()
        
#         # Save downloaded image to temp file (YOLO works better with file paths)
#         # Convert to RGB to ensure compatibility (handles RGBA, palette modes, etc.)
#         temp_input_path = os.path.join(temp_dir, f"{clean_filename}.jpg")
#         if image.mode in ('RGBA', 'LA', 'P'):
#             # Convert images with transparency to RGB with white background
#             background = Image.new('RGB', image.size, (255, 255, 255))
#             if image.mode == 'P':
#                 image = image.convert('RGBA')
#             background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
#             image = background
#         elif image.mode != 'RGB':
#             image = image.convert('RGB')
#         image.save(temp_input_path, 'JPEG', quality=95)
        
#         # Run YOLO prediction with save=True (saves annotated image automatically)
#         results = model.predict(
#             source=temp_input_path, 
#             conf=confidence, 
#             save=True,
#             project=temp_dir,
#             name='predict',
#             exist_ok=True
#         )
        
#         result = results[0]
        
#         # Find the saved predicted image in the output directory
#         # YOLO saves to {project}/{name}/ with the same filename as input
#         predict_dir = os.path.join(temp_dir, 'predict')
#         saved_images = glob_module.glob(os.path.join(predict_dir, '*.jpg')) + \
#                        glob_module.glob(os.path.join(predict_dir, '*.png'))
        
#         if not saved_images:
#             logger.error(f"No predicted image found in {predict_dir}")
#             return create_response(
#                 success=False,
#                 message='Prediction failed',
#                 error='No predicted image was generated',
#                 status_code=500
#             )
        
#         saved_image_path = saved_images[0]
#         logger.info(f"YOLO saved predicted image to: {saved_image_path}")
        
#         # Upload the saved image to Firebase Storage
#         bucket = storage.bucket()
#         blob = bucket.blob(storage_path)
#         blob.upload_from_filename(saved_image_path, content_type='image/jpeg')
        
#         # Make the file publicly accessible
#         blob.make_public()
#         predicted_url = blob.public_url
        
#         logger.info(f"Uploaded annotated image to: {predicted_url}")
        
#         # Extract detection information
#         detections = []
#         boxes = result.boxes
#         if boxes is not None:
#             for box in boxes:
#                 detection = {
#                     'class_id': int(box.cls[0].item()),
#                     'class_name': result.names[int(box.cls[0].item())],
#                     'confidence': float(box.conf[0].item()),
#                     'bbox': {
#                         'x1': float(box.xyxy[0][0].item()),
#                         'y1': float(box.xyxy[0][1].item()),
#                         'x2': float(box.xyxy[0][2].item()),
#                         'y2': float(box.xyxy[0][3].item())
#                     }
#                 }
#                 detections.append(detection)
        
#         processing_time = time.time() - start_time
        
#         json_body = {
#             "Action": "Edit",
#             "Rows": [
#                 {
#                 "ID": "123",
#                 "ApiStatus": "SUCCESS",
#                 "ApiMessage": "Image processed",
#                 "ProcessedAt": datetime.now().isoformat()
#                 }
#             ]
#             }
#         response = requests.post(f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE}/Action", json=json_body)
#         #POST https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE}/Action
#         return create_response(
#             data={
#                 'predicted_url': predicted_url,
#                 'detections': detections,
#                 'detection_count': len(detections),
#                 'original_url': image_url,
#                 'filename': clean_filename
#             },
#             message='YOLO detection completed successfully',
#             metadata={
#                 'processing_time_seconds': round(processing_time, 2),
#                 'model_confidence_threshold': confidence
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Error in yolo_image_generation: {str(e)}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return create_response(
#             success=False,
#             message='YOLO detection failed',
#             error=f'Failed to process image: {str(e)}',
#             status_code=500
#         )
    
#     finally:
#         # Clean up temp directory
#         if temp_dir and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir, ignore_errors=True)