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
from firebase_functions.params import SecretParam

# Injected at deploy when bound on functions that need them (see decorators).
_EVO_FIREBASE_SA_SECRET = SecretParam("EVO_FIREBASE_SERVICE_ACCOUNT_JSON")
_OPENAI_API_KEY_SECRET = SecretParam("OPENAI_API_KEY")
_LLM_SECRETS = [_EVO_FIREBASE_SA_SECRET, _OPENAI_API_KEY_SECRET]
from firebase_admin import initialize_app, storage, firestore
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Local overrides only — never deployed as plain env vars by Firebase CLI.
_functions_dir = Path(__file__).resolve().parent
load_dotenv(_functions_dir / ".env.local")
load_dotenv(_functions_dir.parent / ".env")
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
@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2, secrets=_LLM_SECRETS)  # 9 minutes timeout
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


@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540, cpu=2, secrets=_LLM_SECRETS)
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
    "EVO practice loop (align with this):\n"
    "- Structured plan days may include Practice this step — AI drill material + an "
    "optional quiz per step (not tips, not streaks). Coach reviews judge what the "
    "user actually did; you may nudge them to run one step's material before the "
    "next rep when data is thin.\n"
    "- Missed days are information, not failure. Favor recoverable next actions.\n\n"
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
    Decode Firebase Auth ID token → (uid, tier).

    schedulingclassroom CF runs in a *different* GCP/Firebase project than the
    EVO mobile app (evoforluanching). App users' tokens must be verified via
    verify_evo_id_token + coachSubscription read from EVO Firestore
    (_coach_tier_for_uid). Default verify_id_token is only a fallback.
    """
    auth_header = req.headers.get("Authorization") or req.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return (None, "free")
    id_token = auth_header.split(" ", 1)[1].strip()
    if not id_token:
        return (None, "free")

    uid = None
    try:
        from evo_firebase import verify_evo_id_token
        uid = verify_evo_id_token(id_token)
    except Exception as e:
        logger.warning("EVO token verify import failed: %s", e)

    if not uid:
        try:
            from firebase_admin import auth as fb_auth
            decoded = fb_auth.verify_id_token(id_token)
            uid = decoded.get("uid")
        except Exception as e:
            logger.info("coach_tier: invalid id_token (%s)", type(e).__name__)
            return (None, "free")

    if not uid:
        return (None, "free")

    return (uid, _coach_tier_for_uid(uid))


@https_fn.on_request(
    memory=1024, max_instances=20, timeout_sec=120, cpu=1,
    secrets=_LLM_SECRETS,
)
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
# Practice Card — interactive scenario generator
# =========================
# Generates a small "scenario card" the user opens on dateScreenFull to
# rehearse the upcoming rep mentally before doing it. Format is fixed
# (one situation + 5 choices + a non-judgmental note); variety comes from
# scenario *content*, gated by an anti-repeat list passed from the client.
#
# Tiering — verified server-side via _verify_coach_tier:
#   free    → 1 card / UTC day,   gpt-5.4-mini, no coachFollowUp
#   plus    → unlimited,          gpt-5.4-mini, no coachFollowUp
#   premium → unlimited,          gpt-5.4,      with coachFollowUp paragraph
#
# Cache: users/{uid}/practice/{taskId} with 24h TTL so repeat opens are
# free. Cache write happens only for authenticated users.

_PRACTICE_CACHE_TTL_SEC = 24 * 60 * 60
_PRACTICE_HISTORY_LIMIT = 10
# Bump when the card prompt/logic changes enough to invalidate cached cards
# (alongside the 24h TTL). v2: per-category scenario guidance, thin-step
# grounding, and practice-language output guard. v3: immersive coaching
# language for target-language steps.
_PRACTICE_FORMAT_VERSION = 3

# Semantic role of each scenario choice. Logged with the user's pick so the
# coach loop can mine *what kind* of move a user makes under pressure — not
# just which letter they tapped. The four cover the realistic responses to
# a hard moment in a rep:
#   recovery → reset/regroup, then continue (the plasticity-friendly move)
#   persist  → push through as-is
#   adjust   → change the approach or scope
#   avoid    → disengage / escape
_PRACTICE_INTENTS = {"recovery", "persist", "adjust", "avoid"}
_PRACTICE_CHOICE_COUNT = 5

_PRACTICE_INTENT_ANALYSIS_SYSTEM = (
    "You decide how to frame ONE practice scenario card before it is written.\n"
    "Goal: help the user DEVELOP A NEW SKILL through mental rehearsal — a "
    "specific micro-skill drawn from their planner arc, not generic motivation.\n"
    "The card will be written primarily in the user's SELECTED RESPONSE LANGUAGE "
    "(provided in the user message). Your job is to find the PLANNER language "
    "to blend in — not to pick the response language.\n"
    "Return ONLY valid JSON (no markdown). Fields:\n"
    "- plannerLanguage: full language name for the TARGET language being "
    "practiced/learned — infer primarily from main_goal (planName) and PLANNER "
    "CONTENT INTENT (user_intent, arc_summary, day_steps). e.g. planName "
    "'Learn Chinese 30 days' → Chinese. Do NOT return the response/UI "
    "language unless the user is actually learning that language.\n"
    "- practiceFocus: one short phrase naming the skill or micro-skill to "
    "rehearse (e.g. 'recover after a missed rep without quitting', "
    "'hold form under fatigue', 'use the new vocab in a real sentence').\n"
    '{"plannerLanguage":"...","practiceFocus":"..."}'
)

_PRACTICE_SYSTEM_PROMPT_BASE = (
    "You generate ONE 'scenario card' for a user about to do a task in the "
    "EVO app — a behavior-change tool grounded in neuroplasticity (consistent "
    "recoverable repetition, not streaks).\n\n"
    "Goal of the card: help the user DEVELOP A NEW SKILL by mentally rehearsing "
    "a realistic moment before they do the rep. Embed one concrete learning "
    "hook from the task (a term, form cue, phrase, or constraint) in the "
    "scenario. Ground the card in the planner arc and PRACTICE INTENT when "
    "provided. There is NO correct answer; picking is the rep.\n\n"
    "Voice: warm, direct, concrete. Second-person. No moralizing. No toxic "
    "positivity. No emoji. No scores, no badges, no streaks.\n\n"
    "Hard rules:\n"
    "1. Return ONLY valid JSON. No markdown fences. No preamble.\n"
    "2. `situation` is 2–3 sentences: a concrete moment grounded in the actual "
    "task, including one specific detail the user will face during the rep.\n"
    f"3. `choices` MUST be exactly {_PRACTICE_CHOICE_COUNT} items, each with "
    "keys 'key' (a/b/c/d/e), 'label' (under 90 chars, plausible real action "
    "using vocabulary/phrasing from the task when relevant — not a joke "
    "option), and 'intent'.\n"
    "4. Each choice's `intent` MUST be EXACTLY one of: 'recovery' "
    "(reset/regroup then continue), 'persist' (push through as-is), 'adjust' "
    "(change the approach or scope), 'avoid' (disengage/escape). The five "
    "choices SHOULD span different intents where possible so the user's pick "
    "is informative — never make all five the same intent.\n"
    "5. `afterChoiceNote` is 2–3 sentences that TEACH: (1) every choice is a "
    "valid rehearsal — noticing matters more than picking the 'right' one; "
    "(2) what the intent patterns reveal about building this skill; (3) one "
    "actionable takeaway for the rep ahead.\n"
    "6. `scenarioId` is a kebab-case theme (e.g. 'distraction-at-15min', "
    "'motivation-drop', 'unclear-next-step') — used for anti-repeat.\n"
    "7. If the user state indicates a missed day, generate a GENTLER "
    "return-rep scenario; do not shame.\n"
    "8. Avoid scenario themes listed in `recent_scenarios`.\n"
    "9. Follow the LANGUAGE instruction in the user message exactly — the card "
    "language comes from taskTitle/taskDetail. ONLY when the plan is about "
    "learning a foreign language do you weave that target language into speech "
    "being rehearsed; for any other subject (math, science, fitness, art, "
    "finance, …) write the whole card in the one coaching language. Never "
    "default to English.\n"
)

def _practice_choice_key(index: int) -> str:
    return chr(ord("a") + index)


def _practice_choice_json_template() -> str:
    keys = [_practice_choice_key(i) for i in range(_PRACTICE_CHOICE_COUNT)]
    lines = [
        f'    {{"key":"{k}","label":"...","intent":"recovery|persist|adjust|avoid"}},'
        for k in keys
    ]
    return "\n".join(lines)


_PRACTICE_SYSTEM_PROMPT_FREE = _PRACTICE_SYSTEM_PROMPT_BASE + (
    "\nOutput JSON shape:\n"
    "{\n"
    '  "scenarioId": "<kebab-case>",\n'
    '  "situation": "<1–2 sentences>",\n'
    '  "choices": [\n'
    + _practice_choice_json_template() + "\n"
    "  ],\n"
    '  "afterChoiceNote": "<2–3 teaching sentences>"\n'
    "}\n"
)

_PRACTICE_SYSTEM_PROMPT_PREMIUM = _PRACTICE_SYSTEM_PROMPT_BASE + (
    "\nOutput JSON shape (premium — include `coachFollowUp`):\n"
    "{\n"
    '  "scenarioId": "<kebab-case>",\n'
    '  "situation": "<2–3 sentences>",\n'
    '  "choices": [\n'
    + _practice_choice_json_template() + "\n"
    "  ],\n"
    '  "afterChoiceNote": "<2–3 teaching sentences>",\n'
    '  "coachFollowUp": "<3–4 sentences a coach would say AFTER the user '
    "picks any option — name the micro-skill, link it to the plan arc, what "
    "to try in the next 60 seconds, and one thing to notice after picking. "
    'Same voice rules.>"\n'
    "}\n"
)


def _local_period_key(tz_offset_minutes: Any, fmt: str) -> str:
    """Build a usage-counter doc id in the USER'S LOCAL time, so quota windows
    reset on the user's calendar boundary (local midnight / 1st of the month)
    rather than the UTC boundary.

    `tz_offset_minutes` is the number of minutes to ADD to UTC to reach local
    time — i.e. the client sends `-(new Date().getTimezoneOffset())` (e.g.
    +420 for UTC+7). We clamp to ±14h (the real-world TZ range) so a spoofed
    value can only shift the boundary by hours, never reset a quota mid-period.
    A missing/invalid offset falls back to UTC (offset 0)."""
    try:
        offset = int(tz_offset_minutes)
    except (TypeError, ValueError):
        offset = 0
    offset = max(-840, min(840, offset))
    return (datetime.now(timezone.utc) + timedelta(minutes=offset)).strftime(fmt)


def _practice_daily_count_check_and_inc(uid: str, tz_offset_minutes: Any = 0) -> Tuple[bool, int]:
    """Atomically increment the free-tier daily counter (in the user's local
    day). Returns (allowed, new_count). If the counter is already >= 1 before
    the increment, returns (False, current_count) without writing."""
    if not uid:
        return (True, 0)  # anonymous goes through; rate-limit catches abuse
    try:
        db = firestore.client()
        ref = (db.collection("users").document(uid)
                 .collection("practice_usage")
                 .document(_local_period_key(tz_offset_minutes, "%Y-%m-%d")))
        transaction = db.transaction()

        @firestore.transactional
        def _txn(tx):
            snap = ref.get(transaction=tx)
            current = (snap.to_dict() or {}).get("count", 0) if snap.exists else 0
            if current >= 1:
                return (False, current)
            tx.set(ref, {"count": current + 1, "updatedAt": firestore.SERVER_TIMESTAMP},
                   merge=True)
            return (True, current + 1)

        return _txn(transaction)
    except Exception as e:
        logger.warning("practice daily counter failed for %s: %s — allowing", uid, e)
        return (True, 0)


def _practice_cache_get(uid: str, task_id: str) -> Optional[Dict[str, Any]]:
    if not uid or not task_id:
        return None
    try:
        db = firestore.client()
        snap = (db.collection("users").document(uid)
                  .collection("practice").document(task_id).get())
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        # Cards from an older prompt/logic version are treated as a miss so the
        # next open regenerates with the improved grounding.
        if int(data.get("formatVersion") or 0) < _PRACTICE_FORMAT_VERSION:
            return None
        generated_at = data.get("generatedAt")
        ts = None
        if hasattr(generated_at, "timestamp"):
            ts = generated_at.timestamp()
        elif isinstance(generated_at, (int, float)):
            ts = float(generated_at)
        if ts is None or (time.time() - ts) > _PRACTICE_CACHE_TTL_SEC:
            return None
        return data
    except Exception as e:
        logger.warning("practice cache read failed for %s/%s: %s", uid, task_id, e)
        return None


def _practice_cache_set(uid: str, task_id: str, card: Dict[str, Any]) -> None:
    if not uid or not task_id:
        return
    try:
        db = firestore.client()
        ref = (db.collection("users").document(uid)
                 .collection("practice").document(task_id))
        ref.set({**card,
                 "generatedAt": firestore.SERVER_TIMESTAMP,
                 "formatVersion": _PRACTICE_FORMAT_VERSION}, merge=True)
    except Exception as e:
        logger.warning("practice cache write failed for %s/%s: %s", uid, task_id, e)


def _practice_history_append(uid: str, scenario_id: str) -> None:
    """Roll the user's scenarioId history forward — last N kept for anti-repeat
    on the server side as a safety net when the client doesn't send one."""
    if not uid or not scenario_id:
        return
    try:
        db = firestore.client()
        ref = (db.collection("users").document(uid)
                 .collection("practice_meta").document("history"))
        snap = ref.get()
        prev = (snap.to_dict() or {}).get("scenarioIds", []) if snap.exists else []
        rolled = ([scenario_id] + [s for s in prev if s != scenario_id])[:_PRACTICE_HISTORY_LIMIT]
        ref.set({"scenarioIds": rolled, "updatedAt": firestore.SERVER_TIMESTAMP}, merge=True)
    except Exception as e:
        logger.warning("practice history append failed for %s: %s", uid, e)


def _practice_strip_for_tier(card: Dict[str, Any], tier: str) -> Dict[str, Any]:
    """Premium-only fields are removed before returning to non-premium callers,
    even if the model produced them or the cache holds a richer copy."""
    out = dict(card)
    if tier != "premium":
        out.pop("coachFollowUp", None)
    return out


def _derive_practice_context_from_content_ai(
    content_ai: Any, plan_doc: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Backfill practiceContext for plans saved before the field existed."""
    if not isinstance(content_ai, dict):
        return None
    raw_days = content_ai.get("days")
    if not isinstance(raw_days, list) or not raw_days:
        return None

    def _tips_str(tips: Any) -> Optional[str]:
        if not tips:
            return None
        if isinstance(tips, list):
            joined = " • ".join(str(t).strip() for t in tips if str(t).strip())
            return joined[:300] if joined else None
        s = str(tips).strip()
        return s[:300] if s else None

    days_out = []
    for i, day in enumerate(raw_days[:90]):
        if not isinstance(day, dict):
            continue
        meta = day.get("aiDayMeta") if isinstance(day.get("aiDayMeta"), dict) else {}
        day_num = day.get("dayNumber") or meta.get("dayNumber") or (i + 1)
        title = str(
            meta.get("title") or day.get("title")
            or (day.get("planDetail") or {}).get("headLine") or ""
        ).strip()[:160]
        summary = str(
            meta.get("summary") or day.get("summary") or ""
        ).strip()[:400]
        tasks = day.get("tasks") or day.get("structuredTasks") or []
        task_labels = []
        if isinstance(tasks, list):
            for t in tasks[:12]:
                if not isinstance(t, dict):
                    continue
                label = str(t.get("text") or t.get("label") or "").strip()[:160]
                if label:
                    task_labels.append(label)
        if not (title or summary or task_labels):
            continue
        days_out.append({
            "dayNumber": int(day_num) if str(day_num).isdigit() else (i + 1),
            "title": title,
            "summary": summary,
            "tips": _tips_str(meta.get("tips") or day.get("tips")),
            "taskLabels": task_labels,
        })

    summary_obj = content_ai.get("summary")
    plan_summary = ""
    if isinstance(summary_obj, dict):
        if summary_obj.get("overview"):
            plan_summary = str(summary_obj["overview"])[:600]
    elif isinstance(summary_obj, str):
        plan_summary = summary_obj[:600]

    coaching = plan_doc.get("coachingAttachment") or {}
    becoming = None
    if isinstance(coaching, dict) and coaching.get("becomingPhrase"):
        becoming = str(coaching["becomingPhrase"])[:200]

    return {
        "version": 1,
        "planName": str(content_ai.get("planName") or plan_doc.get("planName") or "")[:120],
        "category": str(content_ai.get("category") or plan_doc.get("category") or "")[:60],
        "totalDays": int(content_ai.get("totalDays") or len(days_out) or 0),
        "planSummary": plan_summary or str(plan_doc.get("planDescription") or "")[:600],
        "tags": [str(t)[:40] for t in (content_ai.get("tags") or [])[:12]],
        "difficultyLevel": str(content_ai.get("difficultyLevel") or "")[:40] or None,
        "detailPrompt": "",
        "becomingPhrase": becoming,
        "days": days_out,
    }


def _load_practice_plan_context(
    uid: Optional[str], plan_id: str, plan_day_number: Any
) -> Optional[Dict[str, Any]]:
    """Load practiceContext from Firestore by planId (user plan, then newsfeed)."""
    plan_id = (plan_id or "").strip()
    if not plan_id:
        return None
    try:
        from evo_firebase import evo_firestore
        db = evo_firestore()
        if db is None:
            logger.warning(
                "practice plan context: EVO Firestore unavailable plan_id=%s",
                plan_id,
            )
            return None
        plan_doc: Optional[Dict[str, Any]] = None
        if uid:
            snap = (db.collection("users").document(uid)
                      .collection("lifestyle-plans").document(plan_id).get())
            if snap.exists:
                plan_doc = snap.to_dict() or {}
        if plan_doc is None:
            snap = db.collection("lifestyle-plans-NewsFeed").document(plan_id).get()
            if snap.exists:
                plan_doc = snap.to_dict() or {}
        if not plan_doc:
            return None

        ctx = plan_doc.get("practiceContext")
        if not isinstance(ctx, dict) or not ctx.get("days"):
            ctx = _derive_practice_context_from_content_ai(
                plan_doc.get("contentAI"), plan_doc
            )
        if not isinstance(ctx, dict):
            return None

        try:
            day_num = int(plan_day_number) if plan_day_number is not None else None
        except (TypeError, ValueError):
            day_num = None

        day_slice = None
        days = ctx.get("days") if isinstance(ctx.get("days"), list) else []
        if day_num is not None and days:
            for d in days:
                if isinstance(d, dict) and d.get("dayNumber") == day_num:
                    day_slice = d
                    break
            if day_slice is None and 1 <= day_num <= len(days):
                candidate = days[day_num - 1]
                if isinstance(candidate, dict):
                    day_slice = candidate

        return {
            "planName": str(ctx.get("planName") or plan_doc.get("planName") or "")[:120],
            "category": str(ctx.get("category") or plan_doc.get("category") or "")[:60],
            "totalDays": int(ctx.get("totalDays") or len(days) or 0),
            "planSummary": str(ctx.get("planSummary") or "")[:600],
            "becomingPhrase": ctx.get("becomingPhrase"),
            "difficultyLevel": ctx.get("difficultyLevel"),
            "detailPrompt": str(ctx.get("detailPrompt") or "")[:400],
            "dayNumber": day_num,
            "day": day_slice,
        }
    except Exception as e:
        logger.warning(
            "practice plan context load failed plan_id=%s uid=%s: %s",
            plan_id, uid or "anon", e,
        )
        return None


def _resolve_task_content_plan_name(
    plan_name: str, plan_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Main goal label — prefer client planName, else loaded plan context."""
    name = (plan_name or "").strip()
    if name:
        return name[:200]
    if isinstance(plan_context, dict):
        name = str(plan_context.get("planName") or "").strip()
        if name:
            return name[:200]
    return ""


def _format_practice_plan_context_block(plan_ctx: Dict[str, Any]) -> str:
    """Turn loaded plan context into prompt lines for the LLM."""
    lines = [
        "PLAN ARC (supports the MAIN GOAL — do not invent unrelated goals):",
        f"- main_goal (planName): {plan_ctx.get('planName') or 'unnamed'}",
        f"- category: {plan_ctx.get('category') or 'general'}",
    ]
    total = plan_ctx.get("totalDays")
    if total:
        lines.append(f"- total_days: {total}")
    if plan_ctx.get("difficultyLevel"):
        lines.append(f"- difficulty: {plan_ctx['difficultyLevel']}")
    if plan_ctx.get("planSummary"):
        lines.append(f"- arc_summary: {plan_ctx['planSummary']}")
    if plan_ctx.get("becomingPhrase"):
        lines.append(f"- identity_line: {plan_ctx['becomingPhrase']}")
    if plan_ctx.get("detailPrompt"):
        lines.append(f"- user_intent: {plan_ctx['detailPrompt']}")

    day = plan_ctx.get("day")
    day_num = plan_ctx.get("dayNumber")
    if isinstance(day, dict):
        lines.append(f"- today_is_plan_day: {day_num or day.get('dayNumber')}")
        if day.get("title"):
            lines.append(f"- day_title: {day['title']}")
        if day.get("summary"):
            lines.append(f"- day_focus: {day['summary']}")
        if day.get("tips"):
            lines.append(f"- day_tips: {day['tips']}")
        labels = day.get("taskLabels") or []
        if isinstance(labels, list) and labels:
            lines.append(f"- day_steps: {labels[:8]}")
    elif day_num is not None:
        lines.append(f"- plan_day_number: {day_num}")

    return "\n".join(lines)


def _practice_analysis_blob(
    task_title: str,
    task_category: str,
    task_detail: str,
    plan_context_block: str,
    plan_context: Optional[Dict[str, Any]] = None,
    plan_name: str = "",
) -> str:
    """Text bundle for intent analysis — main goal first, then step to practice."""
    lines: List[str] = []
    main_goal = (plan_name or "").strip()
    if not main_goal and isinstance(plan_context, dict):
        main_goal = str(plan_context.get("planName") or "").strip()
    if main_goal:
        lines.append(f"main_goal (planName): {main_goal}")
    if isinstance(plan_context, dict):
        if plan_context.get("detailPrompt"):
            lines.append(f"user_intent: {plan_context['detailPrompt']}")
        if plan_context.get("becomingPhrase"):
            lines.append(f"identity_line: {plan_context['becomingPhrase']}")
        if plan_context.get("planSummary"):
            lines.append(f"arc_summary: {plan_context['planSummary']}")
        if plan_context.get("category"):
            lines.append(f"plan_category: {plan_context['category']}")
        day = plan_context.get("day")
        if isinstance(day, dict):
            if day.get("title"):
                lines.append(f"day_title: {day['title']}")
            if day.get("summary"):
                lines.append(f"day_focus: {day['summary']}")
            if day.get("tips"):
                lines.append(f"day_tips: {day['tips']}")
    if task_title:
        lines.append(f"step_to_practice: {task_title}")
    if task_detail and task_detail.strip() != (task_title or "").strip():
        lines.append(f"parent_task: {task_detail}")
    if task_category:
        lines.append(f"task_category: {task_category}")
    if plan_context_block:
        lines.append(plan_context_block)
    return "\n".join(p.strip() for p in lines if p and p.strip())


_UI_LANGUAGE_NAMES = {
    "english": "English",
    "thai": "Thai",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
    "vietnamese": "Vietnamese",
    "indonesian": "Indonesian",
}


def _ui_language_name(language_selected: str) -> str:
    key = (language_selected or "english").lower().strip()
    return _UI_LANGUAGE_NAMES.get(key, key.title() or "English")


def _practice_languages_same(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _practice_planner_language_fallback(
    analysis_blob: str, language_selected: str
) -> str:
    """Detect planner/practice language from task and plan text."""
    try:
        from chatgpt_wrapper import LanguageDetector
        if len(analysis_blob.strip()) >= 12:
            code = LanguageDetector.detect_language(analysis_blob)
            name = LanguageDetector.get_language_name(code)
            if name and len(name) > 1 and name.lower() != code.lower():
                return name
    except Exception as e:
        logger.warning("practice language detect fallback: %s", e)
    return _ui_language_name(language_selected)


def _language_name_to_chat_code(name: str) -> str:
    """Map display language name to chatgpt_wrapper language code."""
    key = (name or "").strip().lower()
    return {
        "english": "en",
        "thai": "th",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "vietnamese": "vi",
        "indonesian": "id",
    }.get(key, key or "en")


def _sanitize_planner_target_language(
    candidate: str,
    blob: str,
    instruction_language: str,
    reply_language: str,
) -> str:
    """Drop UI/instruction language mistaken as the language being practiced."""
    hint = _task_content_practice_language_from_hints(blob)
    if hint:
        return hint
    cand = (candidate or "").strip()
    if not cand or len(cand) > 40:
        return ""
    if _practice_languages_same(cand, instruction_language):
        return ""
    if _practice_languages_same(cand, reply_language):
        return ""
    if cand.lower() == "english" and _text_has_thai_script(blob):
        return ""
    return cand


def _task_content_coaching_language(
    instruction_language: str,
    reply_language: str,
    practice_language: str,
    category_key: str,
    task_title: str = "",
    task_detail: str = "",
) -> str:
    """One coaching language for quiz, notes, and explanations — no EN/TH mix.

    For a language-learning plan the quiz IS the practice, so it follows the
    learner's level rather than the app UI:
      - step written IN the target language (e.g. an English reading passage in
        an English plan) → quiz/notes in the TARGET language. Answering English
        questions about an English passage is the actual skill; translating the
        quiz to the UI language (Thai) defeats the planner intent.
      - step written in the learner's own language (e.g. Thai describing a
        Chinese drill) → quiz in that language so a beginner can follow it.
    For non-language plans the quiz follows the app UI language, but a step the
    user authored in a non-English language is never coached in English.
    """
    instr = (instruction_language or "").strip() or "English"
    respond = (reply_language or instr).strip()
    practice = (practice_language or "").strip()
    step_blob = f"{task_title} {task_detail}"

    if category_key == "learning_language":
        # Immersive: the step is written in the target language → coach in it.
        if practice and _practice_languages_same(instr, practice):
            return practice
        # Beginner: step is in the learner's working language → coach in it.
        if instr == "Thai" or _text_has_thai_script(step_blob):
            return "Thai"
        return respond

    # Non-language plans: Thai (or any non-English) step → never leak English
    # just because the app UI defaulted to English.
    if instr == "Thai" or _text_has_thai_script(step_blob):
        return "Thai"
    if (
        instr
        and not _practice_languages_same(instr, "English")
        and _practice_languages_same(respond, "English")
    ):
        return instr
    return respond


def _analyze_practice_intent(
    analysis_blob: str,
    language_selected: str,
    plan_context: Optional[Dict[str, Any]] = None,
    instruction_language: str = "",
    plan_name: str = "",
) -> Tuple[str, str]:
    """Pre-pass: infer planner practice language + skill focus for mixing."""
    practice_focus = ""
    reply_language = _ui_language_name(language_selected)
    instr = (instruction_language or "").strip()
    blob_lower = (analysis_blob or "").lower()
    hint_lang = _task_content_practice_language_from_hints(blob_lower)
    planner_language = hint_lang or _practice_planner_language_fallback(
        analysis_blob, language_selected
    )
    if len(analysis_blob.strip()) < 8:
        return planner_language, practice_focus

    planner_intent_lines: List[str] = []
    main_goal = _resolve_task_content_plan_name(plan_name, plan_context)
    if main_goal:
        planner_intent_lines.append(f"- main_goal (planName): {main_goal}")
    if isinstance(plan_context, dict):
        if plan_context.get("detailPrompt"):
            planner_intent_lines.append(
                f"- user_intent: {plan_context['detailPrompt']}"
            )
        if plan_context.get("becomingPhrase"):
            planner_intent_lines.append(
                f"- identity_line: {plan_context['becomingPhrase']}"
            )
        if plan_context.get("planSummary"):
            planner_intent_lines.append(
                f"- arc_summary: {plan_context['planSummary']}"
            )
        if plan_context.get("category"):
            planner_intent_lines.append(
                f"- plan_category: {plan_context['category']}"
            )
        day = plan_context.get("day")
        if isinstance(day, dict):
            if day.get("title"):
                planner_intent_lines.append(f"- day_title: {day['title']}")
            if day.get("summary"):
                planner_intent_lines.append(f"- day_focus: {day['summary']}")
            labels = day.get("taskLabels") or []
            if isinstance(labels, list) and labels:
                planner_intent_lines.append(
                    f"- day_steps: {', '.join(str(x) for x in labels[:8])}"
                )

    try:
        from chatgpt_wrapper import chat_with_gpt
        intent_user_lines = [
            f"SELECTED RESPONSE LANGUAGE (app UI — quiz/coaching only, NOT "
            f"plannerLanguage): {reply_language}",
            "main_goal (planName) is the user's overall plan goal. "
            "step_to_practice is what they do on THIS step — not the main goal.",
            "Instruction language comes from step_to_practice — not planName "
            "and not the app UI setting.",
            "Decide plannerLanguage from main_goal + PLANNER CONTENT INTENT — "
            "the target language being practiced (e.g. Chinese when planName "
            "is a Chinese plan). plannerLanguage must NOT be the response/UI "
            "language unless the user is actually learning that language.",
        ]
        if planner_intent_lines:
            intent_user_lines += ["", "PLANNER CONTENT INTENT:"] + planner_intent_lines
        intent_user_lines += [
            "",
            "FULL TASK + PLAN CONTEXT:",
            analysis_blob[:3500],
        ]
        raw = chat_with_gpt(
            system_prompt=_PRACTICE_INTENT_ANALYSIS_SYSTEM,
            user_prompt="\n".join(intent_user_lines),
            model=COACH_MODEL_FREE,
            max_completion_tokens=120,
            auto_detect_language=False,
            reply_language="English",
            response_format={"type": "json_object"},
        )
        if raw and raw.strip():
            parsed = _extract_json_object(raw)
            if isinstance(parsed, dict):
                lang = (
                    parsed.get("plannerLanguage")
                    or parsed.get("contentLanguage")
                    or parsed.get("targetLanguage")
                    or ""
                ).strip()
                if lang and len(lang) <= 40:
                    sanitized = _sanitize_planner_target_language(
                        lang, blob_lower, instr, reply_language,
                    )
                    if sanitized:
                        planner_language = sanitized
                    elif hint_lang:
                        planner_language = hint_lang
                focus = (parsed.get("practiceFocus") or "").strip()
                if focus:
                    practice_focus = focus[:200]
    except Exception as e:
        logger.info("practice intent analysis skipped, using detect: %s", e)

    return planner_language, practice_focus


@https_fn.on_request(
    memory=512, max_instances=20, timeout_sec=60, cpu=1,
    secrets=_LLM_SECRETS,
)
def generate_practice(req: https_fn.Request) -> https_fn.Response:
    """Generate one scenario practice card for a task on dateScreenFull.

    Responds in languageSelected; language-learning cards weave in the
    practice/target language (e.g. Chinese pinyin with Thai pronunciation).
    Tier verified server-side; free tier capped at 1/day."""
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
        task_id = (payload.get("taskId") or "").strip()
        task_title = (payload.get("taskTitle") or "").strip()
        task_category = (payload.get("taskCategory") or "").strip()
        task_detail = (payload.get("taskDetail") or "").strip()
        plan_id = (payload.get("planId") or "").strip()
        plan_day_number = payload.get("planDayNumber")
        user_state = payload.get("userState") or {}
        language_selected = (
            payload.get("languageSelected") or payload.get("language") or "english"
        ).lower()
        force_refresh = bool(payload.get("forceRefresh", False))
        tz_offset_minutes = payload.get("tzOffsetMinutes", 0)

        if not task_id or not task_title:
            return create_response(
                success=False,
                message='Missing required fields',
                error='`taskId` and `taskTitle` are required',
                status_code=400,
            )

        # Bound payload size — same safety net as coach_review.
        if (len(task_title) > 400 or len(task_detail) > 4000
                or len(task_category) > 120):
            return create_response(
                success=False,
                message='Payload too large',
                error='Practice card payload exceeds size limits.',
                status_code=413,
            )

        uid, tier = _verify_coach_tier(req)
        is_premium = tier == "premium"
        is_paid = tier in ("plus", "premium")

        plan_context = None
        if plan_id:
            plan_context = _load_practice_plan_context(uid, plan_id, plan_day_number)
            if plan_context is None:
                logger.warning(
                    "generate_practice: plan_id=%s sent but plan context did "
                    "NOT load (uid=%s, day=%s) — card will not be grounded in "
                    "planner intent",
                    plan_id, uid or "anon", plan_day_number,
                )

        if not _coach_rate_allow(uid or "", is_paid):
            return create_response(
                success=False,
                message='Rate limit',
                error='Too many practice cards in a short window. Try again soon.',
                status_code=429,
            )

        # 1) Cache check — only for authenticated users, only when not forced.
        if uid and not force_refresh:
            cached = _practice_cache_get(uid, task_id)
            if cached:
                return create_response(
                    data={
                        "card": _practice_strip_for_tier(cached, tier),
                        "tier": tier,
                        "source": "cache",
                    },
                    message='Practice card (cached)',
                )

        # 2) Free-tier daily cap — only enforced on a fresh generation, not
        #    on cache hits (above). A user who generated earlier today can
        #    still re-open and read it; they just can't generate a new one.
        if tier == "free":
            allowed, _count = _practice_daily_count_check_and_inc(uid or "", tz_offset_minutes)
            if not allowed:
                return create_response(
                    success=False,
                    message='Daily cap reached',
                    error='You have used today’s free practice card. Upgrade for unlimited reps.',
                    status_code=402,
                    metadata={"capReached": True, "tier": tier},
                )

        # 3) Build the model prompt.
        if is_premium:
            model = COACH_MODEL_PREMIUM
            system_prompt = _PRACTICE_SYSTEM_PROMPT_PREMIUM
            max_tokens = 1100
        else:
            model = COACH_MODEL_FREE
            system_prompt = _PRACTICE_SYSTEM_PROMPT_FREE
            max_tokens = 700

        plan_context_block = (
            _format_practice_plan_context_block(plan_context)
            if plan_context else ""
        )
        plan_name = ((plan_context or {}).get("planName") or "").strip()
        analysis_blob = _practice_analysis_blob(
            task_title,
            task_category,
            task_detail,
            plan_context_block,
            plan_context,
            plan_name,
        )
        reply_language = _ui_language_name(language_selected)

        instruction_language = _task_content_instruction_language(
            task_title, task_detail, language_selected,
        )
        category_key = _task_content_category_key(
            task_category, task_title, task_detail, plan_name, plan_context,
        )
        planner_language, practice_focus = _analyze_practice_intent(
            analysis_blob, language_selected, plan_context, instruction_language,
            plan_name,
        )
        practice_language = _task_content_practice_language(
            task_title,
            task_detail,
            plan_context,
            plan_name,
            language_selected,
            planner_language,
            instruction_language,
            category_key=category_key,
        )
        coaching_language = _task_content_coaching_language(
            instruction_language,
            reply_language,
            practice_language,
            category_key,
            task_title,
            task_detail,
        )
        language_instruction = _task_content_language_mix_instruction(
            instruction_language,
            practice_language,
            category_key,
            artifact="practice_card",
            reply_language=coaching_language,
        )
        grounding_instruction = _task_content_grounding_instruction(
            plan_name, task_title, practice_focus, task_detail,
        )
        pronunciation_guidance = _task_content_pronunciation_guidance(
            instruction_language, practice_language, reply_language,
        )

        recent_scenarios = user_state.get("recentScenarios") or []
        if not isinstance(recent_scenarios, list):
            recent_scenarios = []
        recent_scenarios = [str(s) for s in recent_scenarios[:_PRACTICE_HISTORY_LIMIT]]

        missed_yesterday = bool(user_state.get("missedYesterday", False))
        rest_day = bool(user_state.get("restDayFlag", False))

        user_prompt_lines: List[str] = []
        if plan_name:
            user_prompt_lines += [
                "MAIN GOAL (planName — overall purpose of this plan):",
                f"- planName: {plan_name}",
                "",
            ]
        user_prompt_lines += [
            "THIS STEP — what to practice now (step description):",
            f"- step: {task_title}",
        ]
        if task_detail and task_detail.strip() != (task_title or "").strip():
            user_prompt_lines.append(f"- parent_task: {task_detail}")
        is_language_plan = category_key == "learning_language"
        user_prompt_lines += [
            "",
            f"SELECTED RESPONSE LANGUAGE (languageSelected): {reply_language}",
            f"TASK TEXT LANGUAGE (taskTitle/taskDetail): {instruction_language}",
        ]
        if is_language_plan:
            # Only a foreign-language plan has a separate target language to
            # rehearse; other subjects are coached entirely in the reply language.
            user_prompt_lines.append(
                f"PRACTICE/TARGET LANGUAGE (what the user drills): "
                f"{practice_language}"
            )
        user_prompt_lines += [
            language_instruction,
            "",
            grounding_instruction,
        ]
        if _task_content_step_is_thin(task_title, task_detail):
            user_prompt_lines += [
                "",
                _task_content_thin_step_directive(
                    plan_name,
                    task_title,
                    has_plan_context=bool(plan_context),
                    practice_language=practice_language,
                    artifact="practice_card",
                ),
            ]
        # Category guidance for EVERY category (not just language) so the
        # scenario embeds a concrete, on-plan skill hook instead of generic
        # motivation. The card adapts it — it does not output a drill list.
        category_guidance = _TASK_CONTENT_CATEGORY_GUIDANCE.get(category_key)
        if category_guidance:
            user_prompt_lines += [
                "",
                "CATEGORY GUIDANCE (use the concrete-skill focus and the "
                "WRONG/RIGHT specificity bar; adapt it into the scenario — do "
                "NOT output a drill list or the literal structure):",
                category_guidance,
            ]
        if pronunciation_guidance:
            user_prompt_lines += ["", pronunciation_guidance]
        if task_category:
            user_prompt_lines += ["", f"- taskCategory: {task_category}"]
        if plan_day_number is not None:
            user_prompt_lines.append(f"- plan_day_number: {plan_day_number}")
        user_prompt_lines += [
            "",
            "USER STATE:",
            f"- missed_yesterday: {missed_yesterday}",
            f"- rest_day: {rest_day}",
            f"- recent_scenarios (avoid these themes): {recent_scenarios}",
        ]
        if plan_context_block:
            user_prompt_lines += ["", plan_context_block]
        if practice_focus:
            user_prompt_lines += [
                "",
                "PLANNER INTENT FOCUS (from planName + step + plan arc):",
                f"- {practice_focus}",
            ]
        user_prompt_lines += [
            "",
            f"Generate ONE scenario card with exactly {_PRACTICE_CHOICE_COUNT} "
            "choices per the JSON contract. Teach one concrete skill hook from "
            "the step, aligned with planName; reinforce it in afterChoiceNote.",
        ]
        user_prompt = "\n".join(user_prompt_lines)

        logger.info(
            "generate_practice: uid=%s tier=%s model=%s category=%s "
            "reply_lang=%s instr_lang=%s practice_lang=%s task_id=%s "
            "plan_id=%s has_plan_ctx=%s",
            uid or "anon", tier, model, category_key, reply_language,
            instruction_language, practice_language, task_id, plan_id or "-",
            bool(plan_context),
        )

        from chatgpt_wrapper import chat_with_gpt
        response_text = chat_with_gpt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_completion_tokens=max_tokens,
            # Mixed-language layout is specified in prompts — do not let the
            # wrapper append "Please reply in X" and force a single language.
            auto_detect_language=False,
            reply_language=None,
            response_format={"type": "json_object"},
        )

        if not response_text or not response_text.strip():
            return create_response(
                success=False,
                message='Empty model response',
                error='Could not generate a card right now. Try again in a moment.',
                status_code=502,
            )

        # 4) Parse + validate — tolerant of fences / stray prose around the JSON.
        card = _extract_json_object(response_text)
        if not isinstance(card, dict):
            # The wrapper returns a plain-English sentence (not JSON) on timeout /
            # rate-limit / breaker / refusal. Surface that as a retryable "busy"
            # error rather than a confusing "malformed card" one.
            if _looks_like_model_unavailable(response_text):
                logger.warning(
                    "generate_practice: model unavailable for uid=%s: %s",
                    uid, response_text.strip()[:160],
                )
                return create_response(
                    success=False, message='Service busy',
                    error='The AI is busy right now. Please try again in a moment.',
                    status_code=503,
                )
            logger.warning("generate_practice: JSON parse failed for uid=%s", uid)
            return create_response(
                success=False,
                message='Bad model output',
                error='The card came back malformed. Try again.',
                status_code=502,
            )
        situation = (card.get("situation") or "").strip()
        choices = card.get("choices") or []
        if (not situation or not isinstance(choices, list)
                or len(choices) != _PRACTICE_CHOICE_COUNT):
            return create_response(success=False, message='Bad model output',
                                   error=f'Card must have situation and '
                                   f'{_PRACTICE_CHOICE_COUNT} choices.',
                                   status_code=502)

        # Output-quality guard: a language card must actually weave in the
        # practice/target language (e.g. Chinese phrases to rehearse), not just
        # describe the situation in Thai. Regenerate once if the target script
        # is absent for a script-distinct language.
        def _card_text(c: Dict[str, Any], ch: List[Any]) -> str:
            return " ".join([
                str(c.get("situation") or ""),
                " ".join(str(x.get("label") or "") for x in ch if isinstance(x, dict)),
                str(c.get("afterChoiceNote") or ""),
                str(c.get("coachFollowUp") or ""),
            ])

        if (
            _task_content_needs_practice_script_check(
                instruction_language, practice_language
            )
            and not _task_content_drill_uses_practice_language(
                _card_text(card, choices), practice_language
            )
        ):
            logger.warning(
                "generate_practice: card missing %s script — regenerating once "
                "uid=%s task=%s",
                practice_language, uid or "anon", task_id,
            )
            correction = _task_content_practice_script_correction(
                practice_language, coaching_language
            )
            try:
                retry_text = chat_with_gpt(
                    system_prompt=system_prompt,
                    user_prompt=f"{user_prompt}\n\n{correction}",
                    model=model,
                    max_completion_tokens=max_tokens,
                    auto_detect_language=False,
                    reply_language=None,
                    response_format={"type": "json_object"},
                )
                retry_card = _extract_json_object(retry_text or "")
                if isinstance(retry_card, dict):
                    r_sit = (retry_card.get("situation") or "").strip()
                    r_choices = retry_card.get("choices") or []
                    if (
                        r_sit
                        and isinstance(r_choices, list)
                        and len(r_choices) == _PRACTICE_CHOICE_COUNT
                        and _task_content_drill_uses_practice_language(
                            _card_text(retry_card, r_choices), practice_language
                        )
                    ):
                        card, situation, choices = retry_card, r_sit, r_choices
                        logger.info(
                            "generate_practice: correction succeeded uid=%s task=%s",
                            uid or "anon", task_id,
                        )
                    else:
                        logger.warning(
                            "generate_practice: correction still missing %s "
                            "script uid=%s", practice_language, uid or "anon",
                        )
            except Exception as e:
                logger.warning(
                    "generate_practice: card correction failed uid=%s: %s",
                    uid or "anon", e,
                )

        scenario_id = (card.get("scenarioId") or "scenario").strip() or "scenario"
        def _norm_intent(value: Any) -> str:
            iv = str(value or "").strip().lower()
            return iv if iv in _PRACTICE_INTENTS else "other"

        normalized = {
            "scenarioId": scenario_id,
            "situation": situation,
            "choices": [
                {"key": str(c.get("key", "")).strip()[:2] or _practice_choice_key(i),
                 "label": str(c.get("label", "")).strip()[:200],
                 "intent": _norm_intent(c.get("intent"))}
                for i, c in enumerate(choices[:_PRACTICE_CHOICE_COUNT])
            ],
            "afterChoiceNote": (card.get("afterChoiceNote") or "").strip()[:700],
        }
        if is_premium and card.get("coachFollowUp"):
            normalized["coachFollowUp"] = str(card["coachFollowUp"]).strip()[:900]

        # 5) Persist cache + roll history.
        _practice_cache_set(uid or "", task_id, normalized)
        _practice_history_append(uid or "", scenario_id)

        return create_response(
            data={
                "card": _practice_strip_for_tier(normalized, tier),
                "tier": tier,
                "model_used": model,
                "source": "fresh",
            },
            message='Practice card generated',
        )

    except Exception as e:
        logger.error("generate_practice error: %s", e)
        traceback.print_exc()
        return create_response(
            success=False,
            message='Practice card failed',
            error="We couldn't generate a card right now. Try again in a moment.",
            status_code=500,
        )


# =========================
# Practice outcomes — cross-user aggregation (the literal Q2)
# =========================
# "Which rehearsal patterns predict follow-through?" — i.e. do users who pick
# a 'recovery' move complete the task more often than those who 'avoid'?
# This is a CROSS-USER question, so it cannot run on a client (a user can't
# read other users' data). It runs here with the Admin SDK over a collection
# group scan of every users/{uid}/practice_log, and writes a single
# aggregate doc to practice_aggregates/global.
#
# Privacy by construction:
#   - Output is aggregate-only. No uid, taskId, or text ever leaves the
#     function — only per-intent counts and rates.
#   - k-anonymity floor: an intent's stat is published only when it draws on
#     >= MIN_COHORT_USERS distinct users AND >= MIN_COHORT_DECIDED decided
#     tasks. Otherwise it's suppressed and named in `suppressedIntents`.
#
# Trigger: Cloud Scheduler → HTTP (same pattern as the other jobs here).
# Set PRACTICE_AGG_SECRET and have the scheduler send it as X-Agg-Secret so
# the endpoint can't be triggered by arbitrary callers. If unset (dev), the
# endpoint runs open.
#
# No Firestore index required: the collection-group query uses only a bounded
# .limit() with no filter/order, which is an unordered scan. At larger scale
# this should move to incremental aggregation; the MAX_SCAN cap is the
# backstop until then (logged when hit).

_PRACTICE_AGG_GRACE_SEC = 2 * 24 * 60 * 60
_PRACTICE_AGG_MAX_SCAN = 50000
_PRACTICE_AGG_MIN_COHORT_USERS = 10
_PRACTICE_AGG_MIN_COHORT_DECIDED = 30


@https_fn.on_request(memory=1024, max_instances=2, timeout_sec=540, cpu=1)
def aggregate_practice_outcomes(req: https_fn.Request) -> https_fn.Response:
    """Aggregate pick→completion outcomes across all users, by choice-intent
    and tier, with a privacy floor. Writes practice_aggregates/global."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    expected = os.getenv("PRACTICE_AGG_SECRET")
    if expected:
        provided = req.headers.get("X-Agg-Secret") or req.args.get("secret")
        if provided != expected:
            return create_response(
                success=False, message='Forbidden',
                error='Invalid aggregation secret', status_code=403,
            )

    intents = ("recovery", "persist", "adjust", "avoid")

    try:
        db = firestore.client()
        now = time.time()

        # Collapse to one decision per (uid, taskId): a task counts as
        # completed if ANY of its pick rows was stamped done; the latest
        # pick's intent is the predictive move.
        decisions = {}
        scanned = 0
        truncated = False
        for snap in db.collection_group("practice_log").limit(_PRACTICE_AGG_MAX_SCAN).stream():
            scanned += 1
            data = snap.to_dict() or {}
            try:
                uid = snap.reference.parent.parent.id
            except Exception:
                continue
            task_id = data.get("taskId")
            intent = data.get("pickedIntent")
            if not uid or not task_id or intent not in intents:
                continue
            ts = data.get("pickedAt")
            ts_sec = ts.timestamp() if hasattr(ts, "timestamp") else 0
            completed = data.get("completed") is True
            tier = data.get("tier") or "free"
            key = (uid, str(task_id))
            cur = decisions.get(key)
            if cur is None:
                decisions[key] = {
                    "ts": ts_sec, "intent": intent,
                    "completed": completed, "tier": tier, "uid": uid,
                }
            else:
                cur["completed"] = cur["completed"] or completed
                if ts_sec >= cur["ts"]:
                    cur["ts"] = ts_sec
                    cur["intent"] = intent
                    cur["tier"] = tier
        if scanned >= _PRACTICE_AGG_MAX_SCAN:
            truncated = True
            logger.warning(
                "aggregate_practice_outcomes: hit MAX_SCAN=%d — results truncated",
                _PRACTICE_AGG_MAX_SCAN,
            )

        def new_bucket():
            return {"completed": 0, "total": 0, "users": set()}

        by_intent = {i: new_bucket() for i in intents}
        by_tier_intent = {}

        for d in decisions.values():
            if d["completed"]:
                outcome_completed = True
            elif d["ts"] and (now - d["ts"]) > _PRACTICE_AGG_GRACE_SEC:
                outcome_completed = False
            else:
                continue  # pending — not yet decided
            intent = d["intent"]
            tier = d["tier"] if d["tier"] in ("free", "plus", "premium") else "free"

            b = by_intent[intent]
            b["total"] += 1
            b["users"].add(d["uid"])
            if outcome_completed:
                b["completed"] += 1

            ti = by_tier_intent.setdefault(
                tier, {i: new_bucket() for i in intents}
            )[intent]
            ti["total"] += 1
            ti["users"].add(d["uid"])
            if outcome_completed:
                ti["completed"] += 1

        def publish(bucket):
            users = len(bucket["users"])
            total = bucket["total"]
            if users < _PRACTICE_AGG_MIN_COHORT_USERS or total < _PRACTICE_AGG_MIN_COHORT_DECIDED:
                return None
            return {
                "completed": bucket["completed"],
                "total": total,
                "rate": round(bucket["completed"] / total, 4),
                "users": users,
            }

        out_by_intent = {}
        suppressed = []
        for i in intents:
            pub = publish(by_intent[i])
            if pub:
                out_by_intent[i] = pub
            else:
                suppressed.append(i)

        out_by_tier = {}
        for tier, m in by_tier_intent.items():
            tier_out = {}
            for i in intents:
                pub = publish(m[i])
                if pub:
                    tier_out[i] = pub
            if tier_out:
                out_by_tier[tier] = tier_out

        total_decided = sum(b["total"] for b in by_intent.values())
        cohort_users = len({d["uid"] for d in decisions.values()})

        result = {
            "byIntent": out_by_intent,
            "byTierIntent": out_by_tier,
            "totalDecided": total_decided,
            "cohortUsers": cohort_users,
            "scanned": scanned,
            "truncated": truncated,
            "suppressedIntents": suppressed,
            "minCohortUsers": _PRACTICE_AGG_MIN_COHORT_USERS,
            "minCohortDecided": _PRACTICE_AGG_MIN_COHORT_DECIDED,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        db.collection("practice_aggregates").document("global").set(result)

        logger.info(
            "aggregate_practice_outcomes: scanned=%d decisions=%d decided=%d cohort=%d published=%s suppressed=%s",
            scanned, len(decisions), total_decided, cohort_users,
            list(out_by_intent.keys()), suppressed,
        )

        response_payload = dict(result)
        response_payload.pop("updatedAt", None)  # don't echo the sentinel
        return create_response(
            data=response_payload, message="Practice outcomes aggregated"
        )

    except Exception as e:
        logger.error("aggregate_practice_outcomes error: %s", e)
        traceback.print_exc()
        return create_response(
            success=False, message='Aggregation failed',
            error="Could not aggregate practice outcomes.", status_code=500,
        )


# =========================
# Task learning content + quiz — for AI-generated plans
# =========================
# On dateScreenFull, tasks that belong to an AI-generated plan get a
# "Learn + Quiz" button. It calls this endpoint, which returns:
#   - `content`: category-shaped practice material (language learning includes
#                reading guides + learn-how); facts embedded for the quiz
#   - `quiz`:    up to 5 multiple-choice questions (built in batches of 2 —
#                first pass with content, then lightweight top-up passes)
#
# Tiering (a SEPARATE monthly quota from coach reviews — verified server-side):
#   free    → 10 / month
#   plus    → 30 / month
#   premium → unlimited
#
# Cache: users/{uid}/taskContent/{taskId}. A cache hit is FREE — it never
# spends a monthly credit. Only a fresh generation (first open, or an
# explicit forceRefresh) spends one. Content about a task doesn't go stale,
# so the cache has no TTL; regeneration is user-driven.

_TASK_CONTENT_MONTHLY_CAP = {"free": 10, "plus": 30}  # premium → unlimited
# Quiz is built in batches — 5 total is too heavy for one LLM call alongside
# practice material. First pass: content + _FIRST_BATCH questions; top-up
# passes add _TOPUP_BATCH at a time until _QUIZ_TARGET (free on cache re-open).
_TASK_CONTENT_QUIZ_TARGET = 5
_TASK_CONTENT_QUIZ_FIRST_BATCH = 2
_TASK_CONTENT_QUIZ_TOPUP_BATCH = 2
_TASK_CONTENT_QUIZ_TOPUP_MAX_ATTEMPTS = 3

# Bump whenever the generation logic/prompt changes in a way that should
# invalidate previously cached content (e.g. fixing off-target drills). Cached
# entries stamped below this version are regenerated when quota allows — for
# free when the task was already charged this month — and otherwise served
# as-is so a user at their cap never loses access to existing material.
#   v2: translate-don't-copy drill rule + thin-step + language fixes.
#   v3: immersive coaching language (target-language quiz for target-language
#       steps, e.g. English quiz for an English-learning plan) +
#       genre generalization: the foreign-language drill machinery now fires
#       ONLY for learning_language plans, so math/science/economics/arts/
#       psychology/finance/etc. produce real subject material instead of being
#       skewed toward language practice.
_TASK_CONTENT_LOGIC_VERSION = 3

_TASK_CONTENT_PLAN_CATEGORIES = frozenset({
    "learning", "math", "science", "economics", "arts", "psychology",
    "exercise", "travel", "finance", "health",
    "personal_development", "other",
})

_TASK_CONTENT_CATEGORY_ALIASES = {
    "fitness": "exercise",
    "workout": "exercise",
    "sport": "exercise",
    "study": "learning",
    "education": "learning",
    "language": "learning",
    "wellness": "health",
    "growth": "personal_development",
    "self_improvement": "personal_development",
    "personaldevelopment": "personal_development",
    "money": "finance",
    "financial": "finance",
    "investing": "finance",
    "trip": "travel",
    "tourism": "travel",
    # Academic / subject genres — each maps to genre-specific guidance so the
    # output is real subject material (worked problems, mechanisms, exercises),
    # not generic study advice and not language drills.
    "maths": "math",
    "mathematics": "math",
    "calculus": "math",
    "algebra": "math",
    "geometry": "math",
    "statistics": "math",
    "stats": "math",
    "physics": "science",
    "chemistry": "science",
    "biology": "science",
    "astronomy": "science",
    "engineering": "science",
    "economy": "economics",
    "econ": "economics",
    "macroeconomics": "economics",
    "microeconomics": "economics",
    "business": "economics",
    "art": "arts",
    "drawing": "arts",
    "painting": "arts",
    "music": "arts",
    "design": "arts",
    "writing": "arts",
    "photography": "arts",
    "psych": "psychology",
    "phycology": "psychology",  # common misspelling
    "philosophy": "psychology",
}

_LANGUAGE_LEARNING_HINTS = (
    "language", "vocab", "vocabulary", "grammar", "phrase", "sentences",
    "sentence", "pronunciation", "fluent", "flashcard", "translation",
    "kanji", "hiragana", "katakana", "pinyin", "romaji", "jlpt", "toefl",
    "ielts", "hsk", "topik", "japanese", "chinese", "korean", "thai",
    "english", "spanish", "french", "german", "mandarin", "cantonese",
    "arabic", "hindi", "italian", "portuguese", "vietnamese", "indonesian",
    "russian", "toeic", "delf", "dele", "goethe",
    "中文", "日本語", "한국어", "español", "français", "deutsch", "italiano",
    "português", "русский", "العربية", "हिन्दी",
    "ภาษา", "คำศัพท์", "ไวยากรณ์", "แปล", "อ่าน", "พินอิน", "โทนเสียง",
    "แมนดาริน", "ฮั่นจื้อ",
)

# Target language being practiced (distinct from the instruction language of the step).
_PRACTICE_TARGET_LANGUAGE_HINTS: Dict[str, Tuple[str, ...]] = {
    "Chinese": (
        "chinese", "mandarin", "cantonese", "pinyin", "hanzi", "hsk",
        "汉语", "拼音", "中文", "普通话", "汉字",
        "จีน", "ภาษาจีน", "ภาษาจีนกลาง", "จีนกลาง", "แมนดาริน",
        "พินอิน", "โทนเสียง", "เขียนพินอิน", "เสียงที่", "ตัวอักษรจีน", "ฮั่นจื้อ",
    ),
    "Japanese": (
        "japanese", "jlpt", "kanji", "hiragana", "katakana", "romaji", "日本語",
        "ญี่ปุ่น", "ภาษาญี่ปุ่น", "ฮิรางานะ", "คาตาคานะ", "คันจิ",
    ),
    "Korean": (
        "korean", "hangul", "topik", "한국어",
        "เกาหลี", "ภาษาเกาหลี", "ฮันกึล",
    ),
    "Thai": ("thai", "ภาษาไทย", "ไทย"),
    "English": ("english", "ภาษาอังกฤษ", "อังกฤษ", "toefl", "ielts", "toeic"),
    "French": ("french", "français", "francais", "delf", "ภาษาฝรั่งเศส", "ฝรั่งเศส"),
    "German": ("german", "deutsch", "goethe", "ภาษาเยอรมัน", "เยอรมัน"),
    "Spanish": ("spanish", "español", "espanol", "dele", "ภาษาสเปน", "สเปน"),
    "Italian": ("italian", "italiano", "ภาษาอิตาลี", "อิตาลี"),
    "Portuguese": (
        "portuguese", "português", "portugues", "ภาษาโปรตุเกส", "โปรตุเกส",
    ),
    "Vietnamese": (
        "vietnamese", "tiếng việt", "tieng viet", "ภาษาเวียดนาม", "เวียดนาม",
    ),
    "Indonesian": (
        "indonesian", "bahasa indonesia", "ภาษาอินโดนีเซีย", "อินโดนีเซีย",
    ),
    "Russian": (
        "russian", "русский", "русски", "ภาษารัสเซีย", "รัสเซีย",
    ),
    "Arabic": (
        "arabic", "العربية", "ภาษาอาหรับ", "อาหรับ",
    ),
    "Hindi": (
        "hindi", "हिन्दी", "हिंदी", "ภาษาฮินดี", "ฮินดี",
    ),
}

# When instruction language ≠ practice language, how to spell pronunciation
# for the learner (e.g. Thai phonetics for Chinese pinyin).
_PRONUNCIATION_GUIDANCE_BY_PAIR: Dict[Tuple[str, str], str] = {
    ("Thai", "Chinese"): (
        "PRONUNCIATION (required): Thai learners need คำอ่านภาษาไทย for every "
        "pinyin item. Per word/syllable use this line format:\n"
        "  pinyin (คำอ่านไทย) โทน N — ความหมาย: ...\n"
        "Example: mā (มา) โทน 1 — ความหมาย: มา | má (ม้า) โทน 2 — ความหมาย: ม้า\n"
        "Always pair pinyin with Thai phonetic in parentheses. Label tone "
        "โทน 1–4 in Thai. Add 汉字 only when it helps. Never show bare pinyin "
        "or Chinese without the Thai pronunciation guide."
    ),
    ("Thai", "Japanese"): (
        "PRONUNCIATION (required): add คำอ่านภาษาไทย for every Japanese item "
        "(hiragana/katakana/kanji). Format: 日本語 (คำอ่านไทย) — ความหมาย: ..."
    ),
    ("Thai", "Korean"): (
        "PRONUNCIATION (required): add คำอ่านภาษาไทย for every Korean item. "
        "Format: hangul (คำอ่านไทย) — ความหมาย: ..."
    ),
    ("Thai", "English"): (
        "PRONUNCIATION (required): add คำอ่านภาษาไทย for English words the "
        "user must say aloud. Format: word (คำอ่านไทย) — ความหมาย: ..."
    ),
}


def _task_content_pronunciation_guidance(
    instruction_language: str,
    practice_language: str,
    reply_language: str = "",
) -> str:
    """Extra pronunciation rules when learner needs phonetics in their language."""
    practice = (practice_language or "").strip()
    for learner in (
        (instruction_language or "").strip(),
        (reply_language or "").strip(),
    ):
        if not learner:
            continue
        guidance = _PRONUNCIATION_GUIDANCE_BY_PAIR.get((learner, practice), "")
        if guidance:
            return guidance
    return ""


_TASK_CONTENT_CATEGORY_GUIDANCE: Dict[str, str] = {
    "learning_language": (
        "CATEGORY: language learning — output a DRILL SHEET, not study advice.\n"
        "planName states the MAIN GOAL (e.g. learn Chinese for travel). The step "
        "text says what to practice NOW — your job is to supply the actual "
        "pinyin/words to drill, NOT to repeat the step in Thai.\n"
        "WRONG `content`: 'ฝึกโทนเสียง 1-4 ด้วยคำตัวอย่าง...' (copies the step)\n"
        "RIGHT `content`: 'mā (มา) โทน 1 — มา | má (ม้า) โทน 2 — ม้า | ...' "
        "(lists real items)\n"
        "Pull drill items from the step topic and planName; keep them aligned.\n"
        "Structure `content` as a compact list (100–320 words). For each word:\n"
        "- pinyin / target form in the PRACTICE language (required)\n"
        "- pronunciation in the response language (e.g. คำอ่านไทย) — MANDATORY\n"
        "- tone or reading note when relevant (โทน 1–4, long/short vowel, etc.)\n"
        "- meaning and part of speech in the instruction language\n"
        "- learn-how: one line on how to practice (read aloud, shadow, "
        "notice tone) in the instruction language\n"
        "For each sentence:\n"
        "- sentence in practice language + pronunciation line in instruction "
        "language + brief breakdown in instruction language\n"
        "Quiz should test tones, pronunciation distinctions, meaning, or "
        "usage details present in the material."
    ),
    "learning": (
        "CATEGORY: learning (any academic subject that isn't a foreign "
        "language) — concepts, formulas, or skills.\n"
        "Output the actual STUDY MATERIAL for this step, grounded in planName "
        "and the step topic — not generic study advice.\n"
        "WRONG: 'review the concept and take notes' (advice, no content)\n"
        "RIGHT: the concept itself, e.g. 'Compound interest: A = P(1+r/n)^(nt). "
        "Worked: ฿10,000 at 6%/yr for 2y → …'\n"
        "Structure `content` (90–220 words): key term → plain definition → one "
        "worked mini-example with real numbers/steps → one common mistake → one "
        "recall prompt or practice problem to do now. Use the step's specific "
        "topic; never substitute a generic textbook example. Quiz tests the "
        "definitions, steps, or distinctions embedded in the material."
    ),
    "math": (
        "CATEGORY: math.\n"
        "Output real worked math for THIS step's topic from planName — not "
        "'practice some problems'.\n"
        "WRONG: 'work on quadratic equations today' (advice)\n"
        "RIGHT: state the rule, then SOLVE: 'Quadratic formula x=(-b±√(b²-4ac"
        "))/2a. Solve 2x²-4x-6=0: a=2,b=-4,c=-6 → disc=16+48=64 → x=(4±8)/4 → "
        "x=3 or x=-1.'\n"
        "Structure `content` (90–220 words): the rule/definition → 1–2 fully "
        "worked examples showing EVERY step → one common error → 1–2 practice "
        "problems for the user to solve now (give the answers at the end so it "
        "is self-checkable). Use the step's exact topic and difficulty. Quiz "
        "tests the method, a step, or the result of a worked problem."
    ),
    "science": (
        "CATEGORY: science (physics, chemistry, biology, etc.).\n"
        "Output the actual concept + mechanism for THIS step's topic — not "
        "'read about photosynthesis'.\n"
        "WRONG: 'review Newton's laws' (advice)\n"
        "RIGHT: state and explain, e.g. 'Newton's 2nd law: F=ma. A 2 kg cart "
        "pushed with 10 N accelerates a=F/m=5 m/s². Heavier mass → smaller "
        "acceleration for the same force.'\n"
        "Structure `content` (90–220 words): the principle/term → the mechanism "
        "or cause-and-effect in plain words → one concrete example or quick "
        "calculation with real units → one common misconception → one 'predict "
        "this' or recall prompt to do now. Quiz tests the mechanism, units, or "
        "cause-effect embedded in the material."
    ),
    "economics": (
        "CATEGORY: economics / business.\n"
        "Output the actual model or concept for THIS step with numbers — not "
        "'study supply and demand'.\n"
        "WRONG: 'learn about elasticity' (advice)\n"
        "RIGHT: define + compute, e.g. 'Price elasticity of demand = %Δqty / "
        "%Δprice. Price 100→120 (+20%), demand 50→40 (−20%) → elasticity = "
        "−1.0 (unit elastic): revenue unchanged.'\n"
        "Structure `content` (90–220 words): the term/model → plain intuition "
        "for why it works → one worked example with real figures → one common "
        "misreading → one 'what happens if…' prompt to reason through now. Quiz "
        "tests the model, the calculation, or the intuition in the material."
    ),
    "arts": (
        "CATEGORY: arts (drawing, painting, music, writing, design, etc.).\n"
        "Output a concrete technique + a do-it-now exercise for THIS step — not "
        "'practice drawing'.\n"
        "WRONG: 'work on your shading' (vague)\n"
        "RIGHT: name the technique and the drill, e.g. 'Value scale: draw 5 "
        "boxes, fill them 10%→90% black with even hatching. Squint to check the "
        "jumps are equal. Cue: press lighter for mid-tones, build up in passes.'\n"
        "Structure `content` (80–200 words): the technique/principle in plain "
        "terms → one or two specific exercises to do now (with steps, time, or "
        "counts) → one cue or common mistake → what 'good' looks like so the "
        "user can self-assess. Quiz may test the technique, terms, or cues."
    ),
    "psychology": (
        "CATEGORY: psychology (studying the subject).\n"
        "Output the actual concept + how it shows up in real life for THIS "
        "step — not 'read the chapter'.\n"
        "WRONG: 'review classical conditioning' (advice)\n"
        "RIGHT: define + example, e.g. 'Classical conditioning: a neutral "
        "stimulus paired with one that triggers a response eventually triggers "
        "it alone. Pavlov: bell (neutral) + food (UCS) → bell alone makes the "
        "dog salivate (CR).'\n"
        "Structure `content` (90–220 words): the concept/term → a plain "
        "definition → one everyday example → one related term it's often "
        "confused with (and the difference) → one application or recall prompt. "
        "Quiz tests the definitions, the example, or the distinctions."
    ),
    "exercise": (
        "CATEGORY: fitness / exercise.\n"
        "Output the exact workout to perform NOW for this step, matched to "
        "planName's goal and difficulty — not 'remember to exercise'.\n"
        "WRONG: 'do a good leg workout today' (vague)\n"
        "RIGHT: 'Goblet squat 3×10 @ tempo 3-1-1, rest 60s — cue: chest up, "
        "knees track toes; Walking lunge 3×12/leg …'\n"
        "Structure `content` (80–180 words): named movements, sets × reps, "
        "tempo, rest, form cues, and breathing. Scale to the plan's stated "
        "level. Quiz may test form cues, rep counts, or safety details."
    ),
    "travel": (
        "CATEGORY: travel.\n"
        "Output material usable for THIS trip/destination from planName — real "
        "phrases, names, numbers — not generic travel tips.\n"
        "WRONG: 'research local transport options' (homework, not material)\n"
        "RIGHT: actual lines/facts, e.g. 'BTS Skytrain: 17–62฿, runs 06:00–"
        "24:00; say \"ขอไป… / one ticket to …\"'\n"
        "Structure `content` (80–200 words): useful phrases or local terms, "
        "concrete logistics (transit, timing, cost), one cultural note, and one "
        "mini scenario to rehearse. Quiz tests the phrases/facts in the material."
    ),
    "finance": (
        "CATEGORY: finance.\n"
        "Output the concrete money step to do NOW for this plan's goal, with "
        "real numbers — not generic 'save more money' advice.\n"
        "WRONG: 'start budgeting and track spending' (vague)\n"
        "RIGHT: '50/30/20 on ฿30,000 income → needs ฿15,000, wants ฿9,000, "
        "save ฿6,000. List your 3 biggest \"wants\" to cut ฿1,000 this week.'\n"
        "Structure `content` (80–200 words): today's money action, a concrete "
        "number or formula, a worked example, and what to track. Quiz tests the "
        "numbers, terms, or habit logic in the material."
    ),
    "health": (
        "CATEGORY: health / wellness.\n"
        "Output the specific protocol to do NOW for this plan's goal — exact "
        "action, dose, duration — not 'be healthier'.\n"
        "WRONG: 'drink more water and sleep well' (generic)\n"
        "RIGHT: 'Box breathing 4-4-4-4 × 5 min: inhale 4s, hold 4s, exhale 4s, "
        "hold 4s. Cue: relax shoulders on each exhale.'\n"
        "Structure `content` (80–200 words): the specific action, duration or "
        "dosage, a mindfulness or body cue, and one reflection check-in. Avoid "
        "medical claims; keep it safe and general. Quiz tests the action details."
    ),
    "personal_development": (
        "CATEGORY: personal development.\n"
        "Output a concrete, do-it-now exercise tied to this plan's goal — not a "
        "motivational paragraph.\n"
        "WRONG: 'believe in yourself and stay consistent' (platitude)\n"
        "RIGHT: 'Eisenhower matrix: list today's 5 tasks, mark each "
        "Urgent/Important, do the 1 Important-not-Urgent first for 25 min.'\n"
        "Structure `content` (80–200 words): the habit or skill cue, a short "
        "reflection prompt, a concrete 2-minute action, and what to notice "
        "afterward. Quiz tests the framework or cues in the material."
    ),
    "other": (
        "CATEGORY: general.\n"
        "Output a ready-to-do mini-exercise that realizes THIS step toward "
        "planName's goal, with concrete specifics — never generic filler or a "
        "restatement of the step.\n"
        "Structure `content` (80–200 words) with steps the user performs now "
        "and concrete details the quiz can test."
    ),
}

_TASK_CONTENT_SYSTEM_PROMPT = (
    "You generate PRACTICE / STUDY MATERIAL for a single step of a user's plan. "
    "The plan can be about ANY subject — math, science, economics, finance, "
    "art, music, psychology, history, coding, cooking, travel, fitness, a "
    "language, anything. The user will WORK THROUGH whatever you produce, so "
    "output the actual material to practice or study with, in the form the "
    "CATEGORY GUIDANCE in the user message asks for. Language learning is just "
    "ONE possible subject — do NOT assume it unless the guidance says so.\n\n"
    "What 'material' means depends on the subject: worked math problems with "
    "solutions; a science concept with its mechanism and an example; an "
    "economics model with numbers; a finance action with real figures; an art "
    "technique with a concrete exercise; a workout with sets and cues; a "
    "language drill with target-language items. In EVERY case the rule is the "
    "same: output the substance the user studies/does — NEVER a paragraph that "
    "only restates the step or gives generic 'remember to study' advice.\n\n"
    "QUIZ SUPPORT: when you include a quiz, `content` must carry every fact "
    "the questions test — woven into the practice material itself. A careful "
    "reader should find every correct answer in `content`.\n\n"
    "Hierarchy: planName = the user's MAIN GOAL for the whole plan. The step "
    "text = what to practice on THIS step only. Ground the material in the "
    "step while advancing the planName goal.\n\n"
    "Follow the LANGUAGE instruction in the user message. Write `content`, the "
    "quiz, meanings, and notes in ONE coaching language — never mix two "
    "languages (e.g. English with Thai). ONLY when the plan is about learning a "
    "foreign language do the drill items themselves switch to that target "
    "language; for every other subject all output stays in the coaching "
    "language.\n\n"
    "Voice: direct and usable. No preamble, no meta commentary, no emoji.\n\n"
    "Hard rules:\n"
    "1. Return ONLY valid JSON. No markdown fences, no preamble.\n"
    "2. `content`: follow the CATEGORY GUIDANCE word range and structure. "
    "Ready to use immediately.\n"
    f"3. `quiz`: EXACTLY {_TASK_CONTENT_QUIZ_FIRST_BATCH} questions when "
    "checking the material helps. More are added later. For pure-doing steps "
    "with nothing to check, return []. Each item: 'question', 'choices' "
    "(EXACTLY 4 strings), 'answerIndex' (0–3), 'explanation' (2–3 sentences "
    "that TEACH: why correct using a detail from `content`; why others miss; "
    "one takeaway). Quiz text follows the LANGUAGE instruction.\n"
    "4. Questions answerable from `content` alone. Exactly one correct choice.\n\n"
    "Output JSON shape (quiz may be empty):\n"
    "{\n"
    '  "content": "<practice material>",\n'
    '  "quiz": [\n'
    '    {"question":"...","choices":["...","...","...","..."],'
    '"answerIndex":0,"explanation":"..."}\n'
    "  ]\n"
    "}\n"
)

_TASK_CONTENT_QUIZ_TOPUP_SYSTEM = (
    "You write ADDITIONAL multiple-choice quiz questions for practice material "
    "the user already has. Return ONLY valid JSON — no markdown fences.\n\n"
    "Hard rules:\n"
    "1. Output shape: {\"quiz\": [ ... ]} — do NOT include a `content` field.\n"
    "2. Each new question: 'question', 'choices' (EXACTLY 4 strings), "
    "'answerIndex' (0–3), 'explanation' (2–3 teaching sentences).\n"
    "3. Every question answerable from the PRACTICE MATERIAL alone.\n"
    "4. Do NOT repeat EXISTING QUESTIONS — test different facts in the "
    "material.\n"
    "5. Follow the LANGUAGE instruction in the user message for quiz text.\n"
)


def _normalize_task_content_category(raw: str) -> str:
    key = (raw or "").lower().strip().replace(" ", "_").replace("-", "_")
    if key in _TASK_CONTENT_PLAN_CATEGORIES:
        return key
    return _TASK_CONTENT_CATEGORY_ALIASES.get(key, "other")


def _task_content_category_key(
    task_category: str,
    task_title: str,
    task_detail: str,
    plan_name: str = "",
    plan_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Map task/plan signals to a content-template key."""
    blob = f"{plan_name} {task_category} {task_title} {task_detail}".lower()
    if isinstance(plan_context, dict):
        for field in ("detailPrompt", "planSummary", "category"):
            val = plan_context.get(field)
            if val:
                blob += f" {val}"
        day = plan_context.get("day")
        if isinstance(day, dict):
            for field in ("title", "summary", "tips"):
                val = day.get(field)
                if val:
                    blob += f" {val}"
            labels = day.get("taskLabels") or []
            if isinstance(labels, list):
                blob += " " + " ".join(str(x) for x in labels[:12] if x)
    if any(h in blob for h in _LANGUAGE_LEARNING_HINTS):
        return "learning_language"
    if _task_content_practice_language_from_hints(blob):
        return "learning_language"
    base = _normalize_task_content_category(task_category)
    if base == "other" and isinstance(plan_context, dict):
        base = _normalize_task_content_category(
            str(plan_context.get("category") or "")
        )
    if base in _TASK_CONTENT_CATEGORY_GUIDANCE:
        return base
    return "other"


def _task_content_analysis_blob(
    task_title: str,
    task_category: str,
    task_detail: str,
    plan_name: str,
    plan_context: Optional[Dict[str, Any]] = None,
    plan_context_block: str = "",
) -> str:
    """Blend planName goal, step text, and planner intent for language + focus."""
    return _practice_analysis_blob(
        task_title,
        task_category,
        task_detail,
        plan_context_block,
        plan_context,
        plan_name,
    )


def _task_content_grounding_instruction(
    plan_name: str = "",
    step_text: str = "",
    content_focus: str = "",
    parent_task: str = "",
) -> str:
    """Anchor material in planName goal + this step's practice task."""
    lines = [
        "CONTENT GROUNDING: planName is the MAIN GOAL. The step text is what "
        "the user practices RIGHT NOW. Every drill item must (a) fulfill the "
        "step description and (b) advance the planName goal — not generic "
        "textbook material. If the step names specific words or topics, use "
        "those exactly.",
    ]
    if plan_name:
        lines.append(f"Main goal (planName): {plan_name}")
    if step_text:
        lines.append(f"Step to practice now: {step_text}")
    if parent_task and parent_task.strip() != (step_text or "").strip():
        lines.append(f"Parent task context: {parent_task}")
    if content_focus:
        lines.append(f"Skill/topic focus for this step: {content_focus}")
    return "\n".join(lines)


# Generic verb-only steps that carry no drill specifics on their own.
_THIN_STEP_GENERIC_TOKENS = (
    "review", "practice", "study", "revise", "recap", "warm up", "warmup",
    "continue", "keep going", "do it", "train", "exercise", "read", "listen",
    "ทบทวน", "ฝึก", "ฝึกฝน", "อ่าน", "ทำต่อ", "เรียน", "ออกกำลัง", "ฟัง", "พูด",
)


def _task_content_step_is_thin(task_title: str, task_detail: str = "") -> bool:
    """True when the step text is too brief to specify the drill on its own.

    Thin steps (e.g. 'ทบทวน', 'practice speaking') must be resolved from the
    planner context, not invented — otherwise the model produces off-plan,
    misleading material.
    """
    step = (task_title or "").strip()
    if not step:
        return True
    has_digit = any(c.isdigit() for c in step)
    if _text_has_thai_script(step):
        # Thai has no word spaces — judge by character length. A specific step
        # ('ฝึกออกเสียงพินอิน 5 คำ') is longer than a bare verb ('ทบทวน').
        if has_digit and len(step) >= 12:
            return False
        return len(step) < 18
    words = step.split()
    if has_digit and len(words) >= 3:
        return False
    lower = step.lower()
    only_generic = lower in _THIN_STEP_GENERIC_TOKENS or all(
        w.strip(".,!?;:").lower() in _THIN_STEP_GENERIC_TOKENS or len(w) <= 2
        for w in words
    )
    return len(words) < 4 and (only_generic or len(step) < 22)


def _task_content_thin_step_directive(
    plan_name: str,
    step_text: str,
    has_plan_context: bool,
    practice_language: str = "",
    artifact: str = "content",
) -> str:
    """Force a brief step to be resolved from planner intent, not guesswork.

    artifact: 'content' for a drill sheet, 'practice_card' for a scenario card.
    """
    practice = (practice_language or "").strip()
    practice_foreign = bool(practice) and not _practice_languages_same(practice, "English")
    if artifact == "practice_card":
        drill_clause = (
            f"a scenario that rehearses the next sensible {practice} sub-skill "
            "for this goal"
            if practice_foreign
            else "a scenario that rehearses the next sensible sub-skill for this goal"
        )
    else:
        drill_clause = (
            f"the next sensible {practice} drill items for this goal"
            if practice_foreign
            else "the next sensible drill for this goal"
        )
    lines = [
        "THIN STEP NOTICE: the step text above is brief and does NOT fully "
        "specify what to practice. Do NOT guess a generic topic or invent "
        "material unrelated to the plan.",
    ]
    if has_plan_context:
        lines.append(
            "Resolve the step from the PLAN ARC / day context below "
            f"(main_goal, day_focus, day_steps) and produce {drill_clause}. "
            "The drill MUST stay inside the planName goal's domain and match "
            "where the user is in the plan."
        )
    else:
        goal = (plan_name or "").strip()
        lines.append(
            f"Stay strictly inside the MAIN GOAL"
            + (f" ('{goal}')" if goal else "")
            + f" and produce {drill_clause}. Choose the most plausible next "
            "sub-skill for that goal rather than generic filler, and keep it "
            "concrete enough to actually practice."
        )
    return "\n".join(lines)


def _task_content_drill_sheet_requirements(
    step_text: str,
    plan_name: str,
    practice_language: str,
    coaching_language: str,
) -> str:
    """Step-specific rules so output lists real drill items, not step paraphrase."""
    step = (step_text or "").strip()
    goal = (plan_name or "").strip()
    blob = f"{goal} {step}".lower()
    practice = (practice_language or "Chinese").strip()
    coach = (coaching_language or "Thai").strip()
    lines = [
        "DRILL SHEET (CRITICAL — read before writing `content`):",
        f"- planName goal: {goal or '(see above)'}",
        f"- step task: {step or '(see above)'}",
        f"- `content` MUST list real {practice} items (pinyin/words) the user "
        f"reads aloud — one item per line or bullet.",
        f"- FORBIDDEN: a {coach} paragraph that only explains what to do "
        f"(rephrasing the step). That is NOT practice material.",
        f"- REQUIRED: every drill line has {practice} form + {coach} คำอ่าน "
        f"+ meaning when relevant.",
    ]
    if not _practice_languages_same(practice, coach):
        translate_rule = (
            f"- CRITICAL — TRANSLATE, DON'T COPY: any example words in the step "
            f"(e.g. after 'เช่น' / 'such as' / 'e.g.' / ':') are written in "
            f"{coach} and name the MEANINGS to teach — they are NOT the drill "
            f"items. For each, output the {practice} word the user must learn, "
            f"plus its {coach} pronunciation, plus the {coach} meaning. Drilling "
            f"the {coach} word itself (e.g. a line like '{coach} word — คำอ่าน: "
            f"same {coach} word — meaning: English') is WRONG and useless: the "
            f"user is learning {practice}, not {coach}."
        )
        lines.append(translate_rule)
        if _practice_languages_same(practice, "Chinese"):
            lines.append(
                "  Worked example — step 'เช่น เนื้อ ไก่ หมู' (Thai meanings) "
                "MUST become Chinese drill items: '牛肉 niúròu (หนิว โร่ว) — "
                "เนื้อ | 鸡肉 jīròu (จี โร่ว) — ไก่ | 猪肉 zhūròu (จู โร่ว) — หมู'. "
                "Never list 'เนื้อ — คำอ่าน: เนื้อ'."
            )
    if "โทน" in step or "tone" in blob:
        lines.append(
            "- Step targets TONES: include ≥4 minimal pairs with tone labels "
            "(e.g. mā (มา) โทน 1, má (ม้า) โทน 2, mǎ (ม้า) โทน 3, mà (มา) โทน 4)."
        )
    if "พินอิน" in step or "pinyin" in blob:
        if any(n in step for n in ("6", "๖", "หก", "six")):
            lines.append(
                "- Step asks for 6 sounds: list EXACTLY 6 pinyin syllables with "
                "คำอ่าน (e.g. bō, pō, mō, fō, dō, tō or the set implied by the step)."
            )
        elif any(n in step for n in ("5", "๕", "ห้า", "five")):
            lines.append(
                "- Step asks for 5 items: list EXACTLY 5 pinyin words/syllables."
            )
        else:
            lines.append(
                "- Include ≥5 distinct pinyin syllables/words from the step topic."
            )
    if "จับคู่" in step or "match" in blob:
        lines.append(
            "- Matching exercise: 5 pinyin items with clear sound/spelling pairs."
        )
    if "ทักทาย" in step or "greet" in blob or "ท่องเที่ยว" in goal:
        lines.append(
            "- Travel/survival focus: include greeting phrases "
            "(e.g. nǐ hǎo, xièxie, zàijiàn, duìbuqǐ) with คำอ่าน + meaning."
        )
    return "\n".join(lines)


def _text_has_thai_script(text: str) -> bool:
    return any("\u0e00" <= c <= "\u0e7f" for c in (text or ""))


# Unicode ranges that prove a drill actually contains the target language, for
# script-distinct languages where copying the instruction language is
# unambiguous. (Latin-script targets like French can't be checked this way.)
_PRACTICE_SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "Chinese": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),               # CJK ideographs
    "Japanese": ((0x3040, 0x30FF), (0x4E00, 0x9FFF)),              # kana + kanji
    "Korean": ((0xAC00, 0xD7A3), (0x1100, 0x11FF)),               # hangul
    "Russian": ((0x0400, 0x04FF),),                                # Cyrillic
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),               # Arabic
    "Hindi": ((0x0900, 0x097F),),                                  # Devanagari
    "Thai": ((0x0E00, 0x0E7F),),                                   # Thai (as target)
}
# Toned pinyin vowels \u2014 count as Chinese content even without \u6c49\u5b57.
_PINYIN_TONE_CHARS = frozenset("\u0101\u00e1\u01ce\u00e0\u0113\u00e9\u011b\u00e8\u012b\u00ed\u01d0\u00ec\u014d\u00f3\u01d2\u00f2\u016b\u00fa\u01d4\u00f9\u01d6\u01d8\u01da\u01dc")


def _text_has_script(text: str, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    for c in (text or ""):
        o = ord(c)
        if any(lo <= o <= hi for lo, hi in ranges):
            return True
    return False


def _task_content_needs_practice_script_check(
    instruction_language: str, practice_language: str
) -> bool:
    """True when we can reliably verify the drill is in the practice language."""
    practice = (practice_language or "").strip()
    return (
        practice in _PRACTICE_SCRIPT_RANGES
        and not _practice_languages_same(instruction_language, practice)
    )


def _task_content_drill_uses_practice_language(
    content: str, practice_language: str
) -> bool:
    """Heuristic: does `content` actually contain practice-language material?

    Catches the failure where the model copies instruction-language example
    words instead of drilling the target language (e.g. listing Thai '\u0e40\u0e19\u0e37\u0e49\u0e2d'
    rather than Chinese '\u725b\u8089'). Only judges script-distinct targets; returns
    True (assume fine) for languages we can't check this way.
    """
    practice = (practice_language or "").strip()
    ranges = _PRACTICE_SCRIPT_RANGES.get(practice)
    if not ranges:
        return True
    text = content or ""
    if _text_has_script(text, ranges):
        return True
    if practice == "Chinese" and any(ch in _PINYIN_TONE_CHARS for ch in text):
        return True
    return False


def _task_content_practice_script_correction(
    practice_language: str, coaching_language: str
) -> str:
    """Forceful corrective appended on a regen when the first drill missed."""
    practice = (practice_language or "the target language").strip()
    coach = (coaching_language or "Thai").strip()
    extra = ""
    if practice == "Chinese":
        extra = (
            " Every drill line MUST start with \u6c49\u5b57 + pinyin (with tone marks), "
            "e.g. '\u725b\u8089 ni\u00far\u00f2u (\u0e2b\u0e19\u0e34\u0e27 \u0e42\u0e23\u0e48\u0e27) \u2014 \u0e40\u0e19\u0e37\u0e49\u0e2d'."
        )
    elif practice == "Japanese":
        extra = " Every drill line MUST contain Japanese script (\u304b\u306a/\u6f22\u5b57)."
    elif practice == "Korean":
        extra = " Every drill line MUST contain Hangul (\ud55c\uae00)."
    elif practice == "Russian":
        extra = " Every drill line MUST contain Cyrillic script (\u0430\u0431\u0432\u2026)."
    elif practice == "Arabic":
        extra = " Every drill line MUST contain Arabic script (\u0627\u0628\u062a\u2026)."
    elif practice == "Hindi":
        extra = " Every drill line MUST contain Devanagari script (\u0905\u0906\u0907\u2026)."
    elif practice == "Thai":
        extra = " Every drill line MUST contain Thai script (\u0e01\u0e02\u0e04\u2026)."
    return (
        f"CORRECTION \u2014 your previous attempt drilled {coach} words instead of "
        f"{practice}. That is WRONG and unusable. The user is learning "
        f"{practice}. Output the actual {practice} word/phrase for each item, "
        f"with {coach} pronunciation and meaning. Do NOT list {coach} words as "
        f"the drill items.{extra}"
    )


def _task_content_instruction_language(
    task_title: str,
    task_detail: str,
    language_selected: str,
) -> str:
    """Language of the step description (what to practice now), not planName."""
    step_blob = (task_title or "").strip()
    if not step_blob:
        step_blob = "\n".join(
            p for p in (task_detail,) if p and p.strip()
        ).strip()
    if _text_has_thai_script(step_blob):
        return "Thai"
    if len(step_blob) >= 4:
        return _practice_planner_language_fallback(step_blob, language_selected)
    return _ui_language_name(language_selected)


def _task_content_practice_language_blob(
    task_title: str,
    task_detail: str,
    plan_context: Optional[Dict[str, Any]],
    plan_name: str,
) -> str:
    parts: List[str] = []
    if plan_name:
        parts.append(plan_name)
    parts.extend(p for p in (task_title, task_detail) if p and p.strip())
    if isinstance(plan_context, dict):
        for field in (
            "detailPrompt", "planSummary", "category", "becomingPhrase", "planName",
        ):
            val = plan_context.get(field)
            if val:
                parts.append(str(val))
        day = plan_context.get("day")
        if isinstance(day, dict):
            for field in ("title", "summary", "tips"):
                val = day.get(field)
                if val:
                    parts.append(str(val))
            labels = day.get("taskLabels") or []
            if isinstance(labels, list):
                parts.extend(str(x) for x in labels[:12] if x)
    return " ".join(p for p in parts if p).lower()


def _task_content_practice_language_from_hints(blob: str) -> Optional[str]:
    """Detect target language from step/planner text — skip Thai as practice target."""
    for lang_name, hints in _PRACTICE_TARGET_LANGUAGE_HINTS.items():
        if lang_name == "Thai":
            continue
        if any(h in blob for h in hints):
            return lang_name
    return None


def _task_content_practice_language(
    task_title: str,
    task_detail: str,
    plan_context: Optional[Dict[str, Any]],
    plan_name: str,
    language_selected: str,
    planner_language: str = "",
    instruction_language: str = "",
    category_key: str = "",
) -> str:
    """Target language being practiced (e.g. Chinese), NOT the task title language.

    Only language-learning plans have a distinct "practice/target language".
    For every other genre (math, science, finance, fitness, art, …) the user is
    NOT drilling a foreign language, so the practice language is simply the
    language the step is written in. Returning the instruction language here is
    what neutralises all the downstream foreign-drill machinery (pronunciation
    guides, translate-don't-copy rules, mixed-language layout) for non-language
    genres — otherwise a plan that merely mentions a place or word in another
    language would be mistaken for a language course.
    """
    instr_resolved = (
        (instruction_language or "").strip()
        or _task_content_instruction_language(
            task_title, task_detail, language_selected,
        )
    )
    if category_key and category_key != "learning_language":
        return instr_resolved
    blob = _task_content_practice_language_blob(
        task_title, task_detail, plan_context, plan_name,
    )
    hint_lang = _task_content_practice_language_from_hints(blob)
    if hint_lang:
        return hint_lang
    instr = instr_resolved
    reply_language = _ui_language_name(language_selected)
    sanitized = _sanitize_planner_target_language(
        planner_language, blob, instr, reply_language,
    )
    if sanitized:
        return sanitized
    return instr


def _task_content_planner_intent_note(
    instruction_language: str,
    practice_language: str,
    category_key: str,
    task_title: str,
    task_detail: str,
    plan_name: str = "",
) -> str:
    """Clarify planName goal vs step task for foreign-language drills."""
    instr = (instruction_language or "").strip()
    practice = (practice_language or "").strip()
    blob = f"{plan_name} {task_title} {task_detail}".lower()
    foreign_step = (
        not _practice_languages_same(instr, practice)
        or _task_content_practice_language_from_hints(blob) is not None
        or category_key == "learning_language"
    )
    if not foreign_step:
        return ""
    goal_line = (
        f"planName main goal: {plan_name}. " if plan_name else ""
    )
    return (
        f"PLANNER INTENT: {goal_line}The step text is in {instr} — it describes "
        f"WHAT to practice {practice} on this step, not the language of the "
        f"drill items. `content` MUST include real {practice} practice items "
        f"(pinyin, tone pairs, words, sentences). A coaching-only paraphrase "
        f"with no {practice} material is WRONG. Quiz must test {practice} "
        f"details from `content`. Questions and explanations use the coaching "
        f"language only (no English mixed in)."
    )


def _task_content_language_mix_instruction(
    instruction_language: str,
    practice_language: str,
    category_key: str = "other",
    artifact: str = "content",
    reply_language: Optional[str] = None,
) -> str:
    """Instruction lang from task text; practice lang = target language learned.

    artifact: 'content' for generate_task_content, 'practice_card' for
    generate_practice. reply_language (languageSelected) is the responding
    language for quiz/coaching; drill items use practice_language.
    """
    instr = (instruction_language or "English").strip()
    practice = (practice_language or instr).strip()
    respond = (reply_language or instr).strip()
    if artifact == "practice_card":
        output_desc = (
            "card (`situation`, choice `label`s, `afterChoiceNote`, "
            "`coachFollowUp`)"
        )
        mixed_output = (
            f"Write `situation` in {respond} (languageSelected) with "
            f"{practice} phrases to rehearse. Write choice `label`s in {respond} "
            f"(with {practice} terms and pronunciation in {respond} where "
            f"spoken). Write `afterChoiceNote` and `coachFollowUp` entirely in "
            f"{respond}."
        )
    else:
        output_desc = "`content` and quiz"
        mixed_output = (
            f"The `content` drill list MUST contain real {practice} items "
            f"(pinyin, tones, words — not a paraphrase of the step). "
            f"Write meanings, learn-how, grammar notes, quiz questions, "
            f"choices, and explanations ONLY in {respond}. "
            f"Do NOT mix English with {respond}. English is forbidden unless "
            f"{respond} is English."
        )
    is_foreign_language_drill = (
        category_key == "learning_language"
        or (
            category_key in ("learning", "other")
            and not _practice_languages_same(instr, practice)
        )
    )
    if is_foreign_language_drill:
        if _practice_languages_same(respond, practice):
            return (
                f"LANGUAGE: write {output_desc} in {respond}. Include actual "
                f"{practice} drill material from the step intent — not only "
                f"instructions about studying. Do NOT default to English."
            )
        pron = _task_content_pronunciation_guidance(instr, practice, respond)
        base = (
            f"LANGUAGE (mixed — CRITICAL): the step text is in {instr} — it "
            f"describes what to practice. planName sets the main goal. The user "
            f"is practicing {practice}. Coaching language is {respond} — use it "
            f"for ALL quiz text, "
            f"explanations, meanings, and pronunciation guides. "
            f"Drill items stay in {practice}. {mixed_output} "
            f"Every {practice} syllable/word MUST include pronunciation in "
            f"{respond} (e.g. Thai คำอ่าน for Chinese pinyin)."
        )
        return f"{base}\n{pron}" if pron else base
    return (
        f"LANGUAGE: write {output_desc} entirely in {respond}. "
        f"Ground in planName (main goal) and the step text (practice task). "
        f"Do NOT use English unless languageSelected is English."
    )


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _extract_json_object(raw: str) -> Optional[Any]:
    """Parse a JSON value from a model reply, tolerating prose/fences/trailing text.

    The model occasionally wraps the JSON in a markdown fence, adds a sentence
    before or after it, or appends commentary. A plain json.loads then fails and
    the user sees "content came back malformed". We first try a clean parse, then
    fall back to slicing the outermost {...} / [...] span and parsing that.
    Returns the parsed value, or None when nothing parseable is present.
    """
    text = _strip_json_fence(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not candidates:
        return None
    start = min(candidates)
    close_ch = "}" if text[start] == "{" else "]"
    end = text.rfind(close_ch)
    if end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# Human-readable fallbacks chatgpt_wrapper returns INSTEAD of model content when
# the call times out, is rate-limited, trips the circuit breaker, is refused, or
# comes back empty/truncated. They are not JSON, so a JSON endpoint would parse
# them as "malformed output" and mask the real cause. Detect them so we can
# return an accurate, retryable error and never cache/charge for them.
_MODEL_UNAVAILABLE_SENTINELS = (
    "request timed out",
    "connection issue",
    "experiencing high demand",
    "service temporarily unavailable",
    "service is temporarily unavailable",
    "service configuration error",
    "response was cut off due to token limit",
    "the model returned an empty response",
    "i encountered an unexpected error",
    "network connection issue",
    "unable to process request:",
    "please try rephrasing your request",
)


def _looks_like_model_unavailable(text: str) -> bool:
    low = (text or "").strip().lower()
    # Real content is long JSON; the sentinels are short single sentences. The
    # length guard keeps a genuine long answer that happens to mention one of
    # these phrases from being misread as an outage.
    if not low or len(low) > 400:
        return False
    return any(s in low for s in _MODEL_UNAVAILABLE_SENTINELS)


def _task_content_model_bundle(
    _tier: str = "free", category_key: str = "other",
) -> Tuple[str, int, str]:
    """All signed-in tiers use the flagship model for practice-step material.

    Upsell is monthly credits (free 10 / plus 30 tasks per month, premium
    unlimited), not model quality. Returns (model_id, max_completion_tokens, model_tier).
    """
    tokens = 2100 if category_key == "learning_language" else 1700
    return COACH_MODEL_PREMIUM, tokens, "flagship"


def _task_content_response_extras(
    _tier: str,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
) -> Dict[str, Any]:
    bundle_model, _, bundle_tier = _task_content_model_bundle()
    return {
        "model_used": model or bundle_model,
        "model_tier": model_tier or bundle_tier,
        "quality_upsell": False,
    }


_TASK_CONTENT_CHARGED_MAP_MAX = 64  # bound Firestore doc size for premium months


def _trim_charged_tasks_map(charged: Dict[str, Any]) -> Dict[str, Any]:
    if not charged or len(charged) <= _TASK_CONTENT_CHARGED_MAP_MAX:
        return charged or {}
    # Keep the most recently inserted keys (Py3.7+ dict preserves insertion).
    keys = list(charged.keys())[-_TASK_CONTENT_CHARGED_MAP_MAX:]
    return {k: charged[k] for k in keys}


def _evo_task_content_db():
    """Firestore for EVO app user data (evoforluanching), not the scheduler project."""
    try:
        from evo_firebase import evo_firestore
        db = evo_firestore()
        if db is not None:
            return db
    except Exception as e:
        logger.warning("EVO Firestore unavailable: %s", e)
    return firestore.client()


def _coach_tier_for_uid(uid: str) -> str:
    """Read coachSubscription from the EVO app Firestore project."""
    if not uid:
        return "free"
    try:
        snap = (_evo_task_content_db().collection("users").document(uid)
                .collection("coachSubscription").document("current").get())
        if not snap.exists:
            return "free"
        data = snap.to_dict() or {}
        persisted = (data.get("tier") or "free").lower()
        if persisted not in ("plus", "premium"):
            return "free"
        expires_at = data.get("expiresAt")
        if expires_at:
            try:
                if hasattr(expires_at, "timestamp") and expires_at.timestamp() < time.time():
                    return "free"
            except Exception:
                pass
        return persisted
    except Exception as e:
        logger.warning("task_content: subscription lookup failed for %s: %s", uid, e)
        return "free"


def _resolve_task_content_uid_tier(
    req: https_fn.Request, payload: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """Resolve uid/tier for practice-step content (EVO app project).

    The hosting CF project differs from evoforluanching, so default
    verify_id_token often fails. We verify against EVO when configured,
    else accept client uid when a Bearer token is present.
    """
    uid, tier = _verify_coach_tier(req)
    if uid:
        tier = _coach_tier_for_uid(uid)
        return uid, tier

    auth_header = req.headers.get("Authorization") or req.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return None, "free"
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None, "free"

    client_uid = (payload.get("uid") or "").strip()
    evo_uid = None
    try:
        from evo_firebase import verify_evo_id_token
        evo_uid = verify_evo_id_token(token)
    except Exception as e:
        logger.warning("EVO token verify import failed: %s", e)

    if evo_uid:
        return evo_uid, _coach_tier_for_uid(evo_uid)
    if client_uid:
        logger.info(
            "task_content: client uid fallback for %s (set EVO_FIREBASE_SERVICE_ACCOUNT_JSON)",
            client_uid[:8],
        )
        return client_uid, _coach_tier_for_uid(client_uid)
    return None, "free"


def _resolve_evo_authenticated_uid_tier(
    req: https_fn.Request, payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], str]:
    """EVO app auth for write endpoints (subscription)."""
    return _resolve_task_content_uid_tier(req, payload or {})


def _task_content_quota_peek(
    uid: str, tier: str, tz_offset_minutes: Any = 0, task_quota_key: Optional[str] = None
) -> Tuple[bool, int, Optional[int], bool]:
    """Read monthly quota without spending a credit.

    Returns (allowed, current_count, cap, already_charged_for_task).
    Quota is PER TASK (task_quota_key), not per step."""
    cap = _TASK_CONTENT_MONTHLY_CAP.get(tier)
    is_unlimited = tier == "premium"
    if not uid:
        return (True, 0, cap, False)
    task_key = (task_quota_key or "").strip() or "_unknown"
    try:
        db = _evo_task_content_db()
        ref = (db.collection("users").document(uid)
                 .collection("taskContent_usage")
                 .document(_local_period_key(tz_offset_minutes, "%Y-%m")))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {}
        current = int(data.get("count", 0) or 0)
        charged = data.get("tasks") or {}
        if charged.get(task_key):
            return (True, current, cap, True)
        if (not is_unlimited) and cap is not None and current >= cap:
            return (False, current, cap, False)
        return (True, current, cap, False)
    except Exception as e:
        logger.warning("task content quota peek failed for %s: %s — allowing", uid, e)
        return (True, 0, cap, False)


def _task_content_quota_commit(
    uid: str, tier: str, tz_offset_minutes: Any = 0, task_quota_key: Optional[str] = None
) -> Tuple[int, Optional[int]]:
    """Spend one monthly credit for this task after a successful generation.
    No-op if this task was already charged this month."""
    cap = _TASK_CONTENT_MONTHLY_CAP.get(tier)
    is_unlimited = tier == "premium"
    if not uid:
        return (0, cap)
    task_key = (task_quota_key or "").strip() or "_unknown"
    try:
        db = _evo_task_content_db()
        ref = (db.collection("users").document(uid)
                 .collection("taskContent_usage")
                 .document(_local_period_key(tz_offset_minutes, "%Y-%m")))
        transaction = db.transaction()

        @firestore.transactional
        def _txn(tx):
            snap = ref.get(transaction=tx)
            data = snap.to_dict() if snap.exists else {}
            current = int(data.get("count", 0) or 0)
            charged = dict(data.get("tasks") or {})
            if charged.get(task_key):
                return (current, cap)
            if (not is_unlimited) and cap is not None and current >= cap:
                return (current, cap)
            charged[task_key] = True
            charged = _trim_charged_tasks_map(charged)
            new_count = current + 1
            tx.set(ref, {"count": new_count, "tier": tier, "tasks": charged,
                         "updatedAt": firestore.SERVER_TIMESTAMP}, merge=True)
            return (new_count, cap)

        return _txn(transaction)
    except Exception as e:
        logger.warning("task content quota commit failed for %s: %s", uid, e)
        return (0, cap)


def _task_content_cache_get(uid: str, task_id: str) -> Optional[Dict[str, Any]]:
    if not uid or not task_id:
        return None
    try:
        snap = (_evo_task_content_db().collection("users").document(uid)
                  .collection("taskContent").document(task_id).get())
        if not snap.exists:
            return None
        return snap.to_dict() or None
    except Exception as e:
        logger.warning("task content cache read failed for %s/%s: %s", uid, task_id, e)
        return None


def _task_content_cache_set(uid: str, task_id: str, payload: Dict[str, Any]) -> None:
    if not uid or not task_id:
        return
    try:
        (_evo_task_content_db().collection("users").document(uid)
            .collection("taskContent").document(task_id)
            .set({**payload, "generatedAt": firestore.SERVER_TIMESTAMP}, merge=True))
    except Exception as e:
        logger.warning("task content cache write failed for %s/%s: %s", uid, task_id, e)


def _normalize_quiz(
    raw_quiz: Any,
    max_items: int = _TASK_CONTENT_QUIZ_TARGET,
) -> List[Dict[str, Any]]:
    """Defensive server-side quiz validation. Drops malformed items; keeps
    only questions with exactly 4 string choices and an in-range answerIndex."""
    out = []
    if not isinstance(raw_quiz, list):
        return out
    cap = max(1, min(int(max_items), _TASK_CONTENT_QUIZ_TARGET))
    for q in raw_quiz[:cap]:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        choices = q.get("choices")
        if not question or not isinstance(choices, list) or len(choices) != 4:
            continue
        norm_choices = [str(c).strip()[:200] for c in choices]
        if any(not c for c in norm_choices):
            continue
        try:
            answer_index = int(q.get("answerIndex"))
        except (TypeError, ValueError):
            continue
        if answer_index < 0 or answer_index > 3:
            continue
        out.append({
            "question": question[:400],
            "choices": norm_choices,
            "answerIndex": answer_index,
            "explanation": str(q.get("explanation", "")).strip()[:700],
        })
    return out


def _quiz_question_key(question: str) -> str:
    return " ".join(str(question or "").lower().split())[:240]


def _merge_quiz_lists(
    existing: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
    target: int = _TASK_CONTENT_QUIZ_TARGET,
) -> List[Dict[str, Any]]:
    seen = {_quiz_question_key(q.get("question")) for q in existing}
    merged = list(existing)
    for q in new_items:
        if len(merged) >= target:
            break
        key = _quiz_question_key(q.get("question"))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(q)
    return merged[:target]


def _format_existing_quiz_for_prompt(quiz: List[Dict[str, Any]]) -> str:
    lines = []
    for i, q in enumerate(quiz, 1):
        lines.append(f"{i}. {q.get('question', '')}")
    return "\n".join(lines) if lines else "(none yet)"


def _task_content_generate_quiz_batch(
    content: str,
    existing_quiz: List[Dict[str, Any]],
    count: int,
    model: str,
    task_title: str,
    task_detail: str,
    language_instruction: str = "",
    coaching_language: str = "",
) -> List[Dict[str, Any]]:
    """One lightweight LLM pass — quiz only, no practice material."""
    count = max(1, min(int(count), _TASK_CONTENT_QUIZ_TOPUP_BATCH))
    user_lines = [
        f"Write EXACTLY {count} NEW quiz question(s) for this practice material.",
        "",
        "PRACTICE MATERIAL:",
        content[:2000],
        "",
        "EXISTING QUESTIONS (do not repeat):",
        _format_existing_quiz_for_prompt(existing_quiz),
    ]
    if task_title:
        user_lines += ["", f"STEP CONTEXT: {task_title}"]
    if task_detail and task_detail != task_title:
        user_lines.append(task_detail[:800])
    if language_instruction:
        user_lines += ["", language_instruction]
    user_lines += [
        "",
        "Each explanation must teach: why the correct answer is right (cite the "
        "material), why distractors miss, and one takeaway.",
        "",
        'Return JSON: {"quiz":[...]} per the contract.',
    ]
    try:
        from chatgpt_wrapper import chat_with_gpt
        response_text = chat_with_gpt(
            system_prompt=_TASK_CONTENT_QUIZ_TOPUP_SYSTEM,
            user_prompt="\n".join(user_lines),
            model=model,
            max_completion_tokens=900,
            auto_detect_language=False,
            reply_language=_language_name_to_chat_code(coaching_language) if coaching_language else None,
            response_format={"type": "json_object"},
        )
        if not response_text or not response_text.strip():
            return []
        parsed = _extract_json_object(response_text)
        raw_quiz = parsed.get("quiz") if isinstance(parsed, dict) else None
        return _normalize_quiz(raw_quiz, max_items=count)
    except Exception as e:
        logger.info("task_content quiz top-up skipped: %s", e)
        return []


def _task_content_ensure_quiz_target(
    content: str,
    quiz: List[Dict[str, Any]],
    model: str,
    task_title: str,
    task_detail: str,
    language_instruction: str = "",
    coaching_language: str = "",
) -> List[Dict[str, Any]]:
    """Grow quiz to _TASK_CONTENT_QUIZ_TARGET via batched top-up calls."""
    quiz = _normalize_quiz(quiz, max_items=_TASK_CONTENT_QUIZ_TARGET)
    if not quiz:
        return quiz
    attempts = 0
    while (
        len(quiz) < _TASK_CONTENT_QUIZ_TARGET
        and attempts < _TASK_CONTENT_QUIZ_TOPUP_MAX_ATTEMPTS
    ):
        need = min(
            _TASK_CONTENT_QUIZ_TOPUP_BATCH,
            _TASK_CONTENT_QUIZ_TARGET - len(quiz),
        )
        batch = _task_content_generate_quiz_batch(
            content, quiz, need, model, task_title, task_detail,
            language_instruction, coaching_language,
        )
        if not batch:
            break
        merged = _merge_quiz_lists(quiz, batch, _TASK_CONTENT_QUIZ_TARGET)
        if len(merged) <= len(quiz):
            break
        quiz = merged
        attempts += 1
    return quiz


def _task_content_maybe_topup_cached_quiz(
    uid: str,
    task_id: str,
    cached: Dict[str, Any],
    tier: str,
    task_title: str,
    task_detail: str,
    step_fingerprint: str,
    language_instruction: str = "",
) -> List[Dict[str, Any]]:
    """Backfill older caches (<5 questions) without spending monthly quota."""
    content = (cached.get("content") or "").strip()
    quiz = _normalize_quiz(cached.get("quiz"), max_items=_TASK_CONTENT_QUIZ_TARGET)
    if not content or not quiz or len(quiz) >= _TASK_CONTENT_QUIZ_TARGET:
        return quiz
    category_key = cached.get("contentCategory") or "other"
    model, _, _ = _task_content_model_bundle(tier, category_key)
    lang_inst = language_instruction
    if not lang_inst:
        instr = (
            cached.get("instructionLanguage")
            or cached.get("stepLanguage")
            or cached.get("replyLanguage")
        )
        practice = (
            cached.get("practiceLanguage")
            or cached.get("plannerLanguage")
            or instr
        )
        if instr:
            coach = (
                cached.get("coachingLanguage")
                or cached.get("replyLanguage")
                or instr
            )
            lang_inst = _task_content_language_mix_instruction(
                instr,
                practice,
                cached.get("contentCategory") or "other",
                reply_language=coach,
            )
    coach_lang = cached.get("coachingLanguage") or cached.get("replyLanguage") or ""
    expanded = _task_content_ensure_quiz_target(
        content, quiz, model, task_title, task_detail, lang_inst, coach_lang,
    )
    if len(expanded) > len(quiz):
        _task_content_cache_set(uid, task_id, {
            **cached,
            "content": content,
            "quiz": expanded,
            "stepFingerprint": step_fingerprint or cached.get("stepFingerprint"),
        })
        logger.info(
            "generate_task_content: quiz top-up cache uid=%s task=%s %d→%d",
            uid, task_id, len(quiz), len(expanded),
        )
    return expanded


@https_fn.on_request(
    memory=1024,
    max_instances=20,
    timeout_sec=120,
    cpu=1,
    secrets=_LLM_SECRETS,
)
def generate_task_content(req: https_fn.Request) -> https_fn.Response:
    """Generate practice-step drill material + optional quiz.

    Thai task titles describe foreign-language drills; content uses practice
    language (e.g. Chinese pinyin) with languageSelected for quiz/explanations.
    Tier + monthly quota server-side; cache free."""
    if req.method == 'OPTIONS':
        return handle_preflight_request()

    if req.method != 'POST':
        return create_response(
            success=False, message='Method not allowed',
            error='Only POST method is allowed', status_code=405,
        )

    try:
        payload = req.get_json(silent=True) or {}
        task_id = (payload.get("taskId") or "").strip()
        # Real task id for quota accounting (one credit per task, not per
        # step). Falls back to the cache id when the client doesn't send it.
        task_quota_key = (payload.get("taskQuotaKey") or "").strip() or task_id
        task_title = (payload.get("taskTitle") or "").strip()
        task_category = (payload.get("taskCategory") or "").strip()
        task_detail = (payload.get("taskDetail") or "").strip()
        plan_id = (payload.get("planId") or "").strip()
        plan_name = (payload.get("planName") or "").strip()
        language_selected = (
            payload.get("languageSelected") or payload.get("language") or "english"
        ).lower()
        force_refresh = bool(payload.get("forceRefresh", False))
        tz_offset_minutes = payload.get("tzOffsetMinutes", 0)
        step_fingerprint = (payload.get("stepFingerprint") or "").strip()[:64]
        plan_day_number = payload.get("planDayNumber")
        user_state = payload.get("userState") if isinstance(payload.get("userState"), dict) else {}

        if not task_id or not task_title:
            return create_response(
                success=False, message='Missing required fields',
                error='`taskId` and `taskTitle` are required', status_code=400,
            )
        if (len(task_title) > 400 or len(task_detail) > 4000
                or len(task_category) > 120 or len(plan_name) > 200):
            return create_response(
                success=False, message='Payload too large',
                error='Task content payload exceeds size limits.', status_code=413,
            )

        uid, tier = _resolve_task_content_uid_tier(req, payload)
        is_paid = tier in ("plus", "premium")

        reply_language = _ui_language_name(language_selected)

        plan_context = None
        if plan_id:
            plan_context = _load_practice_plan_context(
                uid, plan_id, plan_day_number,
            )
            if plan_context is None:
                # The client asked us to ground in a specific plan but we could
                # not load it — the model will fall back to generic, off-intent
                # material. Surface this loudly; it is the usual cause of
                # "content doesn't match the planner".
                logger.warning(
                    "generate_task_content: plan_id=%s sent but plan context "
                    "did NOT load (uid=%s, day=%s) — output will not be grounded "
                    "in planner intent",
                    plan_id, uid or "anon", plan_day_number,
                )
        plan_context_block = (
            _format_practice_plan_context_block(plan_context)
            if plan_context else ""
        )
        effective_plan_name = _resolve_task_content_plan_name(
            plan_name, plan_context,
        )
        analysis_blob = _task_content_analysis_blob(
            task_title,
            task_category,
            task_detail,
            effective_plan_name,
            plan_context,
            plan_context_block,
        )
        instruction_language = _task_content_instruction_language(
            task_title, task_detail, language_selected,
        )
        category_key = _task_content_category_key(
            task_category, task_title, task_detail, effective_plan_name, plan_context,
        )
        planner_language, content_focus = _analyze_practice_intent(
            analysis_blob, language_selected, plan_context, instruction_language,
            effective_plan_name,
        )
        practice_language = _task_content_practice_language(
            task_title,
            task_detail,
            plan_context,
            effective_plan_name,
            language_selected,
            planner_language,
            instruction_language,
            category_key=category_key,
        )
        coaching_language = _task_content_coaching_language(
            instruction_language,
            reply_language,
            practice_language,
            category_key,
            task_title,
            task_detail,
        )
        language_instruction = _task_content_language_mix_instruction(
            instruction_language,
            practice_language,
            category_key,
            reply_language=coaching_language,
        )
        planner_intent_note = _task_content_planner_intent_note(
            instruction_language,
            practice_language,
            category_key,
            task_title,
            task_detail,
            effective_plan_name,
        )
        category_guidance = _TASK_CONTENT_CATEGORY_GUIDANCE[category_key]
        grounding_instruction = _task_content_grounding_instruction(
            effective_plan_name, task_title, content_focus, task_detail,
        )

        if not _coach_rate_allow(uid or "", is_paid):
            return create_response(
                success=False, message='Rate limit',
                error='Too many requests in a short window. Try again soon.',
                status_code=429,
            )

        # 1) Cache check — free, never spends quota.
        if uid and not force_refresh:
            cached = _task_content_cache_get(uid, task_id)
            if cached and cached.get("content"):
                cached_fp = (cached.get("stepFingerprint") or "").strip()
                cached_logic_v = int(cached.get("contentLogicVersion") or 0)
                stale_logic = cached_logic_v < _TASK_CONTENT_LOGIC_VERSION
                serve_cache = (
                    not step_fingerprint or not cached_fp
                    or cached_fp == step_fingerprint
                )
                if serve_cache and stale_logic:
                    # Generation logic improved since this entry was written.
                    # Regenerate it — but only if that won't cost the user a new
                    # credit (task already charged this month) or there is quota
                    # headroom. If they are capped, keep serving the old content
                    # rather than blocking access.
                    _ok, _c, _cap, already = _task_content_quota_peek(
                        uid, tier, tz_offset_minutes, task_quota_key
                    )
                    if already or _ok:
                        serve_cache = False
                        logger.info(
                            "generate_task_content: cache logic v%d<v%d — "
                            "regenerating uid=%s task=%s (already_charged=%s)",
                            cached_logic_v, _TASK_CONTENT_LOGIC_VERSION,
                            uid, task_id, already,
                        )
                if serve_cache:
                    quiz = _task_content_maybe_topup_cached_quiz(
                        uid,
                        task_id,
                        cached,
                        tier,
                        task_title,
                        task_detail,
                        step_fingerprint,
                        language_instruction,
                    )
                    _allowed, count, cap, _charged = _task_content_quota_peek(
                        uid, tier, tz_offset_minutes, task_quota_key
                    )
                    return create_response(
                        data={
                            "content": cached.get("content"),
                            "quiz": quiz,
                            "tier": tier,
                            "source": "cache",
                            "used": count,
                            "cap": cap,
                            **_task_content_response_extras(
                                tier,
                                cached.get("model_used"),
                                cached.get("model_tier"),
                            ),
                        },
                        message='Task content (cached)',
                    )

        if not uid:
            return create_response(
                success=False, message='Sign in required',
                error='Sign in to generate practice material for this step. '
                      'Make sure you are logged in and try again.',
                status_code=401,
            )

        # 2) Monthly quota peek — credit spent only after successful generation.
        allowed, count, cap, _already_charged = _task_content_quota_peek(
            uid, tier, tz_offset_minutes, task_quota_key
        )
        if not allowed:
            return create_response(
                success=False, message='Monthly limit reached',
                error='You have used your task learning generations this month. '
                      'Upgrade for more.',
                status_code=402,
                metadata={"capReached": True, "tier": tier, "cap": cap, "used": count},
            )

        # 3) Generate — category-specific material + mixed language layout.
        model, max_tokens, model_tier = _task_content_model_bundle(tier, category_key)
        user_prompt_lines: List[str] = []
        if effective_plan_name:
            user_prompt_lines += [
                "MAIN GOAL (planName — overall purpose of this plan):",
                f"- planName: {effective_plan_name}",
                "",
            ]
        user_prompt_lines += [
            "THIS STEP — what to practice now (step description, not the main goal):",
            f"- step: {task_title}",
        ]
        if task_detail and task_detail.strip() != (task_title or "").strip():
            user_prompt_lines.append(f"- parent_task: {task_detail}")
        is_language_plan = category_key == "learning_language"
        user_prompt_lines += [
            "",
            f"SELECTED RESPONSE LANGUAGE (languageSelected): {reply_language}",
            f"WRITE EVERYTHING IN (one language only — content, quiz, notes, "
            f"explanations): {coaching_language}",
        ]
        if is_language_plan:
            # Only a foreign-language plan has a distinct target language for
            # the drill items; for every other subject the material is written
            # entirely in the coaching language above.
            user_prompt_lines += [
                f"TASK TEXT LANGUAGE (instructions only): {instruction_language}",
                f"PRACTICE/TARGET LANGUAGE (drill items only): {practice_language}",
            ]
        user_prompt_lines.append(language_instruction)
        if planner_intent_note:
            user_prompt_lines += ["", planner_intent_note]
        user_prompt_lines += [
            "",
            grounding_instruction,
        ]
        if _task_content_step_is_thin(task_title, task_detail):
            user_prompt_lines += [
                "",
                _task_content_thin_step_directive(
                    effective_plan_name,
                    task_title,
                    has_plan_context=bool(plan_context),
                    practice_language=practice_language,
                ),
            ]
        user_prompt_lines += [
            "",
            "CATEGORY GUIDANCE:",
            category_guidance,
        ]
        pronunciation_guidance = _task_content_pronunciation_guidance(
            instruction_language, practice_language, reply_language,
        )
        if pronunciation_guidance:
            user_prompt_lines += ["", pronunciation_guidance]
        if (
            category_key == "learning_language"
            or not _practice_languages_same(instruction_language, practice_language)
        ):
            user_prompt_lines += [
                "",
                _task_content_drill_sheet_requirements(
                    task_title,
                    effective_plan_name,
                    practice_language,
                    coaching_language,
                ),
            ]
        if task_category:
            user_prompt_lines += ["", f"- taskCategory: {task_category}"]
        if plan_context_block:
            user_prompt_lines += ["", plan_context_block]
        if content_focus:
            user_prompt_lines += [
                "",
                "PLANNER INTENT FOCUS (from planName + step + plan arc):",
                f"- {content_focus}",
            ]
        if plan_day_number is not None:
            try:
                user_prompt_lines.append(f"(plan day {int(plan_day_number)})")
            except (TypeError, ValueError):
                pass
        if user_state.get("restDayFlag"):
            user_prompt_lines.append(
                "(recovery: rest day — shorter, lighter drill material; no guilt.)"
            )
        if user_state.get("missedYesterday"):
            user_prompt_lines.append(
                "(recovery: missed yesterday — gentle return rep; no streak shame.)"
            )
        user_prompt_lines.append("")
        if is_language_plan:
            produce_line = (
                f"Produce a ready-to-use DRILL SHEET for the step above (real "
                f"{practice_language} items — not a {coaching_language} summary "
                f"of the step), aligned with planName, plus exactly "
            )
        else:
            produce_line = (
                f"Produce ready-to-use STUDY/PRACTICE MATERIAL for the step "
                f"above (the real substance to work through per the CATEGORY "
                f"GUIDANCE — not a summary of the step), written in "
                f"{coaching_language} and aligned with planName, plus exactly "
            )
        user_prompt_lines.append(
            produce_line
            + f"{_TASK_CONTENT_QUIZ_FIRST_BATCH} quiz questions when a quiz "
            "helps (more are added later). Embed every fact the quiz will test "
            "inside the material. Write rich 2–3 sentence explanations per the "
            "JSON contract."
        )
        user_prompt = "\n".join(user_prompt_lines)

        logger.info(
            "generate_task_content: uid=%s tier=%s model=%s category=%s "
            "reply_lang=%s coach_lang=%s instr_lang=%s practice_lang=%s "
            "plan_id=%s has_plan_ctx=%s task_id=%s peek_used=%s/%s",
            uid, tier, model, category_key, reply_language, coaching_language,
            instruction_language, practice_language, plan_id or "-",
            bool(plan_context), task_id,
            count, cap if cap is not None else "∞",
        )

        from chatgpt_wrapper import chat_with_gpt
        response_text = chat_with_gpt(
            system_prompt=_TASK_CONTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            max_completion_tokens=max_tokens,
            auto_detect_language=False,
            reply_language=_language_name_to_chat_code(coaching_language),
            response_format={"type": "json_object"},
        )

        if not response_text or not response_text.strip():
            return create_response(
                success=False, message='Empty model response',
                error='Could not generate content right now. Try again.',
                status_code=502,
            )

        parsed = _extract_json_object(response_text)
        if not isinstance(parsed, dict):
            # The wrapper returns a plain-English sentence (not JSON) when the
            # call times out / is rate-limited / trips the breaker / is refused.
            # Surface that as a retryable "busy" error instead of a confusing
            # "malformed" one — and do NOT spend a credit or retry on it.
            if _looks_like_model_unavailable(response_text):
                logger.warning(
                    "generate_task_content: model unavailable for uid=%s: %s",
                    uid, response_text.strip()[:160],
                )
                return create_response(
                    success=False, message='Service busy',
                    error='The AI is busy right now. Please try again in a moment.',
                    status_code=503,
                )
            # One forceful retry: re-ask for strict JSON only. Common when the
            # model prepends a sentence or truncates the object.
            logger.warning(
                "generate_task_content: JSON parse failed uid=%s — retrying once",
                uid,
            )
            try:
                from chatgpt_wrapper import chat_with_gpt as _chat_json
                retry_text = _chat_json(
                    system_prompt=_TASK_CONTENT_SYSTEM_PROMPT,
                    user_prompt=(
                        f"{user_prompt}\n\nIMPORTANT: Return ONLY the JSON object "
                        "described in the contract — no prose, no markdown fences, "
                        "nothing before or after it."
                    ),
                    model=model,
                    max_completion_tokens=max_tokens,
                    auto_detect_language=False,
                    reply_language=_language_name_to_chat_code(coaching_language),
                    response_format={"type": "json_object"},
                )
                parsed = _extract_json_object(retry_text)
            except Exception as _retry_err:
                logger.warning(
                    "generate_task_content: JSON retry errored uid=%s: %s",
                    uid, _retry_err,
                )
        if not isinstance(parsed, dict):
            return create_response(
                success=False, message='Bad model output',
                error='The content came back malformed. Try again.', status_code=502,
            )

        content = (parsed.get("content") or "").strip() if isinstance(parsed, dict) else ""
        quiz = _normalize_quiz(
            parsed.get("quiz") if isinstance(parsed, dict) else None,
            max_items=_TASK_CONTENT_QUIZ_FIRST_BATCH,
        )
        if not content:
            return create_response(
                success=False, message='Bad model output',
                error='No content produced. Try again.', status_code=502,
            )

        # Output-quality guard: when the target language is script-distinct
        # (Chinese/Japanese/Korean) and differs from the step language, verify
        # the drill is actually in that language. The common failure is copying
        # the instruction-language example words. If so, regenerate ONCE with a
        # forceful correction before charging the user.
        if _task_content_needs_practice_script_check(
            instruction_language, practice_language
        ) and not _task_content_drill_uses_practice_language(
            content, practice_language
        ):
            logger.warning(
                "generate_task_content: drill missing %s script (copied %s?) — "
                "regenerating once uid=%s task=%s",
                practice_language, instruction_language, uid, task_id,
            )
            correction = _task_content_practice_script_correction(
                practice_language, coaching_language
            )
            try:
                from chatgpt_wrapper import chat_with_gpt
                retry_text = chat_with_gpt(
                    system_prompt=_TASK_CONTENT_SYSTEM_PROMPT,
                    user_prompt=f"{user_prompt}\n\n{correction}",
                    model=model,
                    max_completion_tokens=max_tokens,
                    auto_detect_language=False,
                    reply_language=_language_name_to_chat_code(coaching_language),
                    response_format={"type": "json_object"},
                )
                retry_parsed = _extract_json_object(retry_text or "")
                retry_content = (
                    (retry_parsed.get("content") or "").strip()
                    if isinstance(retry_parsed, dict) else ""
                )
                if retry_content and _task_content_drill_uses_practice_language(
                    retry_content, practice_language
                ):
                    content = retry_content
                    quiz = _normalize_quiz(
                        retry_parsed.get("quiz"),
                        max_items=_TASK_CONTENT_QUIZ_FIRST_BATCH,
                    )
                    logger.info(
                        "generate_task_content: correction succeeded uid=%s task=%s",
                        uid, task_id,
                    )
                else:
                    logger.warning(
                        "generate_task_content: correction still missing %s "
                        "script uid=%s task=%s",
                        practice_language, uid, task_id,
                    )
            except Exception as e:
                logger.warning(
                    "generate_task_content: practice-script correction failed "
                    "uid=%s: %s", uid, e,
                )

        if quiz:
            quiz = _task_content_ensure_quiz_target(
                content, quiz, model, task_title, task_detail,
                language_instruction, coaching_language,
            )
            logger.info(
                "generate_task_content: quiz batched uid=%s task=%s count=%d",
                uid, task_id, len(quiz),
            )

        count, cap = _task_content_quota_commit(
            uid, tier, tz_offset_minutes, task_quota_key
        )
        normalized = {
            "content": content,
            "quiz": quiz,
            "stepFingerprint": step_fingerprint or None,
            "model_used": model,
            "model_tier": model_tier,
            "contentCategory": category_key,
            "replyLanguage": reply_language,
            "instructionLanguage": instruction_language,
            "practiceLanguage": practice_language,
            "contentFocus": content_focus or None,
            "plannerLanguage": planner_language or None,
            "coachingLanguage": coaching_language,
            "contentLogicVersion": _TASK_CONTENT_LOGIC_VERSION,
        }
        _task_content_cache_set(uid, task_id, normalized)

        return create_response(
            data={
                "content": content,
                "quiz": quiz,
                "tier": tier,
                "source": "fresh",
                "used": count,
                "cap": cap,
                **_task_content_response_extras(tier, model, model_tier),
            },
            message='Task content generated',
        )

    except Exception as e:
        logger.error("generate_task_content error: %s", e)
        traceback.print_exc()
        return create_response(
            success=False, message='Task content failed',
            error="We couldn't generate this right now. Try again in a moment.",
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


@https_fn.on_request(
    memory=512, max_instances=10, timeout_sec=60, cpu=1,
    secrets=[_EVO_FIREBASE_SA_SECRET],
)
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
        payload = req.get_json(silent=True) or {}
        uid, _ = _resolve_evo_authenticated_uid_tier(req, payload)
        if not uid:
            return create_response(
                success=False, message='Auth required',
                error='Sign in to confirm your purchase.', status_code=401,
            )

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

        # Write subscription in evoforluanching Firestore (not scheduler project).
        from firebase_admin import firestore as _fs
        db = _evo_task_content_db()
        sub_ref = (
            db.collection("users").document(uid)
            .collection("coachSubscription").document("current")
        )
        sub_ref.set({
            "tier": granted_tier,
            "source": source,
            "sku": sku,
            "originalTransactionId": v.get("original_transaction_id") or transaction_id,
            "expiresAt": _fs.Timestamp.from_seconds(expires_ms // 1000) if hasattr(_fs, "Timestamp") else datetime.utcfromtimestamp(expires_ms // 1000),
            "updatedAt": _fs.SERVER_TIMESTAMP if hasattr(_fs, "SERVER_TIMESTAMP") else datetime.now(timezone.utc).isoformat(),
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
    now = datetime.now(timezone.utc).isoformat()
    
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
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    
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
        morning_mode = data.get("morning_mode") or "todo_coach"
        rag_queries = {
            "todo_coach": "morning encouragement today tasks schedule",
            "love_warmth": "morning warmth self compassion belonging",
            "funny_boost": "morning humor playful encouragement",
            "identity_cheer": "streak identity becoming growth morning",
            "gentle_rest": "rest recovery gentle morning no pressure",
        }
        rag_query = rag_queries.get(str(morning_mode).strip().lower(), rag_queries["todo_coach"])
        user_id = data.get('user_id')
        if user_id and isinstance(user_id, str) and user_id.strip():
            user_id = user_id.strip()
            try:
                from user_memory import retrieve_user_context
                user_context = retrieve_user_context(user_id, rag_query, top_k=5)
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
            morning_mode=morning_mode,
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
            event_time = datetime.now(timezone.utc)
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