# Google Cloud Function for School Schedule Optimization
# This function provides HTTP endpoints for generating and managing school schedules

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
import json
import logging
from typing import Dict, Any, Optional
import sys
import os
import traceback
import time

# Import school_scheduler from the same directory
try:
    #from school_scheduler import SchoolScheduler
    
    from school_scheduler import SchoolScheduler
    SCHEDULER_AVAILABLE = True
    logging.info("SchoolScheduler imported successfully")
except ImportError as e:
    logging.error(f"Failed to import SchoolScheduler: {e}")
    logging.error(f"Import traceback: {traceback.format_exc()}")
    SchoolScheduler = None
    SCHEDULER_AVAILABLE = False

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=5)

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

def validate_schedule_request(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate the incoming schedule request data"""
    try:
        required_fields = ['n_teachers', 'grades']
        optional_fields = {
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
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Convert and validate n_teachers
        try:
            data['n_teachers'] = int(data['n_teachers'])
        except (ValueError, TypeError):
            return False, "n_teachers must be a valid integer"
        
        if data['n_teachers'] <= 0:
            return False, "n_teachers must be a positive integer"
        
        if data['n_teachers'] > 50:
            return False, "n_teachers cannot exceed 50"
        
        # Validate grades
        if not isinstance(data['grades'], list) or len(data['grades']) == 0:
            return False, "grades must be a non-empty list"
        
        if len(data['grades']) > 20:
            return False, "grades list cannot exceed 20 items"
        
        # Validate individual grades
        for grade in data['grades']:
            if not isinstance(grade, str) or len(grade) == 0:
                return False, f"Invalid grade format: {grade}"
        
        # Set default values for optional fields
        for field, default_value in optional_fields.items():
            if field not in data:
                data[field] = default_value
        
        # Convert and validate optional numeric fields
        numeric_fields = ['pe_day', 'n_pe_periods', 'start_hour', 'n_hours', 'lunch_hour', 'days_per_week', 'homeroom_mode']
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = int(data[field])
                except (ValueError, TypeError):
                    return False, f"{field} must be a valid integer"
        
        # Validate ranges for optional fields
        if data['pe_day'] < 1 or data['pe_day'] > 7:
            return False, "pe_day must be between 1 and 7"
        
        if data['n_pe_periods'] < 0:
            return False, "n_pe_periods must be a non-negative integer"
        
        if data['start_hour'] < 0 or data['start_hour'] > 23:
            return False, "start_hour must be between 0 and 23"
        
        if data['n_hours'] < 1 or data['n_hours'] > 12:
            return False, "n_hours must be between 1 and 12"
        
        if data['lunch_hour'] < 1 or data['lunch_hour'] > data['n_hours']:
            return False, "lunch_hour must be between 1 and n_hours"
        
        if data['days_per_week'] < 1 or data['days_per_week'] > 7:
            return False, "days_per_week must be between 1 and 7"
        
        if data['homeroom_mode'] not in [0, 1, 2]:
            return False, "homeroom_mode must be 0, 1, or 2"
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error in validate_schedule_request: {e}")
        return False, f"Validation error: {str(e)}"

def create_cors_headers() -> Dict[str, str]:
    """Create CORS headers for responses"""
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin',
        'Access-Control-Max-Age': '3600',
        'Access-Control-Allow-Credentials': 'true',
        'Content-Type': 'application/json'
    }

@https_fn.on_request(
    max_instances=3,
  
    #timeout_seconds=300  # 5 minutes timeout
)
def generate_schedule(req: https_fn.Request) -> https_fn.Response:
    """Generate a school schedule based on provided parameters"""
    
    start_time = time.time()
    headers = create_cors_headers()
    
    try:
        # Handle preflight requests
        if req.method == 'OPTIONS':
            return https_fn.Response(
                '',
                status=200,
                headers=headers
            )
        
        if req.method != 'POST':
            logger.warning(f"Invalid method {req.method} for generate_schedule")
            return https_fn.Response(
                json.dumps({
                    'error': f'Method {req.method} not allowed',
                    'message': 'This endpoint only accepts POST requests with JSON data',
                    'endpoints': {
                        'POST /generate_schedule': 'Generate a school schedule (requires JSON data)',
                        'GET /health_check': 'Check service health',
                        'GET /get_schedule_info': 'Get API information and examples',
                        'GET /debug': 'Get debug information'
                    }
                }),
                status=405,
                headers=headers
            )
        
        # Parse request data
        try:
            data = req.get_json()
            if data is None:
                logger.warning("No JSON data provided in request")
                return https_fn.Response(
                    json.dumps({
                        'error': 'No JSON data provided',
                        'message': 'This endpoint requires a POST request with JSON data in the body',
                        'example': {
                            'n_teachers': 13,
                            'grades': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'M1', 'M2', 'M3']
                        }
                    }),
                    status=400,
                    headers=headers
                )
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return https_fn.Response(
                json.dumps({
                    'error': f'Invalid JSON: {str(e)}',
                    'message': 'This endpoint requires a POST request with valid JSON data in the body',
                    'content_type': 'Make sure to set Content-Type: application/json header'
                }),
                status=400,
                headers=headers
            )
        
        # Validate request data
        is_valid, error_message = validate_schedule_request(data)
        if not is_valid:
            logger.warning(f"Invalid request data: {error_message}")
            return https_fn.Response(
                json.dumps({'error': error_message}),
                status=400,
                headers=headers
            )
        
        # Check if SchoolScheduler is available
        if not SCHEDULER_AVAILABLE:
            logger.error("SchoolScheduler module not available")
            return https_fn.Response(
                json.dumps({'error': 'SchoolScheduler module not available'}),
                status=500,
                headers=headers
            )
        
        # Generate schedule
        logger.info(f"Generating schedule with parameters: {data}")
        
        try:
            scheduler = SchoolScheduler()
            scheduler.set_pe_constraints_enabled(data.get('enable_pe_constraints', False))
            scheduler.set_homeroom_mode(data.get('homeroom_mode', 1))
            
            # Get inputs
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
                return https_fn.Response(
                    json.dumps({'error': 'Failed to initialize scheduler inputs'}),
                    status=500,
                    headers=headers
                )
            
            # Get model and solution
            logger.info("Building optimization model...")
            scheduler.get_model()
            
            logger.info("Solving optimization problem...")
            if not scheduler.get_solution():
                logger.warning("No feasible solution found for the given constraints")
                return https_fn.Response(
                    json.dumps({'error': 'No feasible solution found for the given constraints'}),
                    status=422,
                    headers=headers
                )
            
            # Prepare response data
            logger.info("Preparing response data...")
            schedule_df = scheduler.schedule_df
            homeroom_df = scheduler.homeroom_df
            
            # Convert DataFrames to JSON-serializable format
            schedule_data = []
            if schedule_df is not None:
                schedule_data = schedule_df.to_dict('records')
            
            homeroom_data = []
            if homeroom_df is not None:
                homeroom_data = homeroom_df.to_dict('records')
                
            reformated_schedule_data = []
            for item in schedule_data:
                converted_start_time = item["TimeSlot"].split("-")[0]
                reformated_schedule_data.append({
                    "subject": item["Grade"],
                    "grade": item["Grade"],
                    "teacher": item["Teacher"],
                    "day": item["DayName"],
                    "period": item["Hour"],
                    "time": converted_start_time,
                    "timeslot": item["TimeSlot"],
                    "duration": 1  
                })
            response_data = {
                'success': True,
                'message': 'Schedule generated successfully',
                'schedule': reformated_schedule_data,
                'homeroom': homeroom_data,
                'parameters': data,
                'metadata': {
                    'total_assignments': len(schedule_data) if schedule_data else 0,
                    'homeroom_assignments': len(homeroom_data) if homeroom_data else 0,
                    'processing_time_seconds': round(time.time() - start_time, 2)
                }
            }
            
            logger.info(f"Schedule generated successfully in {time.time() - start_time:.2f} seconds")
            return https_fn.Response(
                json.dumps(response_data, default=str),
                status=200,
                headers=headers
            )
            
        except Exception as e:
            logger.error(f"Error in schedule generation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return https_fn.Response(
                json.dumps({'error': f'Schedule generation failed: {str(e)}'}),
                status=500,
                headers=headers
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_schedule: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return https_fn.Response(
            json.dumps({'error': f'Internal server error: {str(e)}'}),
            status=500,
            headers=headers
        )

@https_fn.on_request()
def health_check(req: https_fn.Request) -> https_fn.Response:
    """Health check endpoint"""
    
    headers = create_cors_headers()
    if req.method == 'OPTIONS':
            return https_fn.Response(
                '',
                status=200,
                headers=headers
            )
    if req.method != 'GET':
        return https_fn.Response(
            json.dumps({'error': 'Only GET method is allowed'}),
            status=405,
            headers=headers
        )
    
    health_data = {
        'status': 'healthy',
        'service': 'school-schedule-optimizer',
        'version': '1.0.0',
        'scheduler_available': SCHEDULER_AVAILABLE,
        'python_version': sys.version,
        'environment': 'production'
    }
    
    return https_fn.Response(
        json.dumps(health_data),
        status=200,
        headers=headers
    )

@https_fn.on_request()
def get_schedule_info(req: https_fn.Request) -> https_fn.Response:
    """Get information about available schedule parameters and constraints"""
    
    headers = create_cors_headers()
    
    if req.method != 'GET':
        return https_fn.Response(
            json.dumps({'error': 'Only GET method is allowed'}),
            status=405,
            headers=headers
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
            'n_teachers': 'Number of teachers (integer, 1-50)',
            'grades': 'List of grade levels (e.g., ["P1", "P2", "P3"], max 20 items)'
        },
        'optional_parameters': {
            'pe_teacher': 'Physical education teacher ID (default: "T13")',
            'pe_grades': 'Grades that have PE (default: ["P4", "P5", "P6", "M1", "M2", "M3"])',
            'pe_day': 'Day for PE classes (default: 3)',
            'n_pe_periods': 'Number of PE periods (default: 6)',
            'start_hour': 'Starting hour (default: 8)',
            'n_hours': 'Number of hours per day (default: 8)',
            'lunch_hour': 'Lunch hour (default: 5)',
            'days_per_week': 'Days per week (default: 5)',
            'enable_pe_constraints': 'Enable PE constraints (default: false)',
            'homeroom_mode': 'Homeroom mode: 0=none, 1=basic, 2=advanced (default: 1)'
        },
        'example_request': {
            'n_teachers': 13,
            'grades': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'M1', 'M2', 'M3'],
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
        },
        'constraints': {
            'max_teachers': 50,
            'max_grades': 20,
            'max_hours_per_day': 10,
            'max_days_per_week': 7
        }
    }
    
    return https_fn.Response(
        json.dumps(info_data, indent=2),
        status=200,
        headers=headers
    )

@https_fn.on_request()
def debug(req: https_fn.Request) -> https_fn.Response:
    """Debug endpoint to check system status and diagnose issues"""
    
    headers = create_cors_headers()
    
    if req.method != 'GET':
        return https_fn.Response(
            json.dumps({'error': 'Only GET method is allowed'}),
            status=405,
            headers=headers
        )
    
    try:
        # Test imports
        import_status = {}
        try:
            import pandas as pd
            import_status['pandas'] = {'available': True, 'version': pd.__version__}
        except ImportError as e:
            import_status['pandas'] = {'available': False, 'error': str(e)}
        
        try:
            import pulp
            import_status['pulp'] = {'available': True, 'version': pulp.__version__}
        except ImportError as e:
            import_status['pulp'] = {'available': False, 'error': str(e)}
        
        try:
            import ortools
            import_status['ortools'] = {'available': True, 'version': ortools.__version__}
        except ImportError as e:
            import_status['ortools'] = {'available': False, 'error': str(e)}
        
        try:
            import plotly
            import_status['plotly'] = {'available': True, 'version': plotly.__version__}
        except ImportError as e:
            import_status['plotly'] = {'available': False, 'error': str(e)}
        
        # Test SchoolScheduler
        scheduler_status = {}
        if SCHEDULER_AVAILABLE:
            try:
                scheduler = SchoolScheduler()
                scheduler_status['import'] = True
                scheduler_status['instantiation'] = True
                
                # Test basic functionality
                test_result = scheduler.get_inputs(
                    n_teachers=3,
                    grades=["P1", "P2"],
                    pe_teacher="T3",
                    pe_grades=["P2"],
                    pe_day=3,
                    n_pe_periods=1,
                    start_hour=8,
                    n_hours=4,
                    lunch_hour=3,
                    days_per_week=3,
                    enable_pe_constraints=False,
                    homeroom_mode=1
                )
                scheduler_status['get_inputs'] = test_result
                
                if test_result:
                    scheduler.get_model()
                    scheduler_status['get_model'] = True
                    
                    # Don't test get_solution as it might timeout
                    scheduler_status['get_solution'] = 'not_tested'
                
            except Exception as e:
                scheduler_status['error'] = str(e)
                scheduler_status['traceback'] = traceback.format_exc()
        else:
            scheduler_status['import'] = False
        
        debug_data = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'platform': sys.platform,
            'environment_variables': {
                'FUNCTION_TARGET': os.environ.get('FUNCTION_TARGET', 'not_set'),
                'FUNCTION_REGION': os.environ.get('FUNCTION_REGION', 'not_set'),
                'GOOGLE_CLOUD_PROJECT': os.environ.get('GOOGLE_CLOUD_PROJECT', 'not_set')
            },
            'import_status': import_status,
            'scheduler_status': scheduler_status,
            'scheduler_available': SCHEDULER_AVAILABLE,
            'memory_info': {
                'available': 'check_psutil_import'
            }
        }
        
        # Try to get memory info if psutil is available
        try:
            import psutil
            debug_data['memory_info'] = {
                'available': True,
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': psutil.virtual_memory().available // (1024 * 1024)
            }
        except ImportError:
            debug_data['memory_info'] = {'available': False, 'error': 'psutil not available'}
        
        return https_fn.Response(
            json.dumps(debug_data, indent=2, default=str),
            status=200,
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return https_fn.Response(
            json.dumps({
                'error': f'Debug endpoint failed: {str(e)}',
                'traceback': traceback.format_exc()
            }),
            status=500,
            headers=headers
        )