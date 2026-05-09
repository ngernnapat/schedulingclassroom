"""
Local server to run the async planner generation API.

Run with: python run_async_local.py

Then test with:
- POST http://localhost:5001/startPlannerGeneration
- GET http://localhost:5001/getPlannerStatus?jobId=xxx
- GET http://localhost:5001/getPlannerResult?jobId=xxx
"""

import os
import sys
import json
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load from multiple locations
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent / '../.env',
        Path.home() / '.env',
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment from: {env_path}")
            break
except ImportError:
    print("Note: python-dotenv not installed, skipping .env file loading")

from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure the OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("\n" + "=" * 60)
    print("ERROR: OPENAI_API_KEY environment variable not set!")
    print("=" * 60)
    print("\nOption 1 - Create a .env file in the functions folder:")
    print(f"  {Path(__file__).parent / '.env'}")
    print("  Content: OPENAI_API_KEY=sk-your-key-here")
    print("\nOption 2 - Set environment variable:")
    print("  export OPENAI_API_KEY='sk-your-key-here'")
    print("\nOption 3 - Run with inline variable:")
    print("  OPENAI_API_KEY='sk-your-key-here' python run_async_local.py")
    print("=" * 60 + "\n")
    sys.exit(1)

# Set up the path to import local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Pydantic first (needed before importing generate_planner_content)
from pydantic import ValidationError

# Import from main module - these don't depend on Firebase
from generate_planner_content import (
    GeneratePlannerRequest,
    PlannerContent,
    ChatWrapper,
    ChatWrapperConfig,
    PlannerGenerationError,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Job storage (in-memory for local testing)
_job_store = {}
_job_lock = threading.Lock()


def estimate_generation_time(total_days: int, fast_mode: bool) -> int:
    """Estimate generation time based on plan parameters"""
    base_time = 30 if fast_mode else 60
    per_day_time = 2 if fast_mode else 4
    return base_time + (total_days * per_day_time)


def create_job(request_data: GeneratePlannerRequest) -> dict:
    """Create a new generation job"""
    job_id = f"plan_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    estimated_time = estimate_generation_time(request_data.totalDays, request_data.fastMode)
    
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "progress_message": "Job created, waiting to start...",
        "estimated_seconds_remaining": estimated_time,
        "created_at": now,
        "updated_at": now,
        "request": request_data.model_dump(),
        "current_stage": "initializing",
        "stages_completed": 0,
        "total_stages": 4,
        "result": None,
        "error": None
    }
    
    with _job_lock:
        _job_store[job_id] = job
    
    return job


def update_job_progress(job_id: str, progress: int, message: str, stage: str, stages_done: int):
    """Update job progress"""
    with _job_lock:
        if job_id in _job_store:
            job = _job_store[job_id]
            job["progress"] = progress
            job["progress_message"] = message
            job["current_stage"] = stage
            job["stages_completed"] = stages_done
            job["updated_at"] = datetime.utcnow().isoformat()
            job["status"] = "processing"
            
            # Update estimated time remaining
            try:
                created = datetime.fromisoformat(job["created_at"])
                elapsed = (datetime.utcnow() - created).total_seconds()
                if progress > 0:
                    total_estimated = (elapsed / progress) * 100
                    remaining = max(0, int(total_estimated - elapsed))
                    job["estimated_seconds_remaining"] = remaining
            except:
                pass
            
            print(f"[Job {job_id}] Progress: {progress}% - {message}")


def complete_job(job_id: str, result: PlannerContent):
    """Mark job as completed with result"""
    with _job_lock:
        if job_id in _job_store:
            job = _job_store[job_id]
            job["status"] = "completed"
            job["progress"] = 100
            job["progress_message"] = "Generation complete!"
            job["current_stage"] = "completed"
            job["stages_completed"] = 4
            job["estimated_seconds_remaining"] = 0
            job["updated_at"] = datetime.utcnow().isoformat()
            job["result"] = result.model_dump()
            print(f"[Job {job_id}] COMPLETED!")


def fail_job(job_id: str, error: str):
    """Mark job as failed with error"""
    with _job_lock:
        if job_id in _job_store:
            job = _job_store[job_id]
            job["status"] = "failed"
            job["progress_message"] = "Generation failed"
            job["updated_at"] = datetime.utcnow().isoformat()
            job["error"] = error
            print(f"[Job {job_id}] FAILED: {error}")


def get_job(job_id: str) -> dict:
    """Get job status"""
    with _job_lock:
        return _job_store.get(job_id)


def run_generation_job(job_id: str, request_data: GeneratePlannerRequest):
    """Run the actual generation in background"""
    try:
        # Stage 1: Initializing (0-10%)
        update_job_progress(job_id, 5, "Initializing planner generation...", "initializing", 0)
        time.sleep(0.5)
        
        # Stage 2: Context extraction (10-25%)
        update_job_progress(job_id, 10, "Analyzing your requirements...", "extracting_context", 1)
        
        # Create chat wrapper
        chat = ChatWrapper(ChatWrapperConfig())
        
        # Stage 3: Generating days (25-90%)
        update_job_progress(job_id, 25, f"Generating {request_data.totalDays}-day plan...", "generating_days", 2)
        
        # Generate content
        content = chat.generate(request_data)
        
        # Stage 4: Finalizing (90-100%)
        update_job_progress(job_id, 90, "Finalizing your planner...", "finalizing", 3)
        time.sleep(0.5)
        
        # Complete
        complete_job(job_id, content)
        
    except PlannerGenerationError as e:
        fail_job(job_id, e.user_message)
    except Exception as e:
        fail_job(job_id, f"An unexpected error occurred: {str(e)}")


# =========================
# API Endpoints
# =========================

@app.route('/startPlannerGeneration', methods=['POST', 'OPTIONS'])
def start_planner_generation():
    """
    Start a planner generation job.
    Returns immediately with a job ID for polling.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        payload = request.get_json() or {}
        parsed_request = GeneratePlannerRequest(**payload)
        
        # Create job
        job = create_job(parsed_request)
        job_id = job["job_id"]
        
        print(f"\n{'='*60}")
        print(f"Starting new job: {job_id}")
        print(f"Plan: {parsed_request.planName} ({parsed_request.totalDays} days)")
        print(f"Category: {parsed_request.category}")
        print(f"Fast Mode: {parsed_request.fastMode}")
        print(f"{'='*60}\n")
        
        # Start background thread for generation
        thread = threading.Thread(
            target=run_generation_job,
            args=(job_id, parsed_request),
            daemon=True
        )
        thread.start()
        
        # Return immediately
        return jsonify({
            "jobId": job_id,
            "status": "pending",
            "estimatedSeconds": job["estimated_seconds_remaining"],
            "pollUrl": f"/getPlannerStatus?jobId={job_id}",
            "message": "Your planner is being generated. Poll for status updates.",
            "tips": [
                "Generation typically takes 1-3 minutes",
                "You can poll /getPlannerStatus for progress updates",
                "Result will be available at /getPlannerResult when complete"
            ]
        }), 202
        
    except ValidationError as ve:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
        return jsonify({"error": "Invalid request", "details": errors}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/getPlannerStatus', methods=['GET', 'OPTIONS'])
def get_planner_status():
    """Get the status of a planner generation job."""
    if request.method == 'OPTIONS':
        return '', 204
    
    job_id = request.args.get('jobId')
    if not job_id:
        return jsonify({"error": "Missing jobId parameter"}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found", "jobId": job_id}), 404
    
    response = {
        "jobId": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "progressMessage": job["progress_message"],
        "currentStage": job["current_stage"],
        "stagesCompleted": job["stages_completed"],
        "totalStages": job["total_stages"],
        "estimatedSecondsRemaining": job["estimated_seconds_remaining"],
        "createdAt": job["created_at"],
        "updatedAt": job["updated_at"]
    }
    
    if job["status"] == "failed":
        response["error"] = job["error"]
    
    if job["status"] == "completed" and job.get("result"):
        result = job["result"]
        response["resultSummary"] = {
            "planName": result.get("planName"),
            "category": result.get("category"),
            "totalDays": result.get("totalDays"),
            "ready": True
        }
    
    return jsonify(response)


@app.route('/getPlannerResult', methods=['GET', 'OPTIONS'])
def get_planner_result():
    """Get the full result of a completed planner generation job."""
    if request.method == 'OPTIONS':
        return '', 204
    
    job_id = request.args.get('jobId')
    if not job_id:
        return jsonify({"error": "Missing jobId parameter"}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found", "jobId": job_id}), 404
    
    if job["status"] != "completed":
        return jsonify({
            "error": "Job not completed",
            "status": job["status"],
            "progress": job["progress"]
        }), 400
    
    return jsonify(job["result"])


@app.route('/generate_planner_content', methods=['POST', 'OPTIONS'])
def generate_planner_content_sync():
    """
    Synchronous planner generation endpoint (for compatibility with existing frontend).
    This blocks until the planner is fully generated.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        payload = request.get_json() or {}
        parsed_request = GeneratePlannerRequest(**payload)
        
        print(f"\n{'='*60}")
        print(f"Synchronous generation request")
        print(f"Plan: {parsed_request.planName} ({parsed_request.totalDays} days)")
        print(f"Category: {parsed_request.category}")
        print(f"Fast Mode: {parsed_request.fastMode}")
        print(f"{'='*60}\n")
        
        # Create chat wrapper and generate directly
        chat = ChatWrapper(ChatWrapperConfig())
        content = chat.generate(parsed_request)
        
        print(f"✓ Generation complete!")
        return jsonify(content.model_dump())
        
    except PlannerGenerationError as e:
        return jsonify({
            "error": "Generation failed",
            "message": e.user_message
        }), 500
    except ValidationError as ve:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
        return jsonify({"error": "Invalid request", "details": errors}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "error": "Generation failed",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "activeJobs": len(_job_store),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        "service": "Async Planner Generation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /startPlannerGeneration": "Start a new planner generation job",
            "GET /getPlannerStatus?jobId=xxx": "Get job status and progress",
            "GET /getPlannerResult?jobId=xxx": "Get completed result",
            "GET /health": "Health check"
        },
        "example": {
            "request": {
                "planName": "30-Day Python Learning",
                "category": "learning",
                "totalDays": 7,
                "detailPrompt": "I'm a beginner wanting to learn Python basics",
                "fastMode": True
            }
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Async Planner Generation API - Local Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST http://localhost:5001/startPlannerGeneration")
    print("  GET  http://localhost:5001/getPlannerStatus?jobId=xxx")
    print("  GET  http://localhost:5001/getPlannerResult?jobId=xxx")
    print("  GET  http://localhost:5001/health")
    print("\nFrontend:")
    print("  Open frontend_example.html and set API_BASE='http://localhost:5001'")
    print("\n" + "=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
