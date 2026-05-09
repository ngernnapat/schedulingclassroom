"""
Async Planner Generation with Background Jobs

This module provides a better UX for long-running planner generation by:
1. Starting generation as a background job
2. Returning a job ID immediately
3. Allowing clients to poll for progress
4. Streaming partial results as they're ready

Usage Flow:
1. POST /startPlannerGeneration → Returns {jobId, estimatedTime}
2. GET /getPlannerStatus?jobId=xxx → Returns {status, progress, partialResult?}
3. GET /getPlannerResult?jobId=xxx → Returns full result when complete
"""

import os
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from firebase_functions import https_fn
from firebase_admin import firestore

# Import from main module
from generate_planner_content import (
    GeneratePlannerRequest,
    PlannerContent,
    ChatWrapper,
    ChatWrapperConfig,
    PlannerGenerationError,
    ValidationError
)

# Job storage (use Firestore in production)
_job_store: Dict[str, Dict[str, Any]] = {}
_job_lock = threading.Lock()

# Firestore client (lazy init)
_firestore_client = None

def get_firestore_client():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.client()
    return _firestore_client


@dataclass
class JobStatus:
    """Status of a planner generation job"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    progress_message: str
    estimated_seconds_remaining: Optional[int]
    created_at: str
    updated_at: str
    request: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress stages for frontend
    current_stage: str = "initializing"  # "initializing", "extracting_context", "generating_days", "finalizing"
    stages_completed: int = 0
    total_stages: int = 4


def estimate_generation_time(total_days: int, fast_mode: bool) -> int:
    """Estimate generation time based on plan parameters"""
    base_time = 30 if fast_mode else 60
    per_day_time = 2 if fast_mode else 4
    return base_time + (total_days * per_day_time)


def create_job(request: GeneratePlannerRequest) -> JobStatus:
    """Create a new generation job"""
    job_id = f"plan_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    estimated_time = estimate_generation_time(request.totalDays, request.fastMode)
    
    job = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        progress_message="Job created, waiting to start...",
        estimated_seconds_remaining=estimated_time,
        created_at=now,
        updated_at=now,
        request=request.model_dump(),
        current_stage="initializing",
        stages_completed=0,
        total_stages=4
    )
    
    # Store in memory (use Firestore for production)
    with _job_lock:
        _job_store[job_id] = asdict(job)
    
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
            elapsed = (datetime.utcnow() - datetime.fromisoformat(job["created_at"])).total_seconds()
            if progress > 0:
                total_estimated = (elapsed / progress) * 100
                remaining = max(0, int(total_estimated - elapsed))
                job["estimated_seconds_remaining"] = remaining


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


def fail_job(job_id: str, error: str):
    """Mark job as failed with error"""
    with _job_lock:
        if job_id in _job_store:
            job = _job_store[job_id]
            job["status"] = "failed"
            job["progress_message"] = "Generation failed"
            job["updated_at"] = datetime.utcnow().isoformat()
            job["error"] = error


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status"""
    with _job_lock:
        return _job_store.get(job_id)


def run_generation_job(job_id: str, request: GeneratePlannerRequest):
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
        update_job_progress(job_id, 25, f"Generating {request.totalDays}-day plan...", "generating_days", 2)
        
        # Generate content
        content = chat.generate(request)
        
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
# HTTP Endpoints
# =========================

def _cors_headers(origin: Optional[str]) -> Dict[str, str]:
    allow_origin = origin or "*"
    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "3600"
    }


@https_fn.on_request(memory=512, timeout_sec=30)
def start_planner_generation(req: https_fn.Request) -> https_fn.Response:
    """
    Start a planner generation job.
    Returns immediately with a job ID for polling.
    
    POST /startPlannerGeneration
    Body: Same as generate_planner_content
    
    Response: {
        "jobId": "plan_abc123",
        "status": "pending",
        "estimatedSeconds": 120,
        "pollUrl": "/getPlannerStatus?jobId=plan_abc123",
        "message": "Your planner is being generated. Poll for status updates."
    }
    """
    origin = req.headers.get("Origin")
    
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=_cors_headers(origin))
    
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Use POST method"}),
            status=405,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    try:
        payload = req.get_json(silent=True) or {}
        parsed_request = GeneratePlannerRequest(**payload)
        
        # Create job
        job = create_job(parsed_request)
        
        # Start background thread for generation
        thread = threading.Thread(
            target=run_generation_job,
            args=(job.job_id, parsed_request),
            daemon=True
        )
        thread.start()
        
        # Return immediately
        response = {
            "jobId": job.job_id,
            "status": "pending",
            "estimatedSeconds": job.estimated_seconds_remaining,
            "pollUrl": f"/getPlannerStatus?jobId={job.job_id}",
            "message": "Your planner is being generated. Poll for status updates.",
            "tips": [
                "Generation typically takes 1-3 minutes",
                "You can close this and come back later",
                "Your plan will be saved once complete"
            ]
        }
        
        return https_fn.Response(
            json.dumps(response),
            status=202,  # Accepted
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
        
    except ValidationError as ve:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()]
        return https_fn.Response(
            json.dumps({"error": "Invalid request", "details": errors}),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({"error": str(e)}),
            status=500,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )


@https_fn.on_request(memory=256, timeout_sec=10)
def get_planner_status(req: https_fn.Request) -> https_fn.Response:
    """
    Get the status of a planner generation job.
    
    GET /getPlannerStatus?jobId=plan_abc123
    
    Response: {
        "jobId": "plan_abc123",
        "status": "processing",  // pending, processing, completed, failed
        "progress": 45,          // 0-100
        "progressMessage": "Generating days 8-14...",
        "currentStage": "generating_days",
        "stagesCompleted": 2,
        "totalStages": 4,
        "estimatedSecondsRemaining": 60,
        "result": null           // Only present when completed
    }
    """
    origin = req.headers.get("Origin")
    
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=_cors_headers(origin))
    
    job_id = req.args.get("jobId")
    if not job_id:
        return https_fn.Response(
            json.dumps({"error": "Missing jobId parameter"}),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    job = get_job(job_id)
    if not job:
        return https_fn.Response(
            json.dumps({"error": "Job not found", "jobId": job_id}),
            status=404,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    # Don't return full result in status - just summary
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
    
    # Include error if failed
    if job["status"] == "failed":
        response["error"] = job["error"]
    
    # Include result summary if completed
    if job["status"] == "completed" and job.get("result"):
        result = job["result"]
        response["resultSummary"] = {
            "planName": result.get("planName"),
            "category": result.get("category"),
            "totalDays": result.get("totalDays"),
            "ready": True
        }
    
    return https_fn.Response(
        json.dumps(response),
        status=200,
        headers={**_cors_headers(origin), "Content-Type": "application/json"}
    )


@https_fn.on_request(memory=256, timeout_sec=10)
def get_planner_result(req: https_fn.Request) -> https_fn.Response:
    """
    Get the full result of a completed planner generation job.
    
    GET /getPlannerResult?jobId=plan_abc123
    
    Response: Full PlannerContent object
    """
    origin = req.headers.get("Origin")
    
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=_cors_headers(origin))
    
    job_id = req.args.get("jobId")
    if not job_id:
        return https_fn.Response(
            json.dumps({"error": "Missing jobId parameter"}),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    job = get_job(job_id)
    if not job:
        return https_fn.Response(
            json.dumps({"error": "Job not found", "jobId": job_id}),
            status=404,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    if job["status"] != "completed":
        return https_fn.Response(
            json.dumps({
                "error": "Job not completed",
                "status": job["status"],
                "progress": job["progress"]
            }),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    
    return https_fn.Response(
        json.dumps(job["result"], ensure_ascii=False),
        status=200,
        headers={**_cors_headers(origin), "Content-Type": "application/json"}
    )
