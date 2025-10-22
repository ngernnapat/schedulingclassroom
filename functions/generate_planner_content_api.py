#!/usr/bin/env python3
"""
Local API server for testing the generate_planner_content function.
This provides a FastAPI interface to test the planner generation locally.
"""

import os
import json
import uvicorn
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Check for OpenAI API key before importing
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
    print("Please set your OpenAI API key:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    print()
    print("You can also create a .env file with:")
    print("OPENAI_API_KEY=your-api-key-here")
    print()

# Import the existing planner generation code
try:
    from generate_planner_content import (
        GeneratePlannerRequest, 
        PlannerContent, 
        ChatWrapper, 
        ChatWrapperConfig,
        PlannerGenerationError
    )
except Exception as e:
    if "OPENAI_API_KEY" in str(e):
        print("❌ Cannot start API without OpenAI API key!")
        print("Please set the OPENAI_API_KEY environment variable and try again.")
        exit(1)
    else:
        raise e

# Initialize FastAPI app
app = FastAPI(
    title="Planner Content Generation API",
    description="Local API for testing planner content generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chat wrapper
chat_wrapper = ChatWrapper(ChatWrapperConfig())

# Request/Response models for the API
class PlannerRequest(BaseModel):
    """Request model for the API endpoint"""
    planName: str = Field(default="30-Day Practice", description="Name of the plan")
    category: str = Field(default="learning", description="Type of planner content")
    totalDays: int = Field(default=30, ge=1, le=90, description="Number of days in the plan")
    detailPrompt: Optional[str] = Field(default=None, description="User specifics")
    minutesPerDay: Optional[int] = Field(default=None, ge=10, le=480, description="Daily time allocation")
    intensity: Optional[str] = Field(default=None, description="Difficulty level")
    language: str = Field(default="en", description="Output language")
    startDate: Optional[str] = Field(default=None, description="Preferred start date")
    timeOfDay: Optional[str] = Field(default=None, description="Preferred time of day")

class PlannerResponse(BaseModel):
    """Response model for successful generation"""
    success: bool = True
    data: Dict[str, Any]
    message: str = "Planner generated successfully"

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Planner Content Generation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Generate planner content",
            "GET /health": "Health check",
            "GET /examples": "Get example requests",
            "GET /categories": "Get available categories"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "planner-generation-api"}

@app.get("/categories")
async def get_categories():
    """Get available plan categories"""
    return {
        "categories": [
            {"value": "learning", "description": "Skill acquisition and knowledge development"},
            {"value": "exercise", "description": "Physical fitness and health"},
            {"value": "travel", "description": "Trip planning and itinerary"},
            {"value": "finance", "description": "Financial management and literacy"},
            {"value": "health", "description": "Holistic wellness and healthy habits"},
            {"value": "personal_development", "description": "Self-improvement and growth"},
            {"value": "other", "description": "Custom plan based on specific needs"}
        ]
    }

@app.get("/examples")
async def get_examples():
    """Get example requests for testing"""
    return {
        "examples": [
            {
                "name": "Basic Learning Plan",
                "request": {
                    "planName": "Python Programming Basics",
                    "category": "learning",
                    "totalDays": 7,
                    "detailPrompt": "Beginner Python programming with focus on data structures",
                    "minutesPerDay": 60,
                    "language": "en"
                }
            },
            {
                "name": "Exercise Plan",
                "request": {
                    "planName": "30-Day Fitness Challenge",
                    "category": "exercise",
                    "totalDays": 30,
                    "detailPrompt": "Home workout routine for beginners, no equipment needed",
                    "minutesPerDay": 45,
                    "intensity": "moderate",
                    "language": "en"
                }
            },
            {
                "name": "Travel Plan",
                "request": {
                    "planName": "Thailand Adventure",
                    "category": "travel",
                    "totalDays": 14,
                    "detailPrompt": "Backpacking trip through Bangkok, Chiang Mai, and Phuket",
                    "language": "en"
                }
            },
            {
                "name": "Thai Language Learning",
                "request": {
                    "planName": "เรียนภาษาไทย 30 วัน",
                    "category": "learning",
                    "totalDays": 30,
                    "detailPrompt": "Basic Thai language learning for English speakers",
                    "minutesPerDay": 30,
                    "language": "th"
                }
            }
        ]
    }

@app.post("/generate", response_model=PlannerResponse)
async def generate_planner(request: PlannerRequest):
    """
    Generate planner content based on the provided request.
    
    This endpoint accepts a PlannerRequest and returns generated planner content
    with daily plans, tasks, and tips.
    """
    try:
        # Convert API request to internal request model
        planner_request = GeneratePlannerRequest(
            planName=request.planName,
            category=request.category,
            totalDays=request.totalDays,
            detailPrompt=request.detailPrompt,
            minutesPerDay=request.minutesPerDay,
            intensity=request.intensity,
            language=request.language,
            startDate=request.startDate,
            timeOfDay=request.timeOfDay
        )
        
        # Generate the planner content
        content = chat_wrapper.generate(planner_request)
        
        # Convert to dict for JSON response
        content_dict = content.model_dump()
        
        return PlannerResponse(
            data=content_dict,
            message=f"Successfully generated {content.totalDays}-day {content.category} plan: {content.planName}"
        )
        
    except PlannerGenerationError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Planner generation failed",
                "message": e.user_message,
                "technical_details": e.message
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )

@app.post("/generate-raw")
async def generate_planner_raw(request: Request):
    """
    Generate planner content with raw JSON input/output.
    This endpoint accepts any JSON and passes it directly to the generation function.
    """
    try:
        # Get raw JSON from request
        raw_data = await request.json()
        
        # Create request model from raw data
        planner_request = GeneratePlannerRequest(**raw_data)
        
        # Generate content
        content = chat_wrapper.generate(planner_request)
        
        # Return raw content
        return content.model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid request or generation failed",
                "message": str(e)
            }
        )

@app.get("/test/{category}")
async def test_category(category: str, days: int = 7, minutes: Optional[int] = None):
    """
    Quick test endpoint for generating a simple plan.
    
    Example: GET /test/learning?days=7&minutes=30
    """
    try:
        # Validate category
        valid_categories = ["learning", "exercise", "travel", "finance", "health", "personal_development", "other"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {valid_categories}"
            )
        
        # Create test request
        test_request = GeneratePlannerRequest(
            planName=f"Test {category.title()} Plan",
            category=category,
            totalDays=days,
            detailPrompt=f"Quick test plan for {category} category",
            minutesPerDay=minutes,
            language="en"
        )
        
        # Generate content
        content = chat_wrapper.generate(test_request)
        
        return {
            "success": True,
            "category": category,
            "days": days,
            "minutes_per_day": minutes,
            "content": content.model_dump()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Test generation failed",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    # Run the server
    print("Starting Planner Content Generation API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Examples: http://localhost:8000/examples")
    print()
    
    uvicorn.run(
        "generate_planner_content_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
