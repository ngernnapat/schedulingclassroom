import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import ValidationError
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# import the same models + chat wrapper from main.py
from main import GeneratePlannerRequest, chat

app = FastAPI(title="Planner Content Generator (Local)")

# Ensure API key is present
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before running local_app.py")

@app.post("/generate_planner_content")
async def generate_planner_content(req: Request):
    try:
        payload = await req.json()
        print(f"Received payload: {payload}")
        
        parsed = GeneratePlannerRequest(**payload)
        print(f"Parsed request: {parsed}")
        
        content = chat.generate(parsed)
        print(f"Generated content: {content.planName} with {len(content.days)} days")
        
        return content.model_dump()
    except ValidationError as ve:
        print(f"Validation error: {ve.errors()}")
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)