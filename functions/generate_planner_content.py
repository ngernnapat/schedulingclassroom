import os
import json
import time
import uuid
from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, asdict

from firebase_functions import https_fn
from firebase_admin import initialize_app

from pydantic import BaseModel, Field, ValidationError, conint, constr

# ---- Initialize Firebase Admin (safe if called multiple times) ----
try:
    initialize_app()
except ValueError:
    # Already initialized in warm container
    pass

# ---- OpenAI (Responses API) ----
# pip install openai>=1.40
from openai import OpenAI

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Data Models (Schemas)
# =========================

PlanCategory = Literal["english", "fitness", "travel"]

class TimeStamp(BaseModel):
    seconds: int = Field(..., description="Unix seconds")
    nanoseconds: int = Field(..., ge=0, lt=1_000_000_000, description="0..999,999,999")

class Task(BaseModel):
    id: constr(strip_whitespace=True, min_length=1) = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: constr(strip_whitespace=True, min_length=1)
    done: bool = False
    duration_min: Optional[conint(ge=0, le=600)] = None   # optional per-task duration
    note: Optional[str] = None

class DayPlan(BaseModel):
    id: constr(strip_whitespace=True, min_length=1) = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    dayNumber: conint(ge=1)                               # 1..N
    title: constr(strip_whitespace=True, min_length=1)
    summary: constr(strip_whitespace=True, min_length=1)
    tasks: List[Task] = Field(default_factory=list)
    tips: Optional[str] = None

class PlannerContent(BaseModel):
    planName: constr(strip_whitespace=True, min_length=1)
    category: PlanCategory
    totalDays: conint(ge=1, le=60) = 30
    coverImage: Optional[str] = None
    coverImageUrl: Optional[str] = None
    createdAt: TimeStamp
    days: List[DayPlan]

# -------- Request --------
class GeneratePlannerRequest(BaseModel):
    planName: constr(strip_whitespace=True, min_length=1) = "30-Day Practice"
    category: PlanCategory = "english"
    totalDays: conint(ge=1, le=60) = 30
    detailPrompt: Optional[str] = Field(
        default=None,
        description="User specifics (level, constraints, destinations, equipment, etc.)"
    )
    # Optional knobs:
    minutesPerDay: Optional[conint(ge=10, le=240)] = None
    intensity: Optional[Literal["easy", "moderate", "hard", "periodized"]] = None
    language: Optional[Literal["en", "th"]] = "en"  # output language


# =========================
# Chat Wrapper
# =========================

@dataclass
class ChatWrapperConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    # Guardrails via JSON schema (response_format)
    json_schema: Dict[str, Any] = None

class ChatWrapper:
    """
    Thin wrapper around OpenAI Chat Completions API that:
    - Sets a strong system prompt for behavior
    - Enforces a JSON schema for our PlannerContent
    """
    def __init__(self, config: ChatWrapperConfig):
        self.config = config

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an expert planner-content generator for a lifestyle planner app. "
            "Generate structured daily plans with clear titles, concise summaries, and actionable tasks. "
            "Respect the category logic:\n"
            "- english: progressive skill-building (listening, speaking, reading, writing), spaced review, weekly reflection.\n"
            "- fitness: alternate focus (strength/cardio/mobility), include rest & recovery, safe progressions.\n"
            "- travel: cluster activities sensibly by location, alternate heavy/light days, include logistics & budget tips.\n"
            "Rules:\n"
            "1) Keep each day practical (3-6 tasks). 2) Add brief tips when helpful. 3) Titles are short and motivating.\n"
            "4) Never invent unsafe or extreme advice; prefer safe defaults. 5) Output MUST be valid JSON per schema."
        )

    def generate(self, req: GeneratePlannerRequest) -> PlannerContent:
        now_s = int(time.time())
        payload = {
            "planName": req.planName,
            "category": req.category,
            "totalDays": req.totalDays,
            "minutesPerDay": req.minutesPerDay,
            "intensity": req.intensity,
            "language": req.language,
            "detailPrompt": req.detailPrompt,
            "unix_now": now_s
        }

        # Build the json schema for the exact PlannerContent shape
        schema = self.config.json_schema or {
            "name": "planner_content",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "planName": {"type": "string"},
                    "category": {"type": "string", "enum": ["english", "fitness", "travel"]},
                    "totalDays": {"type": "integer", "minimum": 1, "maximum": 60},
                    "coverImage": {"type": ["string", "null"]},
                    "coverImageUrl": {"type": ["string", "null"]},
                    "createdAt": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "seconds": {"type": "integer"},
                            "nanoseconds": {"type": "integer", "minimum": 0, "maximum": 999999999}
                        },
                        "required": ["seconds", "nanoseconds"]
                    },
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "dayNumber": {"type": "integer", "minimum": 1},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "tasks": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"},
                                            "done": {"type": "boolean"},
                                            "duration_min": {"type": ["integer", "null"], "minimum": 0, "maximum": 600},
                                            "note": {"type": ["string", "null"]},
                                        },
                                        "required": ["id", "text", "done"]
                                    }
                                },
                                "tips": {"type": ["string", "null"]}
                            },
                            "required": ["id", "dayNumber", "title", "summary", "tasks"]
                        }
                    }
                },
                "required": ["planName", "category", "totalDays", "createdAt", "days"]
            }
        }

        # Assistant guidance for per-category specifics (few-shot style brief)
        category_hints = {
            "english": (
                "User goal: English practice. Include mix each week: listening/speaking/reading/writing, "
                "shadowing, spaced repetition of vocab. Weekly reflection day with lighter load."
            ),
            "fitness": (
                "User goal: general fitness. Rotate muscle groups, include cardio and mobility. "
                "At least one full rest day per week; deload or lighter days as needed. Provide safe, scalable tasks."
            ),
            "travel": (
                "User goal: travel planning. Provide themed days with logistics (transport/time windows), "
                "suggest booking/backup options, and a money/time estimate task. Alternate heavy sightseeing and light days."
            )
        }

        # Language requirement (brief)
        lang_note = "Write in Thai." if req.language == "th" else "Write in English."

        # Build the user message
        user_msg = (
            f"{lang_note}\n"
            f"Category: {req.category}\n"
            f"Plan name: {req.planName}\n"
            f"Total days: {req.totalDays}\n"
            f"Minutes per day (optional): {req.minutesPerDay}\n"
            f"Intensity (optional): {req.intensity}\n"
            f"Details from user (optional): {req.detailPrompt}\n\n"
            f"{category_hints[req.category]}\n"
            "Output a JSON object that strictly matches the provided schema. "
            "Use short, punchy titles; concise summaries; and 3–6 actionable tasks per day."
        )

        # Response format with JSON schema enforcement
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{
                "role": "system",
                "content": self._system_prompt()
            }, {
                "role": "user",
                "content": user_msg
            }],
            response_format={
                "type": "json_schema",
                "json_schema": schema
            }
        )

        # Extract JSON
        raw = response.choices[0].message.content  # SDK provides JSON string when response_format is json_schema
        data = json.loads(raw)

        # Fill in createdAt if model left null, and ensure ids
        seconds = payload["unix_now"]
        data.setdefault("createdAt", {"seconds": seconds, "nanoseconds": 0})
        for i, d in enumerate(data.get("days", []), start=1):
            d.setdefault("id", uuid.uuid4().hex[:8])
            d.setdefault("dayNumber", i)
            for t in d.get("tasks", []):
                t.setdefault("id", uuid.uuid4().hex[:8])
                t.setdefault("done", False)

        # Validate with Pydantic (final gate)
        validated = PlannerContent(**data)
        return validated


# =========================
# HTTP Function
# =========================

chat = ChatWrapper(ChatWrapperConfig())

def _cors_headers(origin: Optional[str]) -> Dict[str, str]:
    # Relaxed CORS; tune for production domains
    allow_origin = origin or "*"
    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "3600"
    }

@https_fn.on_request(memory=1024, max_instances=3)
def generate_planner_content(req: https_fn.Request) -> https_fn.Response:
    origin = req.headers.get("Origin")
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=_cors_headers(origin))

    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Use POST with JSON body."}),
            status=405,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )

    try:
        payload = req.get_json(silent=True) or {}
        parsed = GeneratePlannerRequest(**payload)
        content = chat.generate(parsed)
        body = content.model_dump()
        return https_fn.Response(
            json.dumps(body, ensure_ascii=False),
            status=200,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except ValidationError as ve:
        return https_fn.Response(
            ve.json(),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except Exception as e:
        # Avoid leaking internals; log fully in real deployments
        err = {"error": "Generation failed", "detail": str(e)}
        return https_fn.Response(
            json.dumps(err),
            status=500,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )