import os
import json
import time
import uuid
from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, asdict

from firebase_functions import https_fn
from firebase_admin import initialize_app

from pydantic import BaseModel, Field, ValidationError, conint, constr, model_validator

# ---- Initialize Firebase Admin (safe if called multiple times) ----
try:
    initialize_app()
except ValueError:
    # Already initialized in warm container
    pass

# ---- OpenAI (Responses API) ----
# pip install openai>=1.40
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Data Models (Schemas)
# =========================

PlanCategory = Literal["learning", "exercise", "travel", "finance", "health", "personal_development", "other"]

class TimeStamp(BaseModel):
    seconds: int = Field(..., description="Unix seconds")
    nanoseconds: int = Field(..., ge=0, lt=1_000_000_000, description="0..999,999,999")

class Task(BaseModel):
    id: constr(strip_whitespace=True, min_length=1) = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: constr(strip_whitespace=True, min_length=1)
    done: bool = False
    duration_min: Optional[conint(ge=0, le=600)] = None   # optional per-task duration
    note: Optional[str] = None
    link: Optional[constr(strip_whitespace=True, min_length=1)] = Field(None, description="Optional helpful link or resource for this task")

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
    totalDays: conint(ge=1, le=90) = 30
    minutesPerDay: Optional[conint(ge=10, le=480)] = None
    coverImage: Optional[str] = None
    coverImageUrl: Optional[str] = None
    createdAt: TimeStamp
    days: List[DayPlan]
    warning: Optional[str] = None  # For day count mismatch warnings

# -------- Request --------
class GeneratePlannerRequest(BaseModel):
    """Request model for generating planner content with comprehensive validation."""
    
    planName: constr(strip_whitespace=True, min_length=1, max_length=100) = Field(
        default="30-Day Practice",
        description="Name of the plan to generate (1-100 characters)"
    )
    
    category: PlanCategory = Field(
        default="learning",
        description="Type of planner content to generate"
    )
    
    totalDays: conint(ge=1, le=90) = Field(
        default=30,
        description="Number of days in the plan (1-90)"
    )
    
    detailPrompt: Optional[constr(strip_whitespace=True, max_length=1000)] = Field(
        default=None,
        description="User specifics (level, constraints, destinations, equipment, etc.) - max 1000 characters"
    )
    
    # Optional configuration knobs:
    minutesPerDay: Optional[conint(ge=10, le=480)] = Field(
        default=None,
        description="Daily time allocation in minutes (10-480, i.e., 10 min to 8 hours)"
    )
    
    intensity: Optional[Literal["easy", "moderate", "hard", "periodized"]] = Field(
        default=None,
        description="Difficulty/intensity level of the plan"
    )
    
    language: Literal["en", "th"] = Field(
        default="en",
        description="Output language for the generated content"
    )
    
    # Additional optional fields for better customization
    startDate: Optional[str] = Field(
        default=None,
        description="Preferred start date (YYYY-MM-DD format) for scheduling context"
    )
    
    timeOfDay: Optional[Literal["morning", "afternoon", "evening", "flexible"]] = Field(
        default=None,
        description="Preferred time of day for activities"
    )
    
    @model_validator(mode='after')
    def validate_plan_consistency(self) -> 'GeneratePlannerRequest':
        """Validate business logic constraints with user-friendly suggestions."""
        # Validate minutesPerDay makes sense for the category - use warnings instead of errors
        if self.minutesPerDay and self.category == "exercise":
            if self.minutesPerDay < 15:
                # Auto-adjust to minimum safe duration instead of raising error
                print(f"Warning: Exercise plans should be at least 15 minutes for safety. Adjusting from {self.minutesPerDay} to 15 minutes.")
                self.minutesPerDay = 15
            if self.minutesPerDay > 480:
                # Auto-adjust to maximum safe duration instead of raising error
                print(f"Warning: Exercise plans should not exceed 8 hours for safety. Adjusting from {self.minutesPerDay} to 480 minutes.")
                self.minutesPerDay = 480
        
        # Validate totalDays vs minutesPerDay for reasonable workload - use warnings instead of errors
        if self.minutesPerDay and self.totalDays:
            total_hours = (self.minutesPerDay * self.totalDays) / 60
            if total_hours > 200:  # More than 200 total hours seems excessive
                # Suggest reducing intensity instead of failing
                print(f"Warning: Plan would require {total_hours:.1f} total hours, which may be intensive. Consider reducing daily time or total days.")
                # Auto-adjust to more reasonable duration
                suggested_minutes = int((200 * 60) / self.totalDays)
                if suggested_minutes < self.minutesPerDay:
                    print(f"Auto-adjusting daily time from {self.minutesPerDay} to {suggested_minutes} minutes for better balance.")
                    self.minutesPerDay = suggested_minutes
        
        # Validate startDate format if provided - be more flexible with date formats
        if self.startDate:
            try:
                from datetime import datetime
                # Try multiple common date formats
                date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(self.startDate, fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    print(f"Warning: Date format '{self.startDate}' not recognized. Please use YYYY-MM-DD format. Continuing without date validation.")
                else:
                    # Normalize to YYYY-MM-DD format
                    self.startDate = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"Warning: Could not parse start date '{self.startDate}': {e}. Continuing without date validation.")
        
        return self


# =========================
# Chat Wrapper
# =========================

class PlannerGenerationError(Exception):
    """Custom exception for planner generation errors with user-friendly messages"""
    def __init__(self, message: str, user_message: str):
        self.message = message  # Technical message for logging
        self.user_message = user_message  # User-friendly message
        super().__init__(message)

@dataclass
class ChatWrapperConfig:
    model: str = "gpt-4o"
    temperature: float = 0.7
    chunk_size: int = 30  # Days per chunk for large plans
    max_chunks: int = 3   # Maximum number of chunks (90 days max)
    # Guardrails via JSON schema (response_format)
    json_schema: Dict[str, Any] = None

class ChatWrapper:
    """
    Enhanced wrapper around OpenAI Chat Completions API that:
    - Sets a strong system prompt for behavior
    - Enforces a JSON schema for our PlannerContent
    - Supports chunked generation for large plans (60-90 days)
    - Includes retry mechanisms and error handling
    - Handles rate limiting and exponential backoff
    """
    def __init__(self, config: ChatWrapperConfig):
        self.config = config

    def _parse_json_response(self, raw_response: str) -> dict:
        """Parse JSON response with fallback mechanisms for common issues"""
        # First, try direct parsing
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object boundaries
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = raw_response[start_idx:end_idx + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If all else fails, raise the original error
        raise json.JSONDecodeError("Could not parse JSON from response", raw_response, 0)

    def _validate_task_link(self, link: str, category: str) -> bool:
        """Validate that a task link meets quality and source requirements"""
        if not link or not isinstance(link, str):
            return False
        
        link = link.strip()
        
        # Basic format validation
        if not link.startswith(('http://', 'https://')):
            return False
        
        # Check for placeholder or invalid URLs
        invalid_patterns = [
            'example.com', 'placeholder', 'test.com', 'dummy.com',
            'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl',
            'localhost', '127.0.0.1', '0.0.0.0'
        ]
        
        if any(pattern in link.lower() for pattern in invalid_patterns):
            return False
        
        # Category-specific domain validation
        approved_domains = {
            "learning": [
                "coursera.org", "khanacademy.org", "udemy.com", "edx.org", 
                "codecademy.com", "skillshare.com", "linkedin.com", "youtube.com",
                "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu"
            ],
            "exercise": [
                "nike.com", "fitnessblender.com", "darebee.com", "myfitnesspal.com",
                "bodybuilding.com", "menshealth.com", "womenshealthmag.com", 
                "acefitness.org", "verywellfit.com", "youtube.com"
            ],
            "travel": [
                "tripadvisor.com", "rome2rio.com", "booking.com", "wikitravel.org",
                "lonelyplanet.com", "nationalgeographic.com", "travelandleisure.com",
                "cntraveler.com", "airbnb.com", "expedia.com"
            ],
            "finance": [
                "investopedia.com", "nerdwallet.com", "bankrate.com", "mint.com",
                "yahoo.com", "marketwatch.com", "cnbc.com", "forbes.com",
                "money.cnn.com", "fidelity.com", "vanguard.com"
            ],
            "health": [
                "mayoclinic.org", "healthline.com", "webmd.com", "medlineplus.gov",
                "cdc.gov", "who.int", "harvard.edu", "clevelandclinic.org",
                "hopkinsmedicine.org", "nih.gov"
            ],
            "personal_development": [
                "mindtools.com", "ted.com", "psychologytoday.com", "hbr.org",
                "lifehack.org", "zenhabits.net", "jamesclear.com", "charlesduhigg.com",
                "gretchenrubin.com", "youtube.com"
            ],
            "other": [
                "wikipedia.org", "youtube.com", "reddit.com", "medium.com",
                "quora.com", "stackoverflow.com", "github.com"
            ]
        }
        
        # Check if link contains an approved domain for the category
        category_domains = approved_domains.get(category, approved_domains["other"])
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(link)
            domain = parsed.netloc.lower()
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check if domain is in approved list
            return any(approved_domain in domain for approved_domain in category_domains)
            
        except Exception:
            return False

    def _check_duplicate_links(self, days: List[Dict]) -> List[str]:
        """Check for duplicate links within the plan and return list of duplicates"""
        all_links = []
        duplicates = []
        
        for day in days:
            for task in day.get("tasks", []):
                link = task.get("link")
                if link and isinstance(link, str):
                    link = link.strip()
                    if link:
                        if link in all_links:
                            duplicates.append(link)
                        else:
                            all_links.append(link)
        
        return duplicates

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an expert planner-content generator for a lifestyle planner app. "
            "Generate structured daily plans with clear titles, concise summaries, and actionable tasks. "
            "Respect the category logic:\n"
            "- learning: progressive skill-building with variety, spaced repetition, and weekly reflection.\n"
            "- exercise: alternate focus (strength/cardio/flexibility/mobility), include rest & recovery, safe progressions.\n"
            "- travel: cluster activities by location, alternate heavy/light days, include logistics & budget tips.\n"
            "- finance: budgeting, expense tracking, saving strategies, progressive financial literacy.\n"
            "- health: holistic wellness covering nutrition, sleep, stress management, sustainable habits.\n"
            "- personal_development: goal setting, productivity, mindfulness, self-reflection practices.\n"
            "- other: flexible structure based on user's specific needs and goals.\n"
            "Rules:\n"
            "1) Keep each day practical (3-6 tasks). 2) Add brief tips when helpful. 3) Titles are short and motivating.\n"
            "4) Never invent unsafe or extreme advice; prefer safe defaults.\n"
            "5) CRITICAL: Output MUST be valid JSON matching the exact schema provided.\n"
            "6) Include ALL required fields: planName, category, totalDays, createdAt, days.\n"
            "7) The 'days' field MUST be an array with exactly the requested number of days.\n"
            "8) TIME ALLOCATION: If minutesPerDay is specified, you MUST ensure that the sum of all task durations (duration_min) for each day equals exactly the specified minutesPerDay. Each task must have a duration_min value when minutesPerDay is provided."
            "9) You MUST generate exactly the requested number of days - no more, no less. The number of days in the generated plan MUST match totalDays."
            "10) MANDATORY LINK REQUIREMENT: EVERY single task MUST include a meaningful, helpful link or resource in the 'link' field. This field is ABSOLUTELY REQUIRED and cannot be empty, null, or contain placeholder text.\n"
            "   LINK VALIDATION RULES:\n"
            "   ✓ Must be a real, accessible URL from trusted, reputable sources\n"
            "   ✓ Must point to a specific page/article/video that directly helps with the task\n"
            "   ✓ Each task requires a unique link - no duplicates within the plan\n"
            "   ✓ Links must be current and from authoritative sources\n"
            "   ✓ No shortened URLs, placeholder links, or generic homepage links\n"
            "   ✓ Must include the full URL with https:// protocol\n"
            "   \n"
            "   APPROVED SOURCE DOMAINS BY CATEGORY:\n"
            "   📚 Learning: coursera.org, khanacademy.org, udemy.com, edx.org, codecademy.com, skillshare.com, linkedin.com/learning, youtube.com/education, youtube.com, mit.edu, stanford.edu\n"
            "   💪 Exercise: nike.com/training, fitnessblender.com, darebee.com, myfitnesspal.com, bodybuilding.com, menshealth.com, womenshealthmag.com, acefitness.org, verywellfit.com, youtube.com\n"
            "   ✈️ Travel: tripadvisor.com, rome2rio.com, booking.com, wikitravel.org, lonelyplanet.com, nationalgeographic.com/travel, travelandleisure.com, cntraveler.com, youtube.com\n"
            "   💰 Finance: investopedia.com, nerdwallet.com, bankrate.com, mint.com, yahoo.com/finance, marketwatch.com, cnbc.com, forbes.com/finance, money.cnn.com, youtube.com\n"
            "   🏥 Health: mayoclinic.org, healthline.com, webmd.com, medlineplus.gov, cdc.gov, who.int, harvard.edu/health, clevelandclinic.org, hopkinsmedicine.org, youtube.com\n"
            "   🧠 Personal Development: mindtools.com, ted.com, psychologytoday.com, hbr.org, lifehack.org, zenhabits.net, jamesclear.com, charlesduhigg.com, gretchenrubin.com, youtube.com\n"
            "   \n"
            "   LINK QUALITY EXAMPLES:\n"
            "   ✅ GOOD: 'https://www.coursera.org/learn/python-programming' (specific course)\n"
            "   ✅ GOOD: 'https://www.nike.com/training/guides/beginner-workout-plan' (specific guide)\n"
            "   ❌ BAD: 'https://www.coursera.org' (homepage only)\n"
            "   ❌ BAD: 'https://bit.ly/abc123' (shortened URL)\n"
            "   ❌ BAD: 'https://example.com' (placeholder)\n"
           
        )

    def generate_chunked(self, req: GeneratePlannerRequest) -> PlannerContent:
        """Generate planner content using chunked approach for large plans (>30 days)"""
        if req.totalDays <= self.config.chunk_size:
            # Use single generation for plans <= chunk_size days
            return self.generate_single(req)
        
        # Validate maximum days
        max_days = self.config.chunk_size * self.config.max_chunks
        if req.totalDays > max_days:
            raise PlannerGenerationError(
                f"Plan too large: {req.totalDays} days exceeds maximum of {max_days}",
                f"Plans cannot exceed {max_days} days. Please reduce the number of days and try again."
            )
        
        # For plans > chunk_size days, use chunked generation
        chunk_size = self.config.chunk_size
        all_days = []
        now_s = int(time.time())
        total_chunks = (req.totalDays + chunk_size - 1) // chunk_size
        
        # Generate chunks
        for chunk_idx, chunk_start in enumerate(range(1, req.totalDays + 1, chunk_size), 1):
            chunk_end = min(chunk_start + chunk_size - 1, req.totalDays)
            chunk_days = chunk_end - chunk_start + 1
            
            # Retry mechanism for chunk generation
            max_retries = 2
            chunk_content = None
            
            for retry in range(max_retries + 1):
                try:
                    # Create a modified request for this chunk
                    progress_context = f" (This is chunk {chunk_idx}/{total_chunks}, days {chunk_start}-{chunk_end} of a {req.totalDays}-day plan. Build upon previous progress and maintain consistency.)"
                    chunk_req = GeneratePlannerRequest(
                        planName=req.planName,
                        category=req.category,
                        totalDays=chunk_days,
                        detailPrompt=f"{req.detailPrompt or ''}{progress_context}",
                        minutesPerDay=req.minutesPerDay,
                        intensity=req.intensity,
                        language=req.language,
                        startDate=req.startDate,
                        timeOfDay=req.timeOfDay
                    )
                    
                    # Generate this chunk
                    chunk_content = self.generate_single(chunk_req)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry == max_retries:
                        # Final retry failed
                        raise PlannerGenerationError(
                            f"Failed to generate chunk {chunk_idx}/{total_chunks} (days {chunk_start}-{chunk_end}) after {max_retries + 1} attempts: {str(e)}",
                            f"Could not generate the complete plan. Failed at chunk {chunk_idx} of {total_chunks}. Please try again with fewer days or simpler requirements."
                        )
                    else:
                        # Wait before retry
                        time.sleep(2 ** retry)  # Exponential backoff
            
            # Adjust day numbers and add to all_days
            for day in chunk_content.days:
                day.dayNumber = chunk_start + day.dayNumber - 1
                all_days.append(day)
            
            # Add small delay between chunks to avoid rate limits
            if chunk_end < req.totalDays:
                time.sleep(1)
        
        # Create the final content
        final_content = PlannerContent(
            planName=req.planName,
            category=req.category,
            totalDays=req.totalDays,
            minutesPerDay=req.minutesPerDay,
            coverImage=None,
            coverImageUrl=None,
            createdAt={"seconds": now_s, "nanoseconds": 0},
            days=all_days
        )
        
        return final_content

    def generate(self, req: GeneratePlannerRequest) -> PlannerContent:
        """Main generation method with intelligent routing"""
        # Use chunked generation for large plans, single for small ones
        if req.totalDays > self.config.chunk_size:
            return self.generate_chunked(req)
        else:
            return self.generate_single(req)

    def generate_single(self, req: GeneratePlannerRequest) -> PlannerContent:
        now_s = int(time.time())
        payload = {
            "planName": req.planName,
            "category": req.category,
            "totalDays": req.totalDays,
            "minutesPerDay": req.minutesPerDay,
            "intensity": req.intensity,
            "language": req.language,
            "detailPrompt": req.detailPrompt,
            "startDate": req.startDate,
            "timeOfDay": req.timeOfDay,
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
                    "category": {"type": "string", "enum": ["learning", "exercise", "travel", "finance", "health", "personal_development", "other"]},
                    "totalDays": {"type": "integer", "minimum": 1, "maximum": 90},
                    "minutesPerDay": {"type": ["integer", "null"], "minimum": 10, "maximum": 480},
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
                                            "link": {"type": "string"},
                                        },
                                        "required": ["id", "text", "done", "link"]
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
            "learning": (
                "User goal: skill acquisition and knowledge development. Include variety: active practice, "
                "review/repetition, application exercises, and reflection. Build progressively from basics "
                "to advanced concepts. Include weekly review days with lighter cognitive load. "
                "Adapt to user's specified learning domain (language, coding, music, etc.)."
            ),
            "exercise": (
                "User goal: physical fitness and health. Rotate training focus (strength, cardio, flexibility, mobility), "
                "include proper warm-ups and cool-downs. At least one full rest day per week; incorporate deload weeks. "
                "Progressive overload with safe form cues. Scale exercises for different fitness levels. "
                "Balance intensity across the week."
            ),
            "travel": (
                "User goal: trip planning and itinerary. Group activities by geographic proximity and themes. "
                "Include practical logistics (transport modes, time estimates, booking tips). "
                "Provide budget estimates per activity. Alternate high-intensity sightseeing days with relaxed exploration. "
                "Include contingency plans and local cultural tips."
            ),
            "finance": (
                "User goal: financial management and literacy. Cover budgeting, tracking expenses, saving strategies, "
                "investment basics, and debt management. Include actionable review tasks (e.g., audit subscriptions, "
                "track weekly spending). Build from foundational concepts to advanced planning. "
                "Weekly reflection on progress and adjustments."
            ),
            "health": (
                "User goal: holistic wellness and healthy habits. Include nutrition planning, sleep hygiene, "
                "stress management, hydration tracking, and mental health practices. "
                "Provide evidence-based, sustainable habit formation. Balance physical and mental wellness tasks. "
                "Include weekly self-assessment and adjustment days."
            ),
            "personal_development": (
                "User goal: self-improvement and growth. Cover goal setting, productivity habits, mindfulness, "
                "relationship skills, time management, and self-reflection practices. "
                "Include journaling prompts, actionable exercises, and progress tracking. "
                "Build awareness before action; emphasize consistency over intensity."
            ),
            "other": (
                "User goal: custom plan based on user's specific needs. Analyze the user's detailPrompt carefully "
                "and structure the plan with logical progression, variety, and practical actionable tasks. "
                "Include appropriate rest/reflection days and balance intensity throughout the period."
            )
        }

        # Language requirement (brief)
        lang_note = "Write in Thai." if req.language == "th" else "Write in English."

        # Build the user message
        user_msg_parts = [
            lang_note,
            f"Category: {req.category}",
            f"Plan name: {req.planName}",
            f"Total days: {req.totalDays}",
            f"Minutes per day (optional): {req.minutesPerDay}",
            f"Intensity (optional): {req.intensity}",
            f"Details from user (optional): {req.detailPrompt}",
        ]
        
        # Add optional context if provided
        if req.startDate:
            user_msg_parts.append(f"Preferred start date: {req.startDate}")
        if req.timeOfDay:
            user_msg_parts.append(f"Preferred time of day: {req.timeOfDay}")
        
        user_msg_parts.extend([
            "",
            category_hints[req.category],
            "Output a JSON object that strictly matches the provided schema. "
            "Use short, punchy titles; concise summaries; and 3–6 actionable tasks per day. "
        ])
        
        # Add specific guidance for minutesPerDay
        if req.minutesPerDay:
            user_msg_parts.extend([
                "",
                f"IMPORTANT: You must allocate exactly {req.minutesPerDay} minutes per day across all tasks. "
                f"Each task must have a duration_min value, and the sum of all task durations for each day must equal {req.minutesPerDay} minutes. "
                f"Distribute the time logically across tasks (e.g., if you have 4 tasks for 60 minutes, you might allocate 15, 20, 15, 10 minutes respectively)."
            ])
        else:
            user_msg_parts.append("Task durations are optional when minutesPerDay is not specified.")
        
        user_msg = "\n".join(user_msg_parts)

        # Response format with JSON schema enforcement
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
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
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries:
                    # Final attempt failed, handle the error
                    error_str = str(e).lower()
                    if "rate_limit" in error_str or "rate limit" in error_str:
                        raise PlannerGenerationError(
                            f"OpenAI rate limit: {e}",
                            "We've reached our service limit. Please try again in a few minutes."
                        )
                    elif "timeout" in error_str:
                        raise PlannerGenerationError(
                            f"OpenAI timeout: {e}",
                            "The request took too long. Please try with fewer days or simpler requirements."
                        )
                    elif "api_key" in error_str or "authentication" in error_str:
                        raise PlannerGenerationError(
                            f"OpenAI API key error: {e}",
                            "Service configuration error. Please contact support."
                        )
                    else:
                        raise PlannerGenerationError(
                            f"OpenAI API error: {e}",
                            "We're having trouble connecting to the AI service. Please try again in a moment."
                        )
                else:
                    # Wait before retry
                    time.sleep(1)

        # Extract JSON
        try:
            if not response.choices or not response.choices[0].message.content:
                raise PlannerGenerationError(
                    "Empty response from OpenAI",
                    "The AI service returned an empty response. Please try again."
                )
            
            raw = response.choices[0].message.content
            print(f"DEBUG: Raw AI response: {raw[:500]}...")  # Log first 500 chars
            
            # Try to clean and parse the JSON response
            data = self._parse_json_response(raw)
            
            if not isinstance(data, dict):
                raise PlannerGenerationError(
                    f"Invalid response type: {type(data)}",
                    "The AI service returned an unexpected format. Please try again."
                )
                
            print(f"DEBUG: Parsed data keys: {list(data.keys())}")  # Log available keys
                
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")
            print(f"DEBUG: Raw response that failed to parse: {raw}")
            raise PlannerGenerationError(
                f"JSON decode error from OpenAI response: {e}",
                "The AI service returned malformed data. Please try again."
            )
        except Exception as e:
            print(f"DEBUG: Unexpected error parsing response: {e}")
            print(f"DEBUG: Raw response: {raw}")
            raise PlannerGenerationError(
                f"Error processing AI response: {e}",
                "The AI service returned unexpected data. Please try again."
            )

        # Fill in createdAt if model left null, and ensure ids
        try:
            seconds = payload["unix_now"]
            data.setdefault("createdAt", {"seconds": seconds, "nanoseconds": 0})
            
            # Ensure minutesPerDay is included in response from request
            if "minutesPerDay" not in data and req.minutesPerDay is not None:
                data["minutesPerDay"] = req.minutesPerDay
            
            if "days" not in data or not isinstance(data.get("days"), list):
                available_keys = list(data.keys()) if isinstance(data, dict) else "not a dict"
                raise PlannerGenerationError(
                    f"Missing or invalid 'days' field in response. Available keys: {available_keys}",
                    "The generated plan is missing daily schedules. Please try again."
                )
            
            current_days = len(data.get("days", []))
            if current_days != req.totalDays:
                # Handle day count mismatch gracefully
                warning_message = None
                if current_days < req.totalDays:
                    # Use available days but add a warning
                    warning_message = f"Generated only {current_days} days instead of the requested {req.totalDays}. Using available days."
                    print(f"Warning: {warning_message}")
                    # Keep all available days
                    data["days"] = data["days"][:current_days]
                else:
                    # Trim extra days if more than requested
                    warning_message = f"Generated {current_days} days instead of the requested {req.totalDays}. Trimming to requested amount."
                    print(f"Warning: {warning_message}")
                    data["days"] = data["days"][:req.totalDays]
                
                # Update the totalDays in the data to reflect actual days
                data["totalDays"] = len(data["days"])
                
                # Add warning to the response data if it exists
                if warning_message:
                    data["warning"] = warning_message
            
            # Check for duplicate links across the entire plan
            duplicate_links = self._check_duplicate_links(data.get("days", []))
            if duplicate_links:
                print(f"Warning: Found duplicate links in the plan: {duplicate_links}")
                # Add warning about duplicates but don't fail the generation
                if "warning" in data:
                    data["warning"] += f" Note: Some tasks share the same resource links."
                else:
                    data["warning"] = "Some tasks share the same resource links."
            
            for i, d in enumerate(data.get("days", []), start=1):
                if not isinstance(d, dict):
                    raise PlannerGenerationError(
                        f"Invalid day format at index {i}",
                        "The generated plan has invalid day data. Please try again."
                    )
                d.setdefault("id", uuid.uuid4().hex[:8])
                d.setdefault("dayNumber", i)
                
                if "tasks" not in d or not isinstance(d.get("tasks"), list):
                    raise PlannerGenerationError(
                        f"Missing or invalid tasks for day {i}",
                        f"Day {i} is missing task information. Please try again."
                    )
                
                if len(d.get("tasks", [])) == 0:
                    raise PlannerGenerationError(
                        f"No tasks generated for day {i}",
                        f"Day {i} has no tasks. Please try again."
                    )
                
                for t in d.get("tasks", []):
                    if not isinstance(t, dict):
                        raise PlannerGenerationError(
                            f"Invalid task format on day {i}",
                            f"Day {i} has invalid task data. Please try again."
                        )
                    t.setdefault("id", uuid.uuid4().hex[:8])
                    t.setdefault("done", False)
                    
                    # Validate link field - if invalid, set to None
                    link = t.get("link", "")
                    if not link or not isinstance(link, str):
                        # If link is missing or invalid, set to None
                        t["link"] = None
                    else:
                        # Validate link format and quality
                        link = link.strip()
                        if not self._validate_task_link(link, req.category):
                            # If link doesn't meet quality requirements, set to None
                            t["link"] = None
                        else:
                            t["link"] = link
                
                # Validate minutesPerDay constraint if specified - be more flexible
                if req.minutesPerDay:
                    total_duration = 0
                    tasks_without_duration = []
                    
                    for j, task in enumerate(d.get("tasks", [])):
                        if task.get("duration_min") is None:
                            tasks_without_duration.append(j + 1)
                        else:
                            total_duration += task["duration_min"]
                    
                    # Auto-assign durations to tasks that don't have them
                    if tasks_without_duration:
                        remaining_minutes = req.minutesPerDay - total_duration
                        tasks_needing_duration = len(tasks_without_duration)
                        if tasks_needing_duration > 0 and remaining_minutes > 0:
                            # Distribute remaining time evenly among tasks without duration
                            avg_duration = remaining_minutes // tasks_needing_duration
                            remainder = remaining_minutes % tasks_needing_duration
                            
                            for idx, task_idx in enumerate(tasks_without_duration):
                                duration = avg_duration + (1 if idx < remainder else 0)
                                d["tasks"][task_idx - 1]["duration_min"] = max(1, duration)  # At least 1 minute
                            
                            print(f"Warning: Day {i} had tasks without duration. Auto-assigned durations to complete {req.minutesPerDay} minutes.")
                            # Recalculate total
                            total_duration = sum(task.get("duration_min", 0) for task in d.get("tasks", []))
                    
                    # Allow some flexibility in duration matching (±5 minutes)
                    duration_diff = abs(total_duration - req.minutesPerDay)
                    if duration_diff > 5:
                        # Auto-adjust to match requested duration
                        if total_duration < req.minutesPerDay:
                            # Add time to the longest task
                            longest_task_idx = max(range(len(d.get("tasks", []))), 
                                                 key=lambda x: d["tasks"][x].get("duration_min", 0))
                            d["tasks"][longest_task_idx]["duration_min"] += (req.minutesPerDay - total_duration)
                        elif total_duration > req.minutesPerDay:
                            # Reduce time from the longest task
                            longest_task_idx = max(range(len(d.get("tasks", []))), 
                                                 key=lambda x: d["tasks"][x].get("duration_min", 0))
                            reduction = min(total_duration - req.minutesPerDay, 
                                          d["tasks"][longest_task_idx].get("duration_min", 1) - 1)
                            d["tasks"][longest_task_idx]["duration_min"] -= reduction
                        
                        print(f"Warning: Day {i} task durations adjusted from {total_duration} to {req.minutesPerDay} minutes for consistency.")
        
        except PlannerGenerationError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise PlannerGenerationError(
                f"Error processing response data: {e}",
                "Failed to process the generated plan. Please try again."
            )

        # Validate with Pydantic (final gate)
        try:
            validated = PlannerContent(**data)
            return validated
        except ValidationError as ve:
            # Format validation errors
            errors = "; ".join([f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in ve.errors()])
            raise PlannerGenerationError(
                f"Pydantic validation error: {errors}",
                "The generated plan doesn't meet quality standards. Please try again with different parameters."
            )


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
        # Format validation errors in a user-friendly way
        errors = []
        for error in ve.errors():
            field = " → ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")
        
        err = {
            "error": "Invalid request parameters",
            "message": "Please check the following fields and try again:",
            "details": errors
        }
        return https_fn.Response(
            json.dumps(err, ensure_ascii=False),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except json.JSONDecodeError:
        err = {
            "error": "Invalid JSON",
            "message": "The request body must be valid JSON format."
        }
        return https_fn.Response(
            json.dumps(err),
            status=400,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except PlannerGenerationError as pge:
        # Custom planner generation errors with user-friendly messages
        err = {
            "error": "Generation error",
            "message": pge.user_message
        }
        return https_fn.Response(
            json.dumps(err, ensure_ascii=False),
            status=500,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )
    except Exception as e:
        # Provide user-friendly error message without exposing internals
        error_type = type(e).__name__
        
        # Map common errors to user-friendly messages
        if "API" in str(e) or "openai" in str(e).lower():
            err = {
                "error": "AI service unavailable",
                "message": "We're having trouble generating your planner right now. Please try again in a moment."
            }
        elif "timeout" in str(e).lower():
            err = {
                "error": "Request timeout",
                "message": "The request took too long to process. Please try with fewer days or simpler requirements."
            }
        elif "rate" in str(e).lower() or "quota" in str(e).lower():
            err = {
                "error": "Service temporarily unavailable",
                "message": "We've reached our service limit. Please try again in a few minutes."
            }
        else:
            err = {
                "error": "Generation failed",
                "message": "We couldn't generate your planner. Please check your inputs and try again."
            }
        
        # In development, you might want to include more details
        # Uncomment the next line for debugging (but remove in production)
        # err["debug"] = f"{error_type}: {str(e)}"
        
        return https_fn.Response(
            json.dumps(err, ensure_ascii=False),
            status=500,
            headers={**_cors_headers(origin), "Content-Type": "application/json"}
        )