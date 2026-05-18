import os
import json
import time
import uuid
import asyncio
import concurrent.futures
from typing import List, Optional, Literal, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict

# Firebase imports - optional for local testing
try:
    from firebase_functions import https_fn
    from firebase_admin import initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    https_fn = None
    print("Note: Firebase modules not available - running in local mode")

from pydantic import BaseModel, Field, ValidationError, conint, constr, model_validator

# ---- Initialize Firebase Admin (safe if called multiple times) ----
if FIREBASE_AVAILABLE:
    try:
        initialize_app()
    except ValueError:
        # Already initialized in warm container
        pass

# ---- OpenAI (Responses API) ----
# pip install openai>=1.40
from openai import OpenAI

# Lazy initialization of OpenAI client to prevent cold start failures
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client with lazy initialization."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# =========================
# Data Models (Schemas)
# =========================

PlanCategory = Literal["learning", "exercise", "travel", "finance", "health", "personal_development", "other"]

_GENERIC_PLAN_NAMES = frozenset({"30-day practice", "30-Day Practice", ""})

_CATEGORY_NAME_LABELS = {
    "en": {
        "learning": "Learning",
        "exercise": "Fitness",
        "travel": "Travel",
        "finance": "Finance",
        "health": "Wellness",
        "personal_development": "Growth",
        "other": "Personal",
    },
    "th": {
        "learning": "การเรียนรู้",
        "exercise": "ฟิตเนส",
        "travel": "ท่องเที่ยว",
        "finance": "การเงิน",
        "health": "สุขภาพ",
        "personal_development": "พัฒนาตัวเอง",
        "other": "ส่วนตัว",
    },
}

_DEFAULT_DETAIL_PROMPTS = {
    "en": {
        "learning": "I want a realistic daily learning plan with practice and review. About 45 minutes per day.",
        "exercise": "I want a balanced workout plan I can stick to. About 45 minutes per session, beginner-friendly, include rest days.",
        "travel": "Plan a day-by-day trip itinerary with about 4 hours of activities per day — sights, meals, local experiences, transit, and rest breaks.",
        "finance": "Build simple daily money habits: track spending, save, and learn basics step by step.",
        "health": "I want sustainable wellness habits for body and mind with realistic daily steps.",
        "personal_development": "Help me build focus, habits, and reflection with small daily actions.",
        "other": "Create a realistic daily plan toward my goal with small, clear tasks.",
    },
    "th": {
        "learning": "ต้องการแผนเรียนรู้รายวันที่ทำได้จริง มีฝึกและทบทวน ประมาณ 45 นาทีต่อวัน",
        "exercise": "ต้องการแผนออกกำลังกายที่ทำต่อเนื่องได้ ประมาณ 45 นาทีต่อครั้ง มือใหม่ มีวันพัก",
        "travel": "วางแผนทริปรายวัน ประมาณ 4 ชั่วโมงต่อวัน — สถานที่ อาหาร ประสบการณ์ท้องถิ่น การเดินทาง และพักผ่อน",
        "finance": "สร้างนิสัยการเงินรายวัน ติดตามรายจ่าย ออม และเรียนรู้ทีละขั้น",
        "health": "ต้องการนิสัยสุขภาพที่ยั่งยืน ทั้งกายและใจ ขั้นตอนเล็กๆ ต่อวัน",
        "personal_development": "สร้างสมาธิ นิสัย และการทบทวน ด้วยการลงมือเล็กๆ ทุกวัน",
        "other": "สร้างแผนรายวันที่ทำได้จริง งานชัดเจน ขั้นเล็กๆ",
    },
}


def suggest_plan_name(category: str, total_days: int, language: str = "en") -> str:
    """Auto title when the client omits or sends a generic plan name."""
    lang = "th" if language == "th" else "en"
    labels = _CATEGORY_NAME_LABELS[lang]
    cat_label = labels.get(category, labels["other"])
    if lang == "th":
        return f"แผน{cat_label} {total_days} วัน"
    return f"{total_days}-Day {cat_label} Plan"


def default_detail_prompt_for_category(category: str, language: str = "en") -> str:
    lang = "th" if language == "th" else "en"
    prompts = _DEFAULT_DETAIL_PROMPTS[lang]
    return prompts.get(category, prompts["other"])


# Typical daily commitment and plan length per category (lean mobile form + API defaults)
_CATEGORY_PLAN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "exercise": {"minutesPerDay": 45, "totalDays": 21, "intensity": "moderate"},
    "learning": {"minutesPerDay": 45, "totalDays": 14, "intensity": "moderate"},
    "travel": {"minutesPerDay": 240, "totalDays": 7, "intensity": "moderate"},
    "finance": {"minutesPerDay": 15, "totalDays": 21, "intensity": "easy"},
    "health": {"minutesPerDay": 25, "totalDays": 21, "intensity": "moderate"},
    "personal_development": {"minutesPerDay": 20, "totalDays": 21, "intensity": "moderate"},
    "other": {"minutesPerDay": 30, "totalDays": 7, "intensity": "moderate"},
}


def default_plan_params_for_category(category: str) -> Dict[str, Any]:
    return dict(_CATEGORY_PLAN_DEFAULTS.get(category, _CATEGORY_PLAN_DEFAULTS["other"]))


def resolve_fast_mode(total_days: int, fast_mode: Optional[bool] = None) -> bool:
    """Sensible default: longer plans use faster model unless client overrides."""
    if fast_mode is not None:
        return bool(fast_mode)
    return total_days > 14

# =========================
# Extracted User Context (from detailPrompt)
# =========================

class UserProfile(BaseModel):
    """Extracted user profile information"""
    experience_level: Optional[Literal["beginner", "intermediate", "advanced", "expert"]] = Field(
        default=None,
        description="User's experience level in this domain"
    )
    age_group: Optional[Literal["teen", "young_adult", "adult", "senior"]] = Field(
        default=None,
        description="User's age group for appropriate content"
    )
    physical_limitations: Optional[List[str]] = Field(
        default=None,
        description="Any physical limitations or health conditions mentioned"
    )
    available_resources: Optional[List[str]] = Field(
        default=None,
        description="Equipment, tools, or resources user has access to"
    )
    location: Optional[str] = Field(
        default=None,
        description="User's location or destination (for travel/local activities)"
    )

class UserGoals(BaseModel):
    """Extracted user goals and motivations"""
    primary_goal: Optional[str] = Field(
        default=None,
        description="Main objective user wants to achieve"
    )
    secondary_goals: Optional[List[str]] = Field(
        default=None,
        description="Additional objectives or sub-goals"
    )
    target_outcome: Optional[str] = Field(
        default=None,
        description="Specific measurable outcome (e.g., 'run 5K', 'pass JLPT N3', 'lose 5kg')"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Any mentioned deadline or target date"
    )
    motivation_type: Optional[Literal["achievement", "health", "social", "mastery", "enjoyment", "necessity"]] = Field(
        default=None,
        description="Primary motivation driving the user"
    )

class UserConstraints(BaseModel):
    """Extracted user constraints and preferences"""
    budget_level: Optional[Literal["minimal", "moderate", "flexible", "unlimited"]] = Field(
        default=None,
        description="Budget constraints mentioned"
    )
    time_constraints: Optional[str] = Field(
        default=None,
        description="Any time limitations or busy periods mentioned"
    )
    excluded_activities: Optional[List[str]] = Field(
        default=None,
        description="Activities user wants to avoid"
    )
    preferred_activities: Optional[List[str]] = Field(
        default=None,
        description="Activities user prefers or enjoys"
    )
    rest_requirements: Optional[str] = Field(
        default=None,
        description="Rest day preferences or recovery needs"
    )

class UserLearningStyle(BaseModel):
    """Extracted learning and engagement preferences"""
    learning_style: Optional[Literal["visual", "reading", "hands_on", "auditory", "mixed"]] = Field(
        default=None,
        description="Preferred way of learning"
    )
    pace_preference: Optional[Literal["slow_steady", "moderate", "intensive", "flexible"]] = Field(
        default=None,
        description="Preferred pace of progression"
    )
    feedback_preference: Optional[Literal["detailed", "brief", "encouraging", "challenging"]] = Field(
        default=None,
        description="How user prefers to receive guidance"
    )

class ExtractedUserContext(BaseModel):
    """Complete extracted context from user's detailPrompt"""
    profile: UserProfile = Field(default_factory=UserProfile)
    goals: UserGoals = Field(default_factory=UserGoals)
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    learning_style: UserLearningStyle = Field(default_factory=UserLearningStyle)
    
    # Category-specific extracted information
    category_specific: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Category-specific details extracted from the prompt"
    )
    
    # Raw interpretation
    key_requirements: Optional[List[str]] = Field(
        default=None,
        description="Key requirements extracted as bullet points"
    )
    tone_preference: Optional[Literal["professional", "casual", "motivational", "educational", "friendly"]] = Field(
        default=None,
        description="Preferred tone for the content"
    )
    special_considerations: Optional[List[str]] = Field(
        default=None,
        description="Any special considerations or notes"
    )

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
    tips: Optional[Union[str, List[str]]] = None
    
    @model_validator(mode='before')
    @classmethod
    def convert_tips_to_string(cls, data: Any) -> Any:
        """Convert tips from list to string if needed"""
        if isinstance(data, dict) and 'tips' in data:
            tips = data['tips']
            if isinstance(tips, list):
                # Join list items with newline or bullet points
                data['tips'] = '\n• '.join(tips) if tips else None
        return data

ProgressCallback = Callable[[Dict[str, Any]], None]


class PlanPhaseOutline(BaseModel):
    phase_name: str
    start_day: int
    end_day: int
    focus: str
    goals: List[str] = Field(default_factory=list)


class PlanOutline(BaseModel):
    """High-level plan structure generated before day-by-day content."""
    overview: str
    difficulty_arc: Optional[str] = None
    key_milestones: List[str] = Field(default_factory=list)
    weekly_focus: List[str] = Field(default_factory=list)
    rest_day_numbers: List[int] = Field(default_factory=list)
    phases: List[PlanPhaseOutline] = Field(default_factory=list)


class PlannerSummary(BaseModel):
    """Summary information about the generated planner"""
    overview: Optional[str] = Field(None, description="Brief overview of what this plan covers and its approach")
    #targetAudience: Optional[str] = Field(None, description="Who this plan is best suited for")
    #expectedOutcomes: Optional[List[str]] = Field(None, description="What the user can expect to achieve")
    keyMilestones: Optional[List[str]] = Field(None, description="Major milestones throughout the plan")
    #difficultyProgression: Optional[str] = Field(None, description="How difficulty changes over the plan (e.g., 'Gradual increase from beginner to intermediate')")
    #totalEstimatedHours: Optional[float] = Field(None, description="Total estimated hours to complete the plan")
    #prerequisites: Optional[List[str]] = Field(None, description="What the user should have/know before starting")
    tipsForSuccess: Optional[List[str]] = Field(None, description="Key tips to maximize success with this plan")
    weeklyFocus: Optional[List[str]] = Field(None, description="Brief focus area for each week")

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
    
    # New summary fields
    summary: Optional[PlannerSummary] = Field(None, description="Summary information about the planner")
    tags: Optional[List[str]] = Field(None, description="Relevant tags for categorization and search")
    difficultyLevel: Optional[str] = Field(None, description="Overall difficulty level (beginner, intermediate, advanced)")
    estimatedCompletionRate: Optional[str] = Field(None, description="Expected completion rate with consistent effort")

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
    
    totalDays: Optional[conint(ge=1, le=90)] = Field(
        default=None,
        description="Number of days in the plan (1-90). Omit for a category-typical length.",
    )
    
    detailPrompt: Optional[constr(strip_whitespace=True, max_length=1000)] = Field(
        default=None,
        description="User specifics (level, constraints, destinations, equipment, etc.) - max 1000 characters"
    )

    # Server-only: full draft JSON for refine_plan (not sent from mobile detailPrompt field)
    refinementContext: Optional[str] = Field(
        default=None,
        max_length=120000,
        description="Internal draft plan JSON for refinement requests",
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
    
    # Performance options (None = auto: faster model for plans longer than 14 days)
    fastMode: Optional[bool] = Field(
        default=None,
        description="Enable fast mode for quicker generation. Omit to auto-select from plan length.",
    )
    
    skipContextExtraction: bool = Field(
        default=False,
        description="Skip the context extraction step for faster generation (less personalized)"
    )

    userId: Optional[str] = Field(
        default=None,
        description="Owner uid — used by Firebase to sync completed jobs into draft plans",
    )
    planId: Optional[str] = Field(
        default=None,
        description="Draft lifestyle-plans document id for background generation",
    )
    
    @model_validator(mode='after')
    def validate_plan_consistency(self) -> 'GeneratePlannerRequest':
        """Validate business logic constraints with user-friendly suggestions."""
        if self.minutesPerDay and self.category == "exercise":
            if self.minutesPerDay < 15:
                print(f"Warning: Exercise plans should be at least 15 minutes for safety. Adjusting from {self.minutesPerDay} to 15 minutes.")
                self.minutesPerDay = 15
            if self.minutesPerDay > 480:
                print(f"Warning: Exercise plans should not exceed 8 hours for safety. Adjusting from {self.minutesPerDay} to 480 minutes.")
                self.minutesPerDay = 480

        if self.minutesPerDay and self.category == "travel":
            if self.minutesPerDay < 60:
                print(
                    f"Warning: Travel itineraries need meaningful daily activity time. "
                    f"Adjusting from {self.minutesPerDay} to 120 minutes."
                )
                self.minutesPerDay = 120

        if self.minutesPerDay and self.totalDays:
            total_hours = (self.minutesPerDay * self.totalDays) / 60
            if total_hours > 200:
                print(f"Warning: Plan would require {total_hours:.1f} total hours, which may be intensive. Consider reducing daily time or total days.")
                suggested_minutes = int((200 * 60) / self.totalDays)
                if suggested_minutes < self.minutesPerDay:
                    print(f"Auto-adjusting daily time from {self.minutesPerDay} to {suggested_minutes} minutes for better balance.")
                    self.minutesPerDay = suggested_minutes

        if self.startDate:
            try:
                from datetime import datetime
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
                    self.startDate = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"Warning: Could not parse start date '{self.startDate}': {e}. Continuing without date validation.")

        # Lean-form defaults (mobile sends category + goal; other fields optional)
        if not self.detailPrompt or not str(self.detailPrompt).strip():
            self.detailPrompt = default_detail_prompt_for_category(self.category, self.language)
            print(f"Applied default detailPrompt for category={self.category}")

        cat_defaults = default_plan_params_for_category(self.category)

        if self.minutesPerDay is None:
            self.minutesPerDay = cat_defaults["minutesPerDay"]

        if not self.intensity:
            self.intensity = cat_defaults["intensity"]

        if self.totalDays is None:
            self.totalDays = cat_defaults["totalDays"]

        name_stripped = (self.planName or "").strip()
        if not name_stripped or name_stripped.lower() in _GENERIC_PLAN_NAMES:
            self.planName = suggest_plan_name(self.category, self.totalDays, self.language)

        self.fastMode = resolve_fast_mode(self.totalDays, self.fastMode)

        return self


class RefinePlannerRequest(BaseModel):
    """Refine an existing draft plan based on user feedback."""

    refinementPrompt: constr(strip_whitespace=True, min_length=1, max_length=800) = Field(
        description="What the user wants changed in the current draft"
    )
    existingContent: Dict[str, Any] = Field(
        description="Current PlannerContent object to refine"
    )
    planName: constr(strip_whitespace=True, min_length=1, max_length=100)
    category: PlanCategory
    totalDays: conint(ge=1, le=90)
    minutesPerDay: Optional[conint(ge=10, le=480)] = None
    intensity: Optional[Literal["easy", "moderate", "hard", "periodized"]] = None
    language: Literal["en", "th"] = "en"
    fastMode: bool = True
    refineDayStart: Optional[conint(ge=1, le=90)] = None
    refineDayEnd: Optional[conint(ge=1, le=90)] = None

    @model_validator(mode='after')
    def validate_refine_range(self) -> 'RefinePlannerRequest':
        if self.refineDayStart is not None and self.refineDayEnd is not None:
            if self.refineDayStart > self.refineDayEnd:
                raise ValueError("refineDayStart must be <= refineDayEnd")
            if self.refineDayEnd > self.totalDays:
                raise ValueError("refineDayEnd exceeds totalDays")
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
class PlanChunk:
    """Represents a logical segment of a larger plan"""
    start_day: int
    end_day: int
    phase_name: str
    focus_area: str
    progression_level: str  # "beginner", "intermediate", "advanced", "mastery"
    key_goals: List[str]
    special_instructions: str

@dataclass
class ChatWrapperConfig:
    model: str = "gpt-5.4"  # High quality model for content generation
    fast_model: str = "gpt-5.4-mini"  # Faster model for fast mode
    extraction_model: str = "gpt-5.4-mini"  # Faster model for context extraction
    temperature: float = 1.0  # Default temperature (some models only support 1.0)
    fast_temperature: float = 1.0  # Default temperature for fast mode
    extraction_temperature: float = 1.0  # Default temperature for extraction
    chunk_size: int = 30  # Days per chunk for large plans
    max_chunks: int = 3   # Maximum number of chunks (90 days max)
    # Guardrails via JSON schema (response_format)
    json_schema: Dict[str, Any] = None


class ContextExtractor:
    """
    Extracts structured user context from free-form detailPrompt using LLM.
    This enables personalized planner generation without requiring users to fill many fields.
    """
    
    def __init__(self, model: str = "gpt-5-mini", temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
    
    def _get_extraction_schema(self, category: str) -> Dict[str, Any]:
        """Get JSON schema for context extraction based on category"""
        
        # Category-specific fields to extract
        category_specific_schema = {
            "learning": {
                "type": "object",
                "properties": {
                    "subject_area": {"type": ["string", "null"], "description": "Main subject being learned"},
                    "current_knowledge": {"type": ["string", "null"], "description": "What user already knows"},
                    "target_skill_level": {"type": ["string", "null"], "description": "Desired proficiency level"},
                    "learning_resources": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Available learning materials"},
                    "exam_or_certification": {"type": ["string", "null"], "description": "Any exam or certification goal"},
                    "practice_focus": {"type": ["string", "null"], "description": "Specific areas to focus practice on"}
                }
            },
            "exercise": {
                "type": "object",
                "properties": {
                    "fitness_goal": {"type": ["string", "null"], "description": "Primary fitness objective"},
                    "current_fitness_level": {"type": ["string", "null"], "description": "Current fitness state"},
                    "workout_types_preferred": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Preferred workout types"},
                    "equipment_available": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Available equipment"},
                    "injuries_or_limitations": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Physical limitations"},
                    "workout_location": {"type": ["string", "null"], "description": "Where workouts will happen"},
                    "target_metrics": {"type": ["string", "null"], "description": "Specific metrics to achieve"}
                }
            },
            "travel": {
                "type": "object",
                "properties": {
                    "destination": {"type": ["string", "null"], "description": "Travel destination(s)"},
                    "trip_type": {"type": ["string", "null"], "description": "Type of trip (adventure, relaxation, cultural, etc.)"},
                    "travel_companions": {"type": ["string", "null"], "description": "Who is traveling (solo, couple, family, group)"},
                    "interests": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Activities and interests"},
                    "accommodation_preference": {"type": ["string", "null"], "description": "Preferred accommodation type"},
                    "transportation_preference": {"type": ["string", "null"], "description": "Preferred transportation"},
                    "must_see_places": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Must-visit locations"}
                }
            },
            "finance": {
                "type": "object",
                "properties": {
                    "financial_goal": {"type": ["string", "null"], "description": "Primary financial objective"},
                    "current_situation": {"type": ["string", "null"], "description": "Current financial state"},
                    "income_level": {"type": ["string", "null"], "description": "General income bracket"},
                    "debt_situation": {"type": ["string", "null"], "description": "Any debt to manage"},
                    "saving_target": {"type": ["string", "null"], "description": "Specific saving goal"},
                    "investment_interest": {"type": ["string", "null"], "description": "Interest in investments"},
                    "financial_knowledge": {"type": ["string", "null"], "description": "Current financial literacy level"}
                }
            },
            "health": {
                "type": "object",
                "properties": {
                    "health_goal": {"type": ["string", "null"], "description": "Primary health objective"},
                    "current_health_status": {"type": ["string", "null"], "description": "Current health state"},
                    "health_conditions": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Existing health conditions"},
                    "diet_preferences": {"type": ["string", "null"], "description": "Dietary preferences or restrictions"},
                    "sleep_patterns": {"type": ["string", "null"], "description": "Current sleep habits"},
                    "stress_level": {"type": ["string", "null"], "description": "Current stress level"},
                    "wellness_focus": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Areas to focus on"}
                }
            },
            "personal_development": {
                "type": "object",
                "properties": {
                    "development_area": {"type": ["string", "null"], "description": "Main area of development"},
                    "current_challenges": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Current challenges faced"},
                    "skills_to_develop": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Skills to build"},
                    "habits_to_build": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Habits to establish"},
                    "habits_to_break": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Habits to eliminate"},
                    "life_area_focus": {"type": ["string", "null"], "description": "Life area to focus on"},
                    "role_models": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Mentioned role models or influences"}
                }
            },
            "other": {
                "type": "object",
                "properties": {
                    "main_topic": {"type": ["string", "null"], "description": "Main topic or activity"},
                    "specific_requirements": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Specific requirements mentioned"},
                    "desired_outcomes": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Desired outcomes"}
                }
            }
        }
        
        return {
            "name": "extracted_user_context",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "experience_level": {"type": ["string", "null"], "enum": ["beginner", "intermediate", "advanced", "expert", None]},
                            "age_group": {"type": ["string", "null"], "enum": ["teen", "young_adult", "adult", "senior", None]},
                            "physical_limitations": {"type": ["array", "null"], "items": {"type": "string"}},
                            "available_resources": {"type": ["array", "null"], "items": {"type": "string"}},
                            "location": {"type": ["string", "null"]}
                        },
                        "required": ["experience_level", "age_group", "physical_limitations", "available_resources", "location"]
                    },
                    "goals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary_goal": {"type": ["string", "null"]},
                            "secondary_goals": {"type": ["array", "null"], "items": {"type": "string"}},
                            "target_outcome": {"type": ["string", "null"]},
                            "deadline": {"type": ["string", "null"]},
                            "motivation_type": {"type": ["string", "null"], "enum": ["achievement", "health", "social", "mastery", "enjoyment", "necessity", None]}
                        },
                        "required": ["primary_goal", "secondary_goals", "target_outcome", "deadline", "motivation_type"]
                    },
                    "constraints": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "budget_level": {"type": ["string", "null"], "enum": ["minimal", "moderate", "flexible", "unlimited", None]},
                            "time_constraints": {"type": ["string", "null"]},
                            "excluded_activities": {"type": ["array", "null"], "items": {"type": "string"}},
                            "preferred_activities": {"type": ["array", "null"], "items": {"type": "string"}},
                            "rest_requirements": {"type": ["string", "null"]}
                        },
                        "required": ["budget_level", "time_constraints", "excluded_activities", "preferred_activities", "rest_requirements"]
                    },
                    "learning_style": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "learning_style": {"type": ["string", "null"], "enum": ["visual", "reading", "hands_on", "auditory", "mixed", None]},
                            "pace_preference": {"type": ["string", "null"], "enum": ["slow_steady", "moderate", "intensive", "flexible", None]},
                            "feedback_preference": {"type": ["string", "null"], "enum": ["detailed", "brief", "encouraging", "challenging", None]}
                        },
                        "required": ["learning_style", "pace_preference", "feedback_preference"]
                    },
                    "category_specific": category_specific_schema.get(category, category_specific_schema["other"]),
                    "key_requirements": {"type": ["array", "null"], "items": {"type": "string"}},
                    "tone_preference": {"type": ["string", "null"], "enum": ["professional", "casual", "motivational", "educational", "friendly", None]},
                    "special_considerations": {"type": ["array", "null"], "items": {"type": "string"}}
                },
                "required": ["profile", "goals", "constraints", "learning_style", "category_specific", "key_requirements", "tone_preference", "special_considerations"]
            }
        }
    
    def _get_extraction_prompt(self, category: str) -> str:
        """Get the system prompt for context extraction"""
        return f"""You are an expert at understanding user requirements and extracting structured information.

Your task is to analyze the user's free-form description and extract relevant information for creating a personalized {category} planner.

EXTRACTION GUIDELINES:
1. Extract ONLY information that is explicitly mentioned or strongly implied
2. Use null for any field where information is not available
3. Be conservative - don't make assumptions beyond what's stated
4. Look for implicit cues (e.g., "I've never done this before" implies beginner level)
5. Extract specific details that will help personalize the plan

CATEGORY-SPECIFIC EXTRACTION ({category.upper()}):
{"- For LEARNING: Look for subject area, current knowledge, exam goals, preferred learning methods" if category == "learning" else ""}
{"- For EXERCISE: Look for fitness goals, current fitness level, available equipment, injuries, preferred workout types" if category == "exercise" else ""}
{"- For TRAVEL: Look for destination, trip type, companions, interests, budget, must-see places" if category == "travel" else ""}
{"- For FINANCE: Look for financial goals, current situation, saving targets, debt, investment interest" if category == "finance" else ""}
{"- For HEALTH: Look for health goals, conditions, diet preferences, sleep issues, stress factors" if category == "health" else ""}
{"- For PERSONAL_DEVELOPMENT: Look for skills to develop, habits to build/break, life areas to focus on" if category == "personal_development" else ""}

INFERENCE RULES:
- "beginner/new/first time/never done" → experience_level: beginner
- "some experience/familiar with/done before" → experience_level: intermediate  
- "experienced/skilled/years of experience" → experience_level: advanced
- "expert/professional/master" → experience_level: expert
- "tight budget/cheap/affordable" → budget_level: minimal
- "no budget limit/money is not an issue" → budget_level: unlimited
- "intense/fast/aggressive" → pace_preference: intensive
- "slow/gradual/easy pace" → pace_preference: slow_steady

Output a JSON object with the extracted information. Use null for any fields where information is not available or cannot be inferred."""
    
    def extract_context(self, detail_prompt: str, category: str, plan_name: str) -> Optional[ExtractedUserContext]:
        """
        Extract structured context from user's free-form detail prompt.
        
        Args:
            detail_prompt: User's free-form description of their requirements
            category: The plan category (learning, exercise, etc.)
            plan_name: Name of the plan for additional context
            
        Returns:
            ExtractedUserContext with structured information, or None if extraction fails
        """
        if not detail_prompt or len(detail_prompt.strip()) < 10:
            # Not enough information to extract
            return None
        
        try:
            user_message = f"""Plan Name: {plan_name}
Category: {category}

User's Description:
{detail_prompt}

Extract all relevant structured information from this description."""

            response = get_openai_client().chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self._get_extraction_prompt(category)},
                    {"role": "user", "content": user_message}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": self._get_extraction_schema(category)
                }
            )
            
            if not response.choices or not response.choices[0].message.content:
                print("Warning: Empty response from context extraction")
                return None
            
            raw_response = response.choices[0].message.content
            data = json.loads(raw_response)
            
            # Convert to ExtractedUserContext
            context = ExtractedUserContext(
                profile=UserProfile(**data.get("profile", {})),
                goals=UserGoals(**data.get("goals", {})),
                constraints=UserConstraints(**data.get("constraints", {})),
                learning_style=UserLearningStyle(**data.get("learning_style", {})),
                category_specific=data.get("category_specific"),
                key_requirements=data.get("key_requirements"),
                tone_preference=data.get("tone_preference"),
                special_considerations=data.get("special_considerations")
            )
            
            print(f"Successfully extracted user context: {len(context.key_requirements or [])} key requirements found")
            return context
            
        except Exception as e:
            print(f"Warning: Context extraction failed: {e}. Proceeding without structured context.")
            return None

class ChatWrapper:
    """
    Enhanced wrapper around OpenAI Chat Completions API that:
    - Sets a strong system prompt for behavior
    - Enforces a JSON schema for our PlannerContent
    - Supports intelligent chunked generation for large plans (60-90 days)
    - Includes retry mechanisms and error handling
    - Handles rate limiting and exponential backoff
    - Analyzes plan requirements and creates logical, progressive segments
    """
    def __init__(self, config: ChatWrapperConfig):
        self.config = config

    def _emit_progress(
        self,
        callback: Optional[ProgressCallback],
        *,
        progress: int,
        progress_message: str,
        current_stage: str,
        stages_completed: Optional[int] = None,
    ) -> None:
        if not callback:
            return
        payload: Dict[str, Any] = {
            "progress": progress,
            "progress_message": progress_message,
            "current_stage": current_stage,
        }
        if stages_completed is not None:
            payload["stages_completed"] = stages_completed
        callback(payload)

    def _outline_to_prompt_section(self, outline: Optional[PlanOutline]) -> str:
        if not outline:
            return ""
        lines = [
            "\n=== PLAN OUTLINE (FOLLOW THIS STRUCTURE) ===",
            f"Overview: {outline.overview}",
        ]
        if outline.difficulty_arc:
            lines.append(f"Difficulty arc: {outline.difficulty_arc}")
        if outline.key_milestones:
            lines.append("Key milestones: " + " | ".join(outline.key_milestones[:8]))
        if outline.weekly_focus:
            lines.append("Weekly focus: " + " | ".join(outline.weekly_focus))
        if outline.rest_day_numbers:
            lines.append(f"Rest/light days (day numbers): {outline.rest_day_numbers}")
        for phase in outline.phases:
            goals = ", ".join(phase.goals[:3]) if phase.goals else ""
            lines.append(
                f"Phase '{phase.phase_name}' (days {phase.start_day}-{phase.end_day}): "
                f"{phase.focus}" + (f" | Goals: {goals}" if goals else "")
            )
        lines.append("=== END PLAN OUTLINE ===\n")
        return "\n".join(lines)

    def generate_outline(
        self,
        req: GeneratePlannerRequest,
        extracted_context: Optional[ExtractedUserContext] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[PlanOutline]:
        """Build a structural outline before generating daily content."""
        self._emit_progress(
            progress_callback,
            progress=22,
            progress_message="Planning structure and milestones...",
            current_stage="building_outline",
            stages_completed=2,
        )
        use_model = self.config.fast_model if req.fastMode else self.config.extraction_model
        lang_note = "Write in Thai." if req.language == "th" else "Write in English."
        context_bits = []
        if extracted_context and extracted_context.goals.primary_goal:
            context_bits.append(f"Primary goal: {extracted_context.goals.primary_goal}")
        if req.detailPrompt:
            context_bits.append(f"User description: {req.detailPrompt[:600]}")

        schema = {
            "name": "plan_outline",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "overview": {"type": "string"},
                    "difficulty_arc": {"type": ["string", "null"]},
                    "key_milestones": {"type": "array", "items": {"type": "string"}},
                    "weekly_focus": {"type": "array", "items": {"type": "string"}},
                    "rest_day_numbers": {"type": "array", "items": {"type": "integer"}},
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "phase_name": {"type": "string"},
                                "start_day": {"type": "integer", "minimum": 1},
                                "end_day": {"type": "integer", "minimum": 1},
                                "focus": {"type": "string"},
                                "goals": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["phase_name", "start_day", "end_day", "focus", "goals"],
                        },
                    },
                },
                "required": [
                    "overview",
                    "difficulty_arc",
                    "key_milestones",
                    "weekly_focus",
                    "rest_day_numbers",
                    "phases",
                ],
            },
        }

        user_msg = (
            f"{lang_note}\n"
            f"Category: {req.category}\n"
            f"Plan: {req.planName}\n"
            f"Total days: {req.totalDays}\n"
            f"Intensity: {req.intensity or 'moderate'}\n"
            f"Minutes per day: {req.minutesPerDay or 'flexible'}\n"
            + ("\n".join(context_bits) if context_bits else "")
            + "\n\nCreate a logical outline with phases covering all days 1.."
            f"{req.totalDays} without gaps. Include appropriate rest/light days."
        )

        try:
            response = get_openai_client().chat.completions.create(
                model=use_model,
                temperature=1.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert curriculum and habit-plan designer. "
                            "Output only JSON matching the schema. Phases must cover every day "
                            "from 1 to totalDays with no overlaps or gaps."
                        ),
                    },
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            raw = response.choices[0].message.content if response.choices else None
            if not raw:
                return None
            data = self._parse_json_response(raw)
            return PlanOutline(**data)
        except Exception as e:
            print(f"Warning: outline generation failed, continuing without outline: {e}")
            return None

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
            'localhost', '127.0.0.1', '0.0.0.0', 'example.org',
            'test.org', 'dummy.org', 'sample.com', 'demo.com'
        ]
        
        if any(pattern in link.lower() for pattern in invalid_patterns):
            return False
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(link)
            domain = parsed.netloc.lower()
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # More comprehensive and flexible domain validation
            # Allow common educational, government, and reputable domains
            trusted_domains = [
                # Educational institutions
                '.edu', '.ac.uk', '.ac.jp', '.ac.kr', '.ac.in',
                # Government domains
                '.gov', '.gov.uk', '.gov.au', '.gov.ca', '.gov.in',
                # International organizations
                '.org', '.int', '.un.org', '.who.int', '.unicef.org',
                # Major platforms
                'youtube.com', 'vimeo.com', 'ted.com', 'khanacademy.org',
                'coursera.org', 'edx.org', 'udemy.com', 'skillshare.com',
                'codecademy.com', 'freecodecamp.org', 'w3schools.com',
                'stackoverflow.com', 'github.com', 'gitlab.com',
                # News and reference
                'wikipedia.org', 'britannica.com', 'merriam-webster.com',
                'dictionary.com', 'thesaurus.com', 'oxford.com',
                # Health and medical
                'mayoclinic.org', 'healthline.com', 'webmd.com',
                'medlineplus.gov', 'cdc.gov', 'nih.gov', 'who.int',
                'clevelandclinic.org', 'hopkinsmedicine.org',
                # Finance
                'investopedia.com', 'nerdwallet.com', 'bankrate.com',
                'mint.com', 'yahoo.com', 'marketwatch.com', 'cnbc.com',
                'forbes.com', 'bloomberg.com', 'reuters.com',
                # Fitness and wellness
                'nike.com', 'adidas.com', 'fitnessblender.com', 'darebee.com',
                'myfitnesspal.com', 'bodybuilding.com', 'acefitness.org',
                'verywellfit.com', 'menshealth.com', 'womenshealthmag.com',
                # Travel
                'tripadvisor.com', 'booking.com', 'expedia.com', 'airbnb.com',
                'lonelyplanet.com', 'nationalgeographic.com', 'rome2rio.com',
                # Personal development
                'mindtools.com', 'psychologytoday.com', 'hbr.org',
                'lifehack.org', 'zenhabits.net', 'jamesclear.com',
                'charlesduhigg.com', 'gretchenrubin.com',
                # General platforms
                'medium.com', 'quora.com', 'reddit.com', 'linkedin.com',
                'twitter.com', 'facebook.com', 'instagram.com',
                # Technology
                'mozilla.org', 'w3.org', 'ietf.org', 'apache.org',
                'python.org', 'nodejs.org', 'reactjs.org', 'vuejs.org',
                'angular.io', 'typescript.org', 'developer.mozilla.org'
            ]
            
            # Check if domain matches any trusted domain pattern
            for trusted_domain in trusted_domains:
                if trusted_domain.startswith('.'):
                    # TLD or subdomain pattern (e.g., .edu, .gov)
                    if domain.endswith(trusted_domain):
                        return True
                else:
                    # Exact domain match (e.g., youtube.com)
                    if domain == trusted_domain or domain.endswith('.' + trusted_domain):
                        return True
            
            # Additional category-specific validation for more targeted domains
            category_specific_domains = {
                "learning": [
                    'udacity.com', 'pluralsight.com', 'lynda.com', 'treehouse.com',
                    'datacamp.com', 'kaggle.com', 'leetcode.com', 'hackerrank.com',
                    'codewars.com', 'exercism.io', 'scrimba.com', 'egghead.io'
                ],
                "exercise": [
                    'peloton.com', 'strava.com', 'runtastic.com', 'mapmyrun.com',
                    'myfitnesspal.com', 'cronometer.com', 'fitbit.com', 'garmin.com'
                ],
                "finance": [
                    'mint.com', 'ynab.com', 'personalcapital.com', 'wealthfront.com',
                    'betterment.com', 'robinhood.com', 'etrade.com', 'schwab.com'
                ],
                "health": [
                    'myfitnesspal.com', 'cronometer.com', 'loseit.com', 'sparkpeople.com',
                    'fitbit.com', 'garmin.com', 'apple.com/health', 'google.com/fit'
                ]
            }
            
            # Check category-specific domains
            if category in category_specific_domains:
                for cat_domain in category_specific_domains[category]:
                    if domain == cat_domain or domain.endswith('.' + cat_domain):
                        return True
            
            # Fallback: If no trusted domain matches, use more permissive validation
            # Allow any domain that doesn't match invalid patterns and has a reasonable structure
            if len(domain) > 3 and '.' in domain and not any(bad in domain for bad in ['localhost', '127.0.0.1', 'test', 'dummy', 'example', 'placeholder']):
                # Additional check: ensure it's not a suspicious domain
                suspicious_patterns = ['bit.ly', 'tinyurl', 'short.link', 'goo.gl', 't.co']
                if not any(pattern in domain for pattern in suspicious_patterns):
                    return True
            
            return False
            
        except Exception:
            return False

    def _enhance_task_description(self, task_text: str, category: str) -> str:
        """Enhance task description to be more detailed and actionable"""
        if not task_text or len(task_text.strip()) < 10:
            # Provide category-specific default detailed tasks
            enhanced_tasks = {
                "learning": "Study session: Review previous material for 15 minutes, then practice new concepts with hands-on exercises. Take notes on key points and create a summary of what you learned.",
                "exercise": "Physical activity: Start with 5-minute warm-up (light stretching or walking), perform main exercise for 20 minutes, finish with 5-minute cool-down. Focus on proper form and breathing.",
                "travel": "Travel planning: Research your destination, check weather conditions, create a packing list, and plan your daily itinerary. Consider transportation options and local customs.",
                "finance": "Financial review: Check your bank account balance, review recent transactions, update your budget spreadsheet, and set financial goals for the week ahead.",
                "health": "Wellness check: Take your vital signs (if applicable), review your nutrition for the day, plan healthy meals, and schedule any necessary medical appointments.",
                "personal_development": "Self-reflection: Spend 10 minutes journaling about your goals, review your progress, identify areas for improvement, and plan your next steps.",
                "other": "Task completion: Break down the task into smaller steps, set a timer for focused work, take breaks as needed, and track your progress throughout the session."
            }
            return enhanced_tasks.get(category, enhanced_tasks["other"])
        
        # If task exists but is too short, enhance it
        enhanced_text = task_text.strip()
        
        # Add category-specific enhancements
        if category == "learning" and len(enhanced_text) < 30:
            enhanced_text += " Focus on understanding the concepts, practice with examples, and take notes on key points."
        elif category == "exercise" and len(enhanced_text) < 30:
            enhanced_text += " Start with a warm-up, maintain proper form throughout, and finish with a cool-down. Stay hydrated and listen to your body."
        elif category == "finance" and len(enhanced_text) < 30:
            enhanced_text += " Review your current financial situation, track your progress, and make adjustments as needed."
        elif category == "health" and len(enhanced_text) < 30:
            enhanced_text += " Pay attention to your body's signals, maintain proper nutrition, and consult healthcare professionals when needed."
        
        return enhanced_text

    def _check_duplicate_links(self, days: List[Dict]) -> List[str]:
        """Check for duplicate links within the plan and return list of duplicates"""
        # Since we're not using links anymore, return empty list
        return []

    def _analyze_plan_requirements(self, req: GeneratePlannerRequest) -> Dict[str, Any]:
        """Analyze the plan requirements to determine optimal chunking strategy"""
        analysis = {
            "complexity": "simple",
            "progression_type": "linear",
            "optimal_chunk_size": 7,
            "phases": [],
            "special_considerations": []
        }
        
        # Analyze based on category and total days (OPTIMIZED FOR SPEED - larger chunks = fewer API calls)
        if req.totalDays <= 10:
            analysis["complexity"] = "simple"
            analysis["optimal_chunk_size"] = req.totalDays  # Single chunk
        elif req.totalDays <= 20:
            analysis["complexity"] = "moderate"
            analysis["optimal_chunk_size"] = req.totalDays  # Single chunk for speed
        elif req.totalDays <= 30:
            analysis["complexity"] = "moderate"
            analysis["optimal_chunk_size"] = 30  # Try single chunk first (gpt-4o can handle 30 days)
        else:
            analysis["complexity"] = "complex"
            analysis["optimal_chunk_size"] = 30  # Max 30 days per chunk to balance quality/speed
        
        # Category-specific analysis
        if req.category == "learning":
            if req.totalDays >= 30:
                analysis["progression_type"] = "spiral"
                analysis["phases"] = [
                    {"name": "Foundation", "focus": "Basic concepts and fundamentals"},
                    {"name": "Practice", "focus": "Hands-on application and skill building"},
                    {"name": "Mastery", "focus": "Advanced techniques and real-world projects"}
                ]
            else:
                analysis["progression_type"] = "linear"
                analysis["phases"] = [
                    {"name": "Learning", "focus": "Progressive skill development"}
                ]
        
        elif req.category == "exercise":
            if req.totalDays >= 30:
                analysis["progression_type"] = "periodized"
                analysis["phases"] = [
                    {"name": "Adaptation", "focus": "Building base fitness and movement patterns"},
                    {"name": "Progression", "focus": "Increasing intensity and complexity"},
                    {"name": "Peak", "focus": "Maximum performance and advanced techniques"}
                ]
            else:
                analysis["progression_type"] = "linear"
                analysis["phases"] = [
                    {"name": "Fitness", "focus": "Progressive workout development"}
                ]
        
        elif req.category == "travel":
            analysis["progression_type"] = "thematic"
            if req.totalDays >= 21:
                analysis["phases"] = [
                    {"name": "Planning", "focus": "Research, booking, and preparation"},
                    {"name": "Preparation", "focus": "Final preparations and logistics"},
                    {"name": "Execution", "focus": "Travel activities and experiences"}
                ]
            else:
                analysis["phases"] = [
                    {"name": "Travel", "focus": "Planning and preparation activities"}
                ]
        
        elif req.category == "finance":
            analysis["progression_type"] = "foundational"
            if req.totalDays >= 30:
                analysis["phases"] = [
                    {"name": "Assessment", "focus": "Current financial situation analysis"},
                    {"name": "Planning", "focus": "Budget creation and goal setting"},
                    {"name": "Implementation", "focus": "Active financial management"}
                ]
            else:
                analysis["phases"] = [
                    {"name": "Finance", "focus": "Financial planning and management"}
                ]
        
        elif req.category == "health":
            analysis["progression_type"] = "holistic"
            if req.totalDays >= 30:
                analysis["phases"] = [
                    {"name": "Awareness", "focus": "Health assessment and habit tracking"},
                    {"name": "Implementation", "focus": "Building healthy routines"},
                    {"name": "Optimization", "focus": "Fine-tuning and advanced wellness"}
                ]
            else:
                analysis["phases"] = [
                    {"name": "Wellness", "focus": "Health and wellness development"}
                ]
        
        elif req.category == "personal_development":
            analysis["progression_type"] = "transformational"
            if req.totalDays >= 30:
                analysis["phases"] = [
                    {"name": "Self-Discovery", "focus": "Understanding yourself and your goals"},
                    {"name": "Skill Building", "focus": "Developing new capabilities and habits"},
                    {"name": "Integration", "focus": "Applying skills in real-world situations"}
                ]
            else:
                analysis["phases"] = [
                    {"name": "Growth", "focus": "Personal development and improvement"}
                ]
        
        else:  # "other" category
            analysis["progression_type"] = "custom"
            analysis["phases"] = [
                {"name": "Development", "focus": "Custom plan based on user requirements"}
            ]
        
        # Analyze detail prompt for special considerations
        if req.detailPrompt:
            detail_lower = req.detailPrompt.lower()
            if any(word in detail_lower for word in ["beginner", "basic", "intro"]):
                analysis["special_considerations"].append("beginner_friendly")
            if any(word in detail_lower for word in ["advanced", "expert", "professional"]):
                analysis["special_considerations"].append("advanced_level")
            if any(word in detail_lower for word in ["intensive", "challenging", "difficult"]):
                analysis["special_considerations"].append("high_intensity")
            if any(word in detail_lower for word in ["flexible", "adaptable", "customizable"]):
                analysis["special_considerations"].append("flexible_approach")
        
        return analysis

    def _create_intelligent_chunks(self, req: GeneratePlannerRequest, analysis: Dict[str, Any]) -> List[PlanChunk]:
        """Create intelligent, progressive chunks based on plan analysis"""
        chunks = []
        total_days = req.totalDays
        optimal_chunk_size = analysis["optimal_chunk_size"]
        
        # Calculate number of chunks needed
        num_chunks = (total_days + optimal_chunk_size - 1) // optimal_chunk_size
        
        # Adjust chunk sizes to distribute days evenly
        base_chunk_size = total_days // num_chunks
        remainder = total_days % num_chunks
        
        current_day = 1
        phases = analysis["phases"]
        
        for i in range(num_chunks):
            # Calculate chunk size (distribute remainder across first chunks)
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            end_day = current_day + chunk_size - 1
            
            # Determine phase and progression level
            if len(phases) > 0:
                phase_idx = min(i, len(phases) - 1)
                phase = phases[phase_idx]
                phase_name = phase["name"]
                focus_area = phase["focus"]
            else:
                phase_name = f"Phase {i + 1}"
                focus_area = f"Continued development in {req.category}"
            
            # Determine progression level
            if num_chunks == 1:
                progression_level = "beginner"
            elif i == 0:
                progression_level = "beginner"
            elif i == num_chunks - 1:
                progression_level = "advanced"
            else:
                progression_level = "intermediate"
            
            # Create key goals for this chunk
            key_goals = self._generate_chunk_goals(req, phase_name, progression_level, i + 1, num_chunks)
            
            # Create special instructions
            special_instructions = self._generate_chunk_instructions(
                req, phase_name, progression_level, current_day, end_day, i + 1, num_chunks
            )
            
            chunk = PlanChunk(
                start_day=current_day,
                end_day=end_day,
                phase_name=phase_name,
                focus_area=focus_area,
                progression_level=progression_level,
                key_goals=key_goals,
                special_instructions=special_instructions
            )
            
            chunks.append(chunk)
            current_day = end_day + 1
        
        return chunks

    def _generate_chunk_goals(self, req: GeneratePlannerRequest, phase_name: str, 
                            progression_level: str, chunk_num: int, total_chunks: int) -> List[str]:
        """Generate specific goals for a chunk based on its phase and progression level"""
        goals = []
        
        if req.category == "learning":
            if progression_level == "beginner":
                goals = [
                    "Establish foundational knowledge and basic skills",
                    "Build confidence through guided practice",
                    "Develop consistent learning habits"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Apply knowledge in practical scenarios",
                    "Build upon previous learning with new concepts",
                    "Develop problem-solving skills"
                ]
            else:  # advanced
                goals = [
                    "Master advanced techniques and concepts",
                    "Create original projects or applications",
                    "Develop expertise and teaching ability"
                ]
        
        elif req.category == "exercise":
            if progression_level == "beginner":
                goals = [
                    "Build basic fitness foundation",
                    "Learn proper form and technique",
                    "Establish consistent workout routine"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Increase workout intensity and complexity",
                    "Develop strength and endurance",
                    "Master advanced movement patterns"
                ]
            else:  # advanced
                goals = [
                    "Achieve peak performance levels",
                    "Master advanced training techniques",
                    "Develop specialized skills"
                ]
        
        elif req.category == "travel":
            if progression_level == "beginner":
                goals = [
                    "Research destinations and create itinerary",
                    "Plan logistics and make bookings",
                    "Prepare travel documents and essentials"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Finalize travel arrangements",
                    "Prepare for cultural experiences",
                    "Plan activities and experiences"
                ]
            else:  # advanced
                goals = [
                    "Execute travel plans and activities",
                    "Adapt to local conditions and culture",
                    "Document and reflect on experiences"
                ]
        
        elif req.category == "finance":
            if progression_level == "beginner":
                goals = [
                    "Assess current financial situation",
                    "Create basic budget and track expenses",
                    "Establish financial goals and priorities"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Implement budgeting and saving strategies",
                    "Learn about investment basics",
                    "Optimize spending and reduce debt"
                ]
            else:  # advanced
                goals = [
                    "Advanced investment and wealth building",
                    "Tax optimization and financial planning",
                    "Long-term financial security planning"
                ]
        
        elif req.category == "health":
            if progression_level == "beginner":
                goals = [
                    "Assess current health and wellness",
                    "Establish healthy daily routines",
                    "Track nutrition and exercise habits"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Optimize nutrition and fitness routines",
                    "Develop stress management techniques",
                    "Improve sleep and recovery habits"
                ]
            else:  # advanced
                goals = [
                    "Fine-tune health and wellness systems",
                    "Develop advanced wellness practices",
                    "Maintain long-term health optimization"
                ]
        
        elif req.category == "personal_development":
            if progression_level == "beginner":
                goals = [
                    "Self-assessment and goal setting",
                    "Develop self-awareness and reflection habits",
                    "Build foundational personal skills"
                ]
            elif progression_level == "intermediate":
                goals = [
                    "Develop advanced personal skills",
                    "Improve relationships and communication",
                    "Build productivity and time management systems"
                ]
            else:  # advanced
                goals = [
                    "Master advanced personal development techniques",
                    "Develop leadership and mentoring skills",
                    "Create lasting positive change"
                ]
        
        else:  # "other" category
            goals = [
                f"Progress in {req.category} development",
                f"Build skills and knowledge in {req.category}",
                f"Achieve specific goals in {req.category}"
            ]
        
        return goals

    def _generate_chunk_instructions(self, req: GeneratePlannerRequest, phase_name: str,
                                   progression_level: str, start_day: int, end_day: int,
                                   chunk_num: int, total_chunks: int) -> str:
        """Generate specific instructions for a chunk to ensure continuity and progression"""
        instructions = []
        
        # Base context
        instructions.append(f"This is {phase_name} (Days {start_day}-{end_day}) of a {req.totalDays}-day {req.category} plan.")
        
        # Progression context
        if chunk_num > 1:
            instructions.append(f"This builds upon the previous {chunk_num - 1} phase(s) and should reference previous learning/progress.")
        
        if chunk_num < total_chunks:
            instructions.append(f"This prepares for the upcoming {total_chunks - chunk_num} phase(s) and should set up future development.")
        
        # Progression level specific instructions
        if progression_level == "beginner":
            instructions.append("Focus on building strong foundations, clear explanations, and confidence-building activities.")
            instructions.append("Include more detailed instructions and safety considerations.")
        elif progression_level == "intermediate":
            instructions.append("Build upon previous knowledge with increased complexity and practical applications.")
            instructions.append("Include problem-solving and critical thinking elements.")
        else:  # advanced
            instructions.append("Focus on mastery, advanced techniques, and real-world applications.")
            instructions.append("Include creative challenges and independent project work.")
        
        # Category-specific instructions
        if req.category == "learning":
            instructions.append("Ensure each day builds logically on the previous day's content.")
            instructions.append("Include review and practice opportunities to reinforce learning.")
        elif req.category == "exercise":
            instructions.append("Include proper warm-up and cool-down for each day.")
            instructions.append("Ensure progressive overload while maintaining safety.")
        elif req.category == "travel":
            instructions.append("Consider practical logistics and realistic timelines.")
            instructions.append("Include cultural awareness and local customs.")
        elif req.category == "finance":
            instructions.append("Include practical, actionable financial tasks.")
            instructions.append("Ensure tasks are relevant to the user's financial situation.")
        elif req.category == "health":
            instructions.append("Focus on sustainable, evidence-based health practices.")
            instructions.append("Include both physical and mental wellness aspects.")
        elif req.category == "personal_development":
            instructions.append("Include self-reflection and journaling opportunities.")
            instructions.append("Focus on practical application of personal development concepts.")
        
        # Special considerations from analysis
        if req.detailPrompt:
            instructions.append(f"Consider these specific requirements: {req.detailPrompt}")
        
        return " ".join(instructions)

    def _handle_generation_failure(self, req: GeneratePlannerRequest, error_context: str) -> None:
        """Handle generation failures with proper error reporting instead of fallback plans"""
        error_message = f"Failed to generate {req.totalDays}-day {req.category} plan: {error_context}"
        user_message = f"We couldn't generate your {req.totalDays}-day {req.category} plan. Please try again with fewer days or simpler requirements."
        
        raise PlannerGenerationError(error_message, user_message)

    def _build_system_prompt(
        self,
        category: str,
        extracted_context: Optional[ExtractedUserContext] = None,
        refinement_mode: bool = False,
    ) -> str:
        """Build a personalized system prompt based on category and extracted context"""
        
        if refinement_mode:
            base_prompt = (
                "You are an expert plan editor for a lifestyle planner app. "
                "The user has a DRAFT plan and wants specific changes. "
                "Preserve days, tasks, and structure they did NOT ask to change. "
                "Apply only the requested refinements while keeping the same totalDays. "
                "Output a complete updated plan JSON matching the schema. "
            )
        else:
            base_prompt = (
                "You are an expert planner-content generator for a lifestyle planner app. "
                "Generate structured daily plans with clear titles, concise summaries, and actionable tasks. "
            )
        
        # Category-specific expertise
        category_expertise = {
            "learning": (
                "You are a learning science expert who understands spaced repetition, active recall, "
                "and progressive skill building. Design learning plans that:\n"
                "- Start with foundational concepts before advancing\n"
                "- Include regular review sessions to reinforce retention\n"
                "- Alternate between theory and practical application\n"
                "- Build in weekly reflection and consolidation days\n"
                "- Vary learning activities to maintain engagement\n"
            ),
            "exercise": (
                "You are a certified fitness professional who understands exercise physiology and "
                "progressive training. Design workout plans that:\n"
                "- Follow proper periodization (preparation, building, peak, recovery)\n"
                "- Include appropriate warm-up and cool-down for each session\n"
                "- Alternate muscle groups and training modalities\n"
                "- Build in rest days and deload periods\n"
                "- Progress safely with gradual intensity increases\n"
                "- Adapt to user's available equipment and limitations\n"
            ),
            "travel": (
                "You are an experienced travel planner who understands logistics and local experiences. "
                "Design travel itineraries that:\n"
                "- Fill each day with a realistic schedule using most of the user's daily time budget "
                "(typically several hours: sightseeing blocks, meals, transit, and short rest breaks)\n"
                "- Group activities by geographic proximity to minimize backtracking\n"
                "- Balance busy exploration days with lighter recovery days\n"
                "- Include practical logistics (transport, timing, reservations, opening hours)\n"
                "- Suggest local experiences and hidden gems\n"
                "- Account for weather, seasons, and local events\n"
                "- Include buffer time for unexpected discoveries\n"
            ),
            "finance": (
                "You are a financial literacy expert who understands budgeting, saving, and investing. "
                "Design financial plans that:\n"
                "- Start with assessment and goal-setting\n"
                "- Build foundational habits before complex strategies\n"
                "- Include regular tracking and review tasks\n"
                "- Progress from saving to investing concepts\n"
                "- Provide actionable, specific financial tasks\n"
                "- Account for user's financial situation and goals\n"
            ),
            "health": (
                "You are a wellness expert who understands holistic health and sustainable habits. "
                "Design health plans that:\n"
                "- Address multiple dimensions (physical, mental, nutritional)\n"
                "- Build sustainable habits over quick fixes\n"
                "- Include regular self-assessment checkpoints\n"
                "- Balance action items with rest and recovery\n"
                "- Provide evidence-based recommendations\n"
                "- Adapt to user's health conditions and preferences\n"
            ),
            "personal_development": (
                "You are a personal development coach who understands behavior change and growth. "
                "Design development plans that:\n"
                "- Start with self-reflection and goal clarity\n"
                "- Build habits using proven frameworks (habit stacking, tiny habits)\n"
                "- Include journaling and reflection prompts\n"
                "- Progress from awareness to action to mastery\n"
                "- Balance challenge with achievability\n"
                "- Incorporate accountability mechanisms\n"
            ),
            "other": (
                "You are a versatile planning expert who can adapt to any domain. "
                "Design plans that:\n"
                "- Analyze the user's specific needs carefully\n"
                "- Create logical progression from start to goal\n"
                "- Include variety and engagement\n"
                "- Build in reflection and adjustment points\n"
            )
        }
        
        # Add personalization based on extracted context
        personalization_rules = []
        
        if extracted_context:
            # Experience level adaptation
            if extracted_context.profile.experience_level:
                level = extracted_context.profile.experience_level
                if level == "beginner":
                    personalization_rules.append(
                        "BEGINNER ADAPTATION: Use simple language, provide detailed explanations, "
                        "break tasks into smaller steps, include encouragement, avoid jargon."
                    )
                elif level == "intermediate":
                    personalization_rules.append(
                        "INTERMEDIATE ADAPTATION: Assume basic knowledge, focus on building skills, "
                        "include moderate challenges, reference foundational concepts briefly."
                    )
                elif level == "advanced":
                    personalization_rules.append(
                        "ADVANCED ADAPTATION: Use technical terminology appropriately, focus on optimization, "
                        "include challenging tasks, assume strong foundation."
                    )
                elif level == "expert":
                    personalization_rules.append(
                        "EXPERT ADAPTATION: Focus on mastery and refinement, include advanced techniques, "
                        "provide nuanced recommendations, challenge conventional approaches."
                    )
            
            # Age group adaptation
            if extracted_context.profile.age_group:
                age = extracted_context.profile.age_group
                if age == "teen":
                    personalization_rules.append(
                        "TEEN ADAPTATION: Use engaging, relatable language, include social elements, "
                        "shorter task durations, gamification where appropriate."
                    )
                elif age == "senior":
                    personalization_rules.append(
                        "SENIOR ADAPTATION: Prioritize safety and accessibility, moderate intensity, "
                        "clear instructions, health considerations, appropriate pacing."
                    )
            
            # Physical limitations
            if extracted_context.profile.physical_limitations:
                limitations = ", ".join(extracted_context.profile.physical_limitations)
                personalization_rules.append(
                    f"PHYSICAL LIMITATIONS: User has mentioned: {limitations}. "
                    "Provide safe alternatives, avoid contraindicated activities, include modifications."
                )
            
            # Goal-based personalization
            if extracted_context.goals.primary_goal:
                personalization_rules.append(
                    f"PRIMARY GOAL: User wants to '{extracted_context.goals.primary_goal}'. "
                    "Every day should contribute toward this goal. Reference progress toward it."
                )
            
            if extracted_context.goals.target_outcome:
                personalization_rules.append(
                    f"TARGET OUTCOME: User aims for '{extracted_context.goals.target_outcome}'. "
                    "Include measurable progress markers and milestone celebrations."
                )
            
            if extracted_context.goals.deadline:
                personalization_rules.append(
                    f"DEADLINE: User has a target date of '{extracted_context.goals.deadline}'. "
                    "Pace the plan appropriately to achieve goals by this date."
                )
            
            # Motivation type
            if extracted_context.goals.motivation_type:
                motivation = extracted_context.goals.motivation_type
                motivation_styles = {
                    "achievement": "Focus on progress metrics, milestones, and accomplishments.",
                    "health": "Emphasize health benefits and long-term wellbeing outcomes.",
                    "social": "Include social elements, accountability, and sharing opportunities.",
                    "mastery": "Focus on skill development, depth of understanding, expertise.",
                    "enjoyment": "Prioritize fun, variety, and intrinsic satisfaction.",
                    "necessity": "Focus on practical outcomes and efficient goal achievement."
                }
                personalization_rules.append(
                    f"MOTIVATION: User is motivated by {motivation}. {motivation_styles.get(motivation, '')}"
                )
            
            # Constraints
            if extracted_context.constraints.budget_level:
                budget = extracted_context.constraints.budget_level
                if budget == "minimal":
                    personalization_rules.append(
                        "BUDGET: User has a tight budget. Prioritize free/low-cost options, DIY approaches."
                    )
                elif budget == "unlimited":
                    personalization_rules.append(
                        "BUDGET: User has flexible budget. Can suggest premium options and services."
                    )
            
            if extracted_context.constraints.excluded_activities:
                excluded = ", ".join(extracted_context.constraints.excluded_activities)
                personalization_rules.append(
                    f"EXCLUDED ACTIVITIES: Do NOT include: {excluded}"
                )
            
            if extracted_context.constraints.preferred_activities:
                preferred = ", ".join(extracted_context.constraints.preferred_activities)
                personalization_rules.append(
                    f"PREFERRED ACTIVITIES: Prioritize including: {preferred}"
                )
            
            # Learning style
            if extracted_context.learning_style.learning_style:
                style = extracted_context.learning_style.learning_style
                style_guidance = {
                    "visual": "Include diagrams, visualizations, demonstrations, video references.",
                    "reading": "Provide detailed written instructions, reading materials, documentation.",
                    "hands_on": "Focus on practical exercises, experiments, doing over theory.",
                    "auditory": "Include verbal instructions, discussions, audio resources.",
                    "mixed": "Vary between different learning modalities."
                }
                personalization_rules.append(
                    f"LEARNING STYLE: User prefers {style} learning. {style_guidance.get(style, '')}"
                )
            
            # Pace preference
            if extracted_context.learning_style.pace_preference:
                pace = extracted_context.learning_style.pace_preference
                pace_guidance = {
                    "slow_steady": "Gradual progression, more practice time, thorough coverage.",
                    "moderate": "Balanced pace with adequate practice and progression.",
                    "intensive": "Aggressive progression, challenging tasks, maximum output.",
                    "flexible": "Adaptable structure with optional extensions."
                }
                personalization_rules.append(
                    f"PACE: User prefers {pace} pace. {pace_guidance.get(pace, '')}"
                )
            
            # Tone preference
            if extracted_context.tone_preference:
                tone = extracted_context.tone_preference
                tone_guidance = {
                    "professional": "Use formal language, focus on efficiency and results.",
                    "casual": "Use friendly, relaxed language, conversational tone.",
                    "motivational": "Use encouraging, inspiring language, celebrate progress.",
                    "educational": "Use informative, teaching-focused language, explain concepts.",
                    "friendly": "Use warm, supportive language, personal touch."
                }
                personalization_rules.append(
                    f"TONE: Write in a {tone} tone. {tone_guidance.get(tone, '')}"
                )
            
            # Category-specific context
            if extracted_context.category_specific:
                cs = extracted_context.category_specific
                if category == "exercise" and cs:
                    if cs.get("equipment_available"):
                        equip = ", ".join(cs["equipment_available"])
                        personalization_rules.append(f"EQUIPMENT: User has access to: {equip}")
                    if cs.get("workout_location"):
                        personalization_rules.append(f"LOCATION: Workouts will be at: {cs['workout_location']}")
                    if cs.get("injuries_or_limitations"):
                        injuries = ", ".join(cs["injuries_or_limitations"])
                        personalization_rules.append(f"INJURIES/LIMITATIONS: Be careful of: {injuries}")
                
                elif category == "learning" and cs:
                    if cs.get("subject_area"):
                        personalization_rules.append(f"SUBJECT: Focus on learning {cs['subject_area']}")
                    if cs.get("exam_or_certification"):
                        personalization_rules.append(f"EXAM GOAL: Preparing for {cs['exam_or_certification']}")
                    if cs.get("current_knowledge"):
                        personalization_rules.append(f"CURRENT KNOWLEDGE: User already knows: {cs['current_knowledge']}")
                
                elif category == "travel" and cs:
                    if cs.get("destination"):
                        personalization_rules.append(f"DESTINATION: Planning trip to {cs['destination']}")
                    if cs.get("travel_companions"):
                        personalization_rules.append(f"TRAVELERS: {cs['travel_companions']}")
                    if cs.get("must_see_places"):
                        places = ", ".join(cs["must_see_places"])
                        personalization_rules.append(f"MUST VISIT: Include {places}")
                
                elif category == "finance" and cs:
                    if cs.get("financial_goal"):
                        personalization_rules.append(f"FINANCIAL GOAL: {cs['financial_goal']}")
                    if cs.get("saving_target"):
                        personalization_rules.append(f"SAVING TARGET: {cs['saving_target']}")
                
                elif category == "health" and cs:
                    if cs.get("health_goal"):
                        personalization_rules.append(f"HEALTH GOAL: {cs['health_goal']}")
                    if cs.get("health_conditions"):
                        conditions = ", ".join(cs["health_conditions"])
                        personalization_rules.append(f"HEALTH CONDITIONS: Consider: {conditions}")
                    if cs.get("diet_preferences"):
                        personalization_rules.append(f"DIET: {cs['diet_preferences']}")
                
                elif category == "personal_development" and cs:
                    if cs.get("skills_to_develop"):
                        skills = ", ".join(cs["skills_to_develop"])
                        personalization_rules.append(f"SKILLS TO DEVELOP: {skills}")
                    if cs.get("habits_to_build"):
                        habits = ", ".join(cs["habits_to_build"])
                        personalization_rules.append(f"HABITS TO BUILD: {habits}")
            
            # Key requirements
            if extracted_context.key_requirements:
                reqs = "\n".join([f"  - {r}" for r in extracted_context.key_requirements])
                personalization_rules.append(f"KEY REQUIREMENTS:\n{reqs}")
            
            # Special considerations
            if extracted_context.special_considerations:
                special = "\n".join([f"  - {s}" for s in extracted_context.special_considerations])
                personalization_rules.append(f"SPECIAL CONSIDERATIONS:\n{special}")
        
        # Build the complete prompt
        personalization_section = ""
        if personalization_rules:
            personalization_section = (
                "\n\n=== PERSONALIZATION (CRITICAL - FOLLOW THESE) ===\n" +
                "\n".join(personalization_rules) +
                "\n=== END PERSONALIZATION ===\n"
            )
        
        rules = """
GENERATION RULES:
1) Keep each day practical (2-4 tasks tailored to the user's context)
2) Add brief tips that are relevant to the user's situation
3) Titles should be short, motivating, and reflect the day's focus
4) Never invent unsafe or extreme advice; prefer safe defaults
5) CRITICAL: Output MUST be valid JSON matching the exact schema provided
6) Include ALL required fields: planName, category, totalDays, createdAt, days, summary, tags, difficultyLevel, estimatedCompletionRate
7) ABSOLUTE REQUIREMENT: The 'days' array MUST contain EXACTLY the number of days specified in totalDays
8) TIME ALLOCATION: If minutesPerDay is specified, allocate time based on task complexity (±20% flexibility)
9) DAY NUMBERING: dayNumber must start from 1 and increment sequentially
10) DETAILED TASKS: Each task MUST include comprehensive, actionable instructions

PLAN SUMMARY REQUIREMENTS (REQUIRED):
You MUST include a comprehensive 'summary' object with these fields:
- overview: 2-3 sentence overview of the entire plan and its approach
- targetAudience: Who this plan is ideal for (e.g., "Beginners with no prior experience" or "Intermediate learners looking to advance")
- expectedOutcomes: List of 3-5 specific outcomes users can expect to achieve
- keyMilestones: List of 3-5 major milestones throughout the plan (e.g., "Week 1: Master fundamentals", "Week 2: Build first project")
- difficultyProgression: How difficulty evolves (e.g., "Starts easy, gradually increases to intermediate level by week 3")
- totalEstimatedHours: Calculate total hours based on daily time allocation and number of days
- prerequisites: List any prerequisites (or null if none required)
- tipsForSuccess: 3-5 actionable tips for users to maximize success
- weeklyFocus: Brief description of each week's main focus area

Also include:
- tags: 5-8 relevant tags for categorization (e.g., ["python", "programming", "beginner", "30-day", "coding"])
- difficultyLevel: Overall difficulty ("beginner", "intermediate", "advanced", or "mixed")
- estimatedCompletionRate: Realistic completion expectation (e.g., "85% with consistent daily practice")

TASK QUALITY REQUIREMENTS:
✓ Provide specific, actionable steps personalized to the user
✓ Include relevant tips, techniques, or methods
✓ Give clear success criteria or what to expect
✓ Include safety considerations where applicable
✓ Make tasks self-contained and complete
✓ Use the 'note' field for additional helpful details
✓ Reference user's goals, equipment, and preferences when relevant

TASK EXAMPLES:
✅ GOOD: 'Practice Python variables: Create 5 different variable types (string, integer, float, boolean, list). Write a simple program that uses each type and prints the results. Focus on proper naming conventions and data type understanding.'
✅ GOOD: 'Morning cardio workout: Do 20 minutes of moderate-intensity exercise (brisk walking, jogging, or cycling). Start with 5-minute warm-up, maintain steady pace for 15 minutes, finish with 5-minute cool-down.'
❌ BAD: 'Learn Python' (too vague)
❌ BAD: 'Do some exercise' (not specific enough)
"""
        
        return base_prompt + category_expertise.get(category, category_expertise["other"]) + personalization_section + rules

    def _generate_chunk_worker(
        self,
        chunk_info: Tuple[
            int, 'PlanChunk', GeneratePlannerRequest,
            Optional[ExtractedUserContext], int, Optional[PlanOutline],
            Optional[ProgressCallback],
        ],
    ) -> Tuple[int, Optional[PlannerContent], Optional[str]]:
        """
        Worker function for parallel chunk generation.
        Returns: (chunk_idx, content, error_message)
        """
        chunk_idx, chunk, req, extracted_context, total_chunks, plan_outline, progress_callback = chunk_info
        chunk_days = chunk.end_day - chunk.start_day + 1
        
        max_retries = 2
        for retry in range(max_retries + 1):
            try:
                # Create enhanced request for this chunk
                enhanced_detail_prompt = self._build_chunk_prompt(
                    req, chunk, chunk_idx, total_chunks, plan_outline
                )
                
                chunk_req = GeneratePlannerRequest(
                    planName=f"{req.planName} - {chunk.phase_name}",
                    category=req.category,
                    totalDays=chunk_days,
                    detailPrompt=enhanced_detail_prompt,
                    minutesPerDay=req.minutesPerDay,
                    intensity=req.intensity,
                    language=req.language,
                    startDate=req.startDate,
                    timeOfDay=req.timeOfDay,
                    fastMode=req.fastMode,  # Pass through fast mode
                    skipContextExtraction=True  # Always skip for chunks (already extracted)
                )
                
                # Generate this chunk with pre-extracted context (no re-extraction needed)
                chunk_content = self.generate_single(
                    chunk_req,
                    extracted_context=extracted_context,
                    plan_outline=plan_outline,
                    progress_callback=progress_callback,
                )
                
                # Adjust day numbers
                for day in chunk_content.days:
                    day.dayNumber = chunk.start_day + (day.dayNumber - 1)
                
                return (chunk_idx, chunk_content, None)
                
            except Exception as e:
                if retry == max_retries:
                    return (chunk_idx, None, f"Failed chunk {chunk_idx} ({chunk.phase_name}): {str(e)}")
                time.sleep(0.5)
        
        return (chunk_idx, None, f"Failed chunk {chunk_idx} after retries")

    def generate_chunked(
        self,
        req: GeneratePlannerRequest,
        progress_callback: Optional[ProgressCallback] = None,
        plan_outline: Optional[PlanOutline] = None,
    ) -> PlannerContent:
        """Generate planner content using PARALLEL chunked approach for large plans (>7 days)"""
        if req.totalDays <= 7:
            return self.generate_single(
                req,
                progress_callback=progress_callback,
                plan_outline=plan_outline,
            )
        
        # Validate maximum days
        max_days = 60
        if req.totalDays > max_days:
            raise PlannerGenerationError(
                f"Plan too large: {req.totalDays} days exceeds maximum of {max_days}",
                f"Plans cannot exceed {max_days} days. Please reduce the number of days and try again."
            )
        
        generation_start_time = time.time()
        
        if req.fastMode:
            print(f"Fast mode enabled for chunked generation")
        
        # Stage 1: Extract user context ONCE at the beginning (shared across all chunks)
        # Skip if skipContextExtraction is enabled
        extracted_context = None
        if req.detailPrompt and not req.skipContextExtraction:
            print(f"Extracting user context for chunked generation...")
            context_extractor = ContextExtractor(
                model=self.config.extraction_model,
                temperature=self.config.extraction_temperature
            )
            extracted_context = context_extractor.extract_context(
                detail_prompt=req.detailPrompt,
                category=req.category,
                plan_name=req.planName
            )
            if extracted_context:
                print(f"Context extraction successful. Primary goal: {extracted_context.goals.primary_goal}")
            else:
                print("Context extraction returned None, proceeding without structured context")
        elif req.skipContextExtraction:
            print("Context extraction skipped (skipContextExtraction=true)")

        self._emit_progress(
            progress_callback,
            progress=20,
            progress_message="Understanding your goals...",
            current_stage="extracting_context",
            stages_completed=1,
        )

        if plan_outline is None:
            plan_outline = self.generate_outline(req, extracted_context, progress_callback)
        
        context_time = time.time() - generation_start_time
        print(f"Context extraction completed in {context_time:.2f}s")
        
        # Analyze plan requirements to determine optimal chunking strategy
        analysis = self._analyze_plan_requirements(req)
        
        # Create intelligent chunks based on analysis
        chunks = self._create_intelligent_chunks(req, analysis)
        print(f"Created {len(chunks)} chunks for PARALLEL generation")
        
        now_s = int(time.time())
        
        # Prepare chunk info for parallel processing
        chunk_infos = [
            (idx + 1, chunk, req, extracted_context, len(chunks), plan_outline, progress_callback)
            for idx, chunk in enumerate(chunks)
        ]

        self._emit_progress(
            progress_callback,
            progress=35,
            progress_message=f"Generating {req.totalDays} days in {len(chunks)} phases...",
            current_stage="generating_days",
            stages_completed=3,
        )
        
        # PARALLEL GENERATION using ThreadPoolExecutor
        results = {}
        errors = []
        
        # Use max 4 workers to avoid rate limits, but process chunks in parallel
        max_workers = min(len(chunks), 4)
        print(f"Starting parallel generation with {max_workers} workers...")
        
        parallel_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for parallel processing
            future_to_chunk = {
                executor.submit(self._generate_chunk_worker, info): info[0]
                for info in chunk_infos
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    idx, content, error = future.result()
                    if error:
                        errors.append(error)
                        print(f"Chunk {idx} failed: {error}")
                    else:
                        results[idx] = content
                        print(f"Chunk {idx} completed successfully")
                        done_chunks = len(results)
                        pct = 35 + int((done_chunks / len(chunks)) * 50)
                        self._emit_progress(
                            progress_callback,
                            progress=min(pct, 85),
                            progress_message=f"Completed phase {done_chunks} of {len(chunks)}...",
                            current_stage="generating_days",
                        )
                except Exception as e:
                    errors.append(f"Chunk {chunk_idx} exception: {str(e)}")
        
        parallel_time = time.time() - parallel_start
        print(f"Parallel generation completed in {parallel_time:.2f}s")
        
        # Check if any chunks failed
        if errors:
            raise PlannerGenerationError(
                f"Parallel generation failed: {'; '.join(errors)}",
                f"Could not generate the complete plan. Please try again with fewer days or simpler requirements."
            )
        
        # Assemble days in correct order
        all_days = []
        all_tags = set()
        first_summary = None
        first_difficulty = None
        first_completion_rate = None
        
        for idx in sorted(results.keys()):
            chunk_content = results[idx]
            all_days.extend(chunk_content.days)
            
            # Collect tags from all chunks
            if chunk_content.tags:
                all_tags.update(chunk_content.tags)
            
            if idx == 1:
                first_summary = chunk_content.summary
                first_difficulty = chunk_content.difficultyLevel
                first_completion_rate = chunk_content.estimatedCompletionRate

        if plan_outline:
            outline_summary = PlannerSummary(
                overview=plan_outline.overview,
                keyMilestones=plan_outline.key_milestones or None,
                tipsForSuccess=(first_summary.tipsForSuccess if first_summary else None),
                weeklyFocus=plan_outline.weekly_focus or None,
            )
            first_summary = outline_summary
        
        # Validate that we have the correct number of days
        if len(all_days) != req.totalDays:
            raise PlannerGenerationError(
                f"Chunked generation failed: Expected {req.totalDays} days but got {len(all_days)}",
                f"Could not generate the complete {req.totalDays}-day plan. Please try again."
            )
        
        # Validate and fix day numbering
        for i, day in enumerate(all_days):
            expected_day_num = i + 1
            if day.dayNumber != expected_day_num:
                day.dayNumber = expected_day_num
        
        # Create the final content with merged summary data
        final_content = PlannerContent(
            planName=req.planName,
            category=req.category,
            totalDays=req.totalDays,
            minutesPerDay=req.minutesPerDay,
            coverImage=None,
            coverImageUrl=None,
            createdAt={"seconds": now_s, "nanoseconds": 0},
            days=all_days,
            summary=first_summary,
            tags=list(all_tags) if all_tags else None,
            difficultyLevel=first_difficulty,
            estimatedCompletionRate=first_completion_rate
        )
        
        self._emit_progress(
            progress_callback,
            progress=92,
            progress_message="Finalizing your plan...",
            current_stage="finalizing",
            stages_completed=4,
        )

        total_time = time.time() - generation_start_time
        print(f"Total generation time: {total_time:.2f}s (context: {context_time:.2f}s, parallel gen: {parallel_time:.2f}s)")
        
        return final_content

    def _build_chunk_prompt(
        self,
        req: GeneratePlannerRequest,
        chunk: PlanChunk,
        chunk_idx: int,
        total_chunks: int,
        plan_outline: Optional[PlanOutline] = None,
    ) -> str:
        """Build an enhanced prompt for a specific chunk with progression context"""
        prompt_parts = []
        
        # Original user prompt (truncated if too long)
        if req.detailPrompt:
            original_prompt = req.detailPrompt[:200] + "..." if len(req.detailPrompt) > 200 else req.detailPrompt
            prompt_parts.append(f"User requirements: {original_prompt}")
        
        # Chunk context (concise)
        prompt_parts.append(f"Phase: {chunk.phase_name} (Days {chunk.start_day}-{chunk.end_day}/{req.totalDays})")
        prompt_parts.append(f"Level: {chunk.progression_level} | Focus: {chunk.focus_area}")
        
        # Key goals (first 2 only to save space)
        goals_text = ", ".join(chunk.key_goals[:2])
        prompt_parts.append(f"Goals: {goals_text}")
        
        # Progression context (concise)
        if chunk_idx > 1:
            prompt_parts.append(f"Builds upon previous {chunk_idx - 1} phase(s) - ensure continuity")
        
        if chunk_idx < total_chunks:
            prompt_parts.append(f"Prepares for {total_chunks - chunk_idx} upcoming phase(s) - set foundations")
        
        # Special instructions (truncated)
        special_instructions = chunk.special_instructions[:300] + "..." if len(chunk.special_instructions) > 300 else chunk.special_instructions
        prompt_parts.append(f"Instructions: {special_instructions}")

        if plan_outline:
            phase_outline = next(
                (p for p in plan_outline.phases if p.start_day == chunk.start_day and p.end_day == chunk.end_day),
                None,
            )
            if not phase_outline:
                phase_outline = next(
                    (
                        p for p in plan_outline.phases
                        if p.start_day <= chunk.start_day <= p.end_day
                    ),
                    None,
                )
            if phase_outline:
                prompt_parts.append(f"Outline focus: {phase_outline.focus}")
                if phase_outline.goals:
                    prompt_parts.append("Outline goals: " + ", ".join(phase_outline.goals[:3]))
        
        # Quality requirements (concise)
        prompt_parts.append("Requirements: Unique daily content, logical progression, specific actionable tasks, variety in activities")
        
        full_prompt = " | ".join(prompt_parts)
        
        # Ensure we stay within the 1000 character limit
        if len(full_prompt) > 1000:
            # Truncate further if needed
            full_prompt = full_prompt[:997] + "..."
        
        return full_prompt

    def _compact_plan_snapshot(
        self,
        content: PlannerContent,
        day_start: Optional[int] = None,
        day_end: Optional[int] = None,
        *,
        aggressive: bool = False,
    ) -> Dict[str, Any]:
        """Compact plan for refinement prompts (token-efficient)."""
        summary_limit = 200 if aggressive else 400
        task_limit = 120 if aggressive else 220
        days_out = []
        for day in content.days:
            if day_start is not None and day.dayNumber < day_start:
                continue
            if day_end is not None and day.dayNumber > day_end:
                continue
            days_out.append({
                "dayNumber": day.dayNumber,
                "title": (day.title or "")[:120],
                "summary": (day.summary or "")[:summary_limit],
                "tasks": [
                    {
                        "text": (t.text or "")[:task_limit],
                        "duration_min": t.duration_min,
                    }
                    for t in (day.tasks or [])[:6]
                ],
            })
        summary = content.summary
        return {
            "planName": content.planName,
            "category": content.category,
            "totalDays": content.totalDays,
            "minutesPerDay": content.minutesPerDay,
            "difficultyLevel": content.difficultyLevel,
            "summary": {
                "overview": summary.overview if summary else None,
                "keyMilestones": summary.keyMilestones if summary else None,
                "weeklyFocus": summary.weeklyFocus if summary else None,
            } if summary else None,
            "days": days_out,
        }

    def _build_refinement_context_json(
        self,
        content: PlannerContent,
        day_start: Optional[int] = None,
        day_end: Optional[int] = None,
        max_chars: int = 100000,
    ) -> str:
        """Serialize compact draft for refinement; shrink if needed."""
        compact = self._compact_plan_snapshot(content, day_start, day_end, aggressive=False)
        payload = json.dumps(compact, ensure_ascii=False)
        if len(payload) <= max_chars:
            return payload
        compact = self._compact_plan_snapshot(content, day_start, day_end, aggressive=True)
        payload = json.dumps(compact, ensure_ascii=False)
        if len(payload) <= max_chars:
            return payload
        return payload[: max_chars - 24] + "\n/* draft truncated */"

    def refine_plan(
        self,
        req: RefinePlannerRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PlannerContent:
        """Refine an existing draft plan from user feedback (full plan or day range)."""
        existing = PlannerContent.model_validate(req.existingContent)
        created_at = existing.createdAt

        self._emit_progress(
            progress_callback,
            progress=15,
            progress_message="Applying your changes...",
            current_stage="refining",
            stages_completed=1,
        )

        partial = req.refineDayStart is not None and req.refineDayEnd is not None
        if partial:
            start, end = req.refineDayStart, req.refineDayEnd
            scope_note = f"Only regenerate days {start} through {end}. Other days stay unchanged in the final merged plan."
            slice_days = end - start + 1
        else:
            scope_note = "Update the full plan as needed."
            start, end, slice_days = 1, existing.totalDays, existing.totalDays

        lang_note = "Write in Thai." if req.language == "th" else "Write in English."
        draft_json = self._build_refinement_context_json(
            existing,
            day_start=start if partial else None,
            day_end=end if partial else None,
        )
        refine_detail = (
            f"{lang_note}\n"
            f"REFINEMENT REQUEST:\n{req.refinementPrompt}\n\n"
            f"{scope_note}\n"
            f"Original plan had {existing.totalDays} days.\n"
            "Apply the refinement. Keep logical progression and the user's goals.\n"
            "The current draft JSON is provided in refinementContext."
        )

        gen_req = GeneratePlannerRequest(
            planName=req.planName,
            category=req.category,
            totalDays=slice_days if partial else existing.totalDays,
            detailPrompt=refine_detail[:1000],
            refinementContext=draft_json,
            minutesPerDay=req.minutesPerDay or existing.minutesPerDay,
            intensity=req.intensity,
            language=req.language,
            fastMode=req.fastMode,
            skipContextExtraction=True,
        )

        self._emit_progress(
            progress_callback,
            progress=40,
            progress_message="Updating plan content...",
            current_stage="generating_days",
            stages_completed=2,
        )

        if partial:
            slice_content = self.generate_single(
                gen_req,
                progress_callback=progress_callback,
                is_refinement=True,
            )
            day_by_num = {d.dayNumber: d for d in existing.days}
            for d in slice_content.days:
                target_num = start + (d.dayNumber - 1)
                d.dayNumber = target_num
                d.id = d.id or uuid.uuid4().hex[:8]
                day_by_num[target_num] = d
            merged_days = [day_by_num[i] for i in range(1, existing.totalDays + 1) if i in day_by_num]
            result = PlannerContent(
                planName=existing.planName,
                category=existing.category,
                totalDays=existing.totalDays,
                minutesPerDay=req.minutesPerDay or existing.minutesPerDay,
                coverImage=existing.coverImage,
                coverImageUrl=existing.coverImageUrl,
                createdAt=created_at,
                days=merged_days,
                summary=slice_content.summary or existing.summary,
                tags=slice_content.tags or existing.tags,
                difficultyLevel=slice_content.difficultyLevel or existing.difficultyLevel,
                estimatedCompletionRate=slice_content.estimatedCompletionRate or existing.estimatedCompletionRate,
            )
        else:
            result = self.generate_single(
                gen_req,
                progress_callback=progress_callback,
                is_refinement=True,
            )
            result.createdAt = created_at

        self._emit_progress(
            progress_callback,
            progress=95,
            progress_message="Finalizing updates...",
            current_stage="finalizing",
            stages_completed=3,
        )
        return result

    def generate(
        self,
        req: GeneratePlannerRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PlannerContent:
        """Main generation method with intelligent routing"""
        self._emit_progress(
            progress_callback,
            progress=5,
            progress_message="Starting generation...",
            current_stage="initializing",
            stages_completed=0,
        )
        if req.totalDays > 7:
            return self.generate_chunked(req, progress_callback=progress_callback)
        return self.generate_single(req, progress_callback=progress_callback)

    def generate_single(
        self,
        req: GeneratePlannerRequest,
        extracted_context: Optional[ExtractedUserContext] = None,
        progress_callback: Optional[ProgressCallback] = None,
        plan_outline: Optional[PlanOutline] = None,
        is_refinement: bool = False,
    ) -> PlannerContent:
        """
        Generate planner content with optional pre-extracted context.
        
        Args:
            req: The generation request
            extracted_context: Pre-extracted context (if None, will extract from detailPrompt)
        """
        now_s = int(time.time())
        
        # Determine model and temperature based on fast mode
        use_model = self.config.fast_model if req.fastMode else self.config.model
        use_temperature = self.config.fast_temperature if req.fastMode else self.config.temperature
        
        if req.fastMode:
            print(f"Fast mode enabled: using {use_model}")
        
        # Stage 1: Extract structured context from detailPrompt if not already provided
        # Skip if fastMode + skipContextExtraction is enabled
        if extracted_context is None and req.detailPrompt and not req.skipContextExtraction:
            print(f"Extracting user context from detailPrompt...")
            context_extractor = ContextExtractor(
                model=self.config.extraction_model,
                temperature=self.config.extraction_temperature
            )
            extracted_context = context_extractor.extract_context(
                detail_prompt=req.detailPrompt,
                category=req.category,
                plan_name=req.planName
            )
            if extracted_context:
                print(f"Context extraction successful. Primary goal: {extracted_context.goals.primary_goal}")
            else:
                print("Context extraction returned None, proceeding without structured context")
        elif req.skipContextExtraction:
            print("Context extraction skipped (skipContextExtraction=true)")

        if not is_refinement and plan_outline is None and not req.skipContextExtraction and req.detailPrompt:
            self._emit_progress(
                progress_callback,
                progress=18,
                progress_message="Understanding your goals...",
                current_stage="extracting_context",
                stages_completed=1,
            )
            plan_outline = self.generate_outline(req, extracted_context, progress_callback)

        self._emit_progress(
            progress_callback,
            progress=40,
            progress_message=(
                f"Updating your {req.totalDays}-day plan..."
                if is_refinement
                else f"Generating your {req.totalDays}-day plan..."
            ),
            current_stage="generating_days",
            stages_completed=3,
        )
        
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

        # Build the json schema for the exact PlannerContent shape with summary fields
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
                    # New summary fields
                    "summary": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "properties": {
                            "overview": {"type": ["string", "null"], "description": "Brief overview of the plan (2-3 sentences)"},
                            #"targetAudience": {"type": ["string", "null"], "description": "Who this plan is best suited for"},
                            #"expectedOutcomes": {"type": ["array", "null"], "items": {"type": "string"}, "description": "3-5 expected outcomes"},
                            "keyMilestones": {"type": ["array", "null"], "items": {"type": "string"}, "description": "3-5 key milestones"},
                            #"difficultyProgression": {"type": ["string", "null"], "description": "How difficulty changes"},
                            #"totalEstimatedHours": {"type": ["number", "null"], "description": "Total hours to complete"},
                            #"prerequisites": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Prerequisites if any"},
                            "tipsForSuccess": {"type": ["array", "null"], "items": {"type": "string"}, "description": "3-5 tips for success"},
                            "weeklyFocus": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Focus for each week"}
                        },
                        "required": ["overview", "keyMilestones", "tipsForSuccess", "weeklyFocus"]
                    },
                    "tags": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Relevant tags for the plan"},
                    "difficultyLevel": {"type": ["string", "null"], "enum": ["beginner", "intermediate", "advanced", "mixed", None], "description": "Overall difficulty"},
                    "estimatedCompletionRate": {"type": ["string", "null"], "description": "Expected completion rate"},
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 90,
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
                                "tips": {"type": ["string", "array", "null"], "items": {"type": "string"}}
                            },
                            "required": ["id", "dayNumber", "title", "summary", "tasks"]
                        }
                    }
                },
                "required": ["planName", "category", "totalDays", "createdAt", "days", "summary", "tags", "difficultyLevel", "estimatedCompletionRate"]
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

        # Build the user message with extracted context
        user_msg_parts = [
            lang_note,
            f"Category: {req.category}",
            f"Plan name: {req.planName}",
            f"Total days: {req.totalDays}",
        ]
        
        # Add basic parameters
        if req.minutesPerDay:
            user_msg_parts.append(f"Minutes per day: {req.minutesPerDay}")
        if req.intensity:
            user_msg_parts.append(f"Intensity: {req.intensity}")
        if req.startDate:
            user_msg_parts.append(f"Preferred start date: {req.startDate}")
        if req.timeOfDay:
            user_msg_parts.append(f"Preferred time of day: {req.timeOfDay}")
        
        # Add original detail prompt for reference
        if req.detailPrompt:
            if is_refinement:
                user_msg_parts.append(f"\nRefinement instructions:\n{req.detailPrompt}")
            else:
                user_msg_parts.append(f"\nOriginal user description:\n{req.detailPrompt}")

        if is_refinement and req.refinementContext:
            user_msg_parts.append(
                f"\n=== CURRENT DRAFT PLAN (JSON) ===\n{req.refinementContext}\n=== END DRAFT ==="
            )
        
        # Add extracted context as structured information
        if extracted_context:
            user_msg_parts.append("\n=== EXTRACTED USER PROFILE (USE THIS FOR PERSONALIZATION) ===")
            
            # Profile
            if extracted_context.profile:
                profile = extracted_context.profile
                if profile.experience_level:
                    user_msg_parts.append(f"Experience Level: {profile.experience_level}")
                if profile.age_group:
                    user_msg_parts.append(f"Age Group: {profile.age_group}")
                if profile.physical_limitations:
                    user_msg_parts.append(f"Physical Limitations: {', '.join(profile.physical_limitations)}")
                if profile.available_resources:
                    user_msg_parts.append(f"Available Resources: {', '.join(profile.available_resources)}")
                if profile.location:
                    user_msg_parts.append(f"Location: {profile.location}")
            
            # Goals
            if extracted_context.goals:
                goals = extracted_context.goals
                if goals.primary_goal:
                    user_msg_parts.append(f"\nPrimary Goal: {goals.primary_goal}")
                if goals.secondary_goals:
                    user_msg_parts.append(f"Secondary Goals: {', '.join(goals.secondary_goals)}")
                if goals.target_outcome:
                    user_msg_parts.append(f"Target Outcome: {goals.target_outcome}")
                if goals.deadline:
                    user_msg_parts.append(f"Deadline: {goals.deadline}")
                if goals.motivation_type:
                    user_msg_parts.append(f"Motivation Type: {goals.motivation_type}")
            
            # Constraints
            if extracted_context.constraints:
                constraints = extracted_context.constraints
                if constraints.budget_level:
                    user_msg_parts.append(f"\nBudget Level: {constraints.budget_level}")
                if constraints.time_constraints:
                    user_msg_parts.append(f"Time Constraints: {constraints.time_constraints}")
                if constraints.excluded_activities:
                    user_msg_parts.append(f"EXCLUDE These Activities: {', '.join(constraints.excluded_activities)}")
                if constraints.preferred_activities:
                    user_msg_parts.append(f"PREFER These Activities: {', '.join(constraints.preferred_activities)}")
                if constraints.rest_requirements:
                    user_msg_parts.append(f"Rest Requirements: {constraints.rest_requirements}")
            
            # Learning style
            if extracted_context.learning_style:
                ls = extracted_context.learning_style
                if ls.learning_style:
                    user_msg_parts.append(f"\nLearning Style: {ls.learning_style}")
                if ls.pace_preference:
                    user_msg_parts.append(f"Pace Preference: {ls.pace_preference}")
                if ls.feedback_preference:
                    user_msg_parts.append(f"Feedback Preference: {ls.feedback_preference}")
            
            # Category-specific details
            if extracted_context.category_specific:
                user_msg_parts.append(f"\nCategory-Specific Details:")
                for key, value in extracted_context.category_specific.items():
                    if value:
                        if isinstance(value, list):
                            user_msg_parts.append(f"  - {key}: {', '.join(value)}")
                        else:
                            user_msg_parts.append(f"  - {key}: {value}")
            
            # Key requirements
            if extracted_context.key_requirements:
                user_msg_parts.append(f"\nKey Requirements:")
                for req_item in extracted_context.key_requirements:
                    user_msg_parts.append(f"  • {req_item}")
            
            # Tone preference
            if extracted_context.tone_preference:
                user_msg_parts.append(f"\nTone: {extracted_context.tone_preference}")
            
            # Special considerations
            if extracted_context.special_considerations:
                user_msg_parts.append(f"\nSpecial Considerations:")
                for consideration in extracted_context.special_considerations:
                    user_msg_parts.append(f"  ⚠️ {consideration}")
            
            user_msg_parts.append("=== END EXTRACTED CONTEXT ===")

        outline_section = self._outline_to_prompt_section(plan_outline)
        if outline_section:
            user_msg_parts.append(outline_section)
        
        # Add category hints
        user_msg_parts.extend([
            "",
            category_hints[req.category],
            "",
            "Output a JSON object that strictly matches the provided schema. "
            "Use short, punchy titles; concise summaries; and 3–6 actionable tasks per day. "
            "IMPORTANT: Personalize all content based on the extracted user context above."
        ])
        
        # Add specific guidance for minutesPerDay
        if req.minutesPerDay:
            user_msg_parts.extend([
                "",
                f"TIME ALLOCATION GUIDELINE: Aim for approximately {req.minutesPerDay} minutes per day total, but allocate time to each task based on its natural requirements and complexity. "
                f"Each task should have a duration_min value that reflects how long it realistically takes to complete. "
                f"Flexibility is preferred over rigid matching."
            ])
        else:
            user_msg_parts.append("Task durations are optional. If you include durations, base them on the natural time requirements of each task.")
        
        user_msg = "\n".join(user_msg_parts)

        # Build personalized system prompt using extracted context
        system_prompt = self._build_system_prompt(
            req.category, extracted_context, refinement_mode=is_refinement
        )
        
        # Response format with JSON schema enforcement
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = get_openai_client().chat.completions.create(
                    model=use_model,  # Uses fast_model or model based on fastMode
                    temperature=use_temperature,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
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
                self._handle_generation_failure(req, "Empty response from OpenAI API")
            else:
                raw = response.choices[0].message.content
                print(f"DEBUG: Raw AI response: {raw[:500]}...")  # Log first 500 chars
                
                # Try to clean and parse the JSON response
                data = self._parse_json_response(raw)
                
                if not isinstance(data, dict):
                    self._handle_generation_failure(req, f"Invalid response type: {type(data)}")
                else:
                    print(f"DEBUG: Parsed data keys: {list(data.keys())}")  # Log available keys
                    
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")
            print(f"DEBUG: Raw response that failed to parse: {raw}")
            self._handle_generation_failure(req, f"JSON parsing error: {str(e)}")
        except Exception as e:
            print(f"DEBUG: Unexpected error parsing response: {e}")
            print(f"DEBUG: Raw response: {raw}")
            self._handle_generation_failure(req, f"Response parsing error: {str(e)}")

        # Fill in createdAt if model left null, and ensure ids
        try:
            seconds = payload["unix_now"]
            data.setdefault("createdAt", {"seconds": seconds, "nanoseconds": 0})
            
            # Ensure minutesPerDay is included in response from request
            if "minutesPerDay" not in data and req.minutesPerDay is not None:
                data["minutesPerDay"] = req.minutesPerDay
            
            if "days" not in data or not isinstance(data.get("days"), list):
                available_keys = list(data.keys()) if isinstance(data, dict) else "not a dict"
                print(f"Warning: AI response missing 'days' field. Available keys: {available_keys}")
                self._handle_generation_failure(req, f"Missing 'days' field in AI response. Available keys: {available_keys}")
            
            current_days = len(data.get("days", []))
            if current_days != req.totalDays:
                # Day count mismatch - this should not happen with proper AI generation
                self._handle_generation_failure(req, f"Day count mismatch: generated {current_days} days instead of {req.totalDays}")
            
            # No need to check for duplicate links since we're not using external links
            
            for i, d in enumerate(data.get("days", []), start=1):
                if not isinstance(d, dict):
                    raise PlannerGenerationError(
                        f"Invalid day format at index {i}",
                        "The generated plan has invalid day data. Please try again."
                    )
                d.setdefault("id", uuid.uuid4().hex[:8])
                # Ensure dayNumber is correct and sequential
                expected_day_num = i
                if d.get("dayNumber") != expected_day_num:
                    print(f"Warning: Day {i} has incorrect dayNumber {d.get('dayNumber')}, correcting to {expected_day_num}")
                    d["dayNumber"] = expected_day_num
                else:
                    d.setdefault("dayNumber", expected_day_num)
                
                # Convert tips from list to string if needed
                if "tips" in d and isinstance(d["tips"], list):
                    d["tips"] = '\n• '.join(d["tips"]) if d["tips"] else None
                
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
                    
                    # Set link field to None since we're not using external links
                    t["link"] = None
                    
                    # Validate task text quality - ensure it's detailed and actionable
                    task_text = t.get("text", "")
                    if not task_text or len(task_text.strip()) < 20:
                        # If task is too short or vague, provide a more detailed version
                        t["text"] = self._enhance_task_description(task_text, req.category)
                
                # Validate and fill in missing durations - flexible approach
                if req.minutesPerDay:
                    total_duration = 0
                    tasks_without_duration = []
                    
                    for j, task in enumerate(d.get("tasks", [])):
                        if task.get("duration_min") is None:
                            tasks_without_duration.append(j + 1)
                        else:
                            total_duration += task["duration_min"]
                    
                    # Only auto-assign durations to tasks that are missing them
                    # Use a flexible estimate based on remaining time and task count
                    if tasks_without_duration:
                        # Calculate reasonable average duration based on remaining time
                        # Use minutesPerDay as a rough guide, but don't force exact matching
                        remaining_minutes = max(0, req.minutesPerDay - total_duration)
                        tasks_needing_duration = len(tasks_without_duration)
                        
                        if tasks_needing_duration > 0:
                            if remaining_minutes > 0:
                                # Distribute remaining time proportionally
                                avg_duration = max(5, remaining_minutes // tasks_needing_duration)  # At least 5 min per task
                                remainder = remaining_minutes % tasks_needing_duration
                                
                                for idx, task_idx in enumerate(tasks_without_duration):
                                    duration = avg_duration + (1 if idx < remainder else 0)
                                    d["tasks"][task_idx - 1]["duration_min"] = duration
                            else:
                                # If we've exceeded the guideline, give minimum durations to tasks without them
                                for task_idx in tasks_without_duration:
                                    d["tasks"][task_idx - 1]["duration_min"] = 5  # Minimum 5 minutes
                            
                            print(f"Info: Day {i} had {len(tasks_without_duration)} task(s) without duration. Assigned reasonable durations.")
                    
                    # Calculate final total for logging only - don't force adjustment
                    final_total = sum(task.get("duration_min", 0) for task in d.get("tasks", []))
                    if final_total != req.minutesPerDay:
                        variance_percent = abs(final_total - req.minutesPerDay) / req.minutesPerDay * 100
                        if variance_percent <= 20:
                            print(f"Info: Day {i} total duration is {final_total} minutes (target: {req.minutesPerDay}, variance: {variance_percent:.1f}%) - acceptable flexibility.")
                        else:
                            print(f"Note: Day {i} total duration is {final_total} minutes (target: {req.minutesPerDay}, variance: {variance_percent:.1f}%) - prioritizing task-appropriate durations over exact matching.")
        
        except PlannerGenerationError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise PlannerGenerationError(
                f"Error processing response data: {e}",
                "Failed to process the generated plan. Please try again."
            )

        # Ensure summary fields are properly structured
        if "summary" in data and isinstance(data["summary"], dict):
            # Convert summary dict to PlannerSummary if needed
            print(f"DEBUG: Summary data found: {list(data['summary'].keys())}")
        else:
            # Set default empty summary if not present
            data.setdefault("summary", None)
        
        # Ensure other summary-related fields have defaults
        data.setdefault("tags", None)
        data.setdefault("difficultyLevel", None)
        data.setdefault("estimatedCompletionRate", None)
        
        print(f"DEBUG: Final data keys before validation: {list(data.keys())}")

        # Validate with Pydantic (final gate)
        try:
            validated = PlannerContent(**data)
            return validated
        except ValidationError as ve:
            # Format validation errors
            errors = "; ".join([f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in ve.errors()])
            print(f"Warning: Pydantic validation error: {errors}")
            print("Attempting to fix validation issues and generate fallback plan...")
            
            # Try to fix common validation issues
            try:
                # Ensure all required fields are present
                data.setdefault("planName", req.planName)
                data.setdefault("category", req.category)
                data.setdefault("totalDays", req.totalDays)
                data.setdefault("createdAt", {"seconds": int(time.time()), "nanoseconds": 0})
                data.setdefault("summary", None)
                data.setdefault("tags", None)
                data.setdefault("difficultyLevel", None)
                data.setdefault("estimatedCompletionRate", None)
                
                # If days is still missing or invalid, fail with proper error
                if "days" not in data or not isinstance(data.get("days"), list) or len(data["days"]) == 0:
                    self._handle_generation_failure(req, "Days field is missing or invalid after validation fixes")
                
                # Try validation again
                validated = PlannerContent(**data)
                return validated
                
            except Exception as fix_error:
                print(f"Could not fix validation issues: {fix_error}")
                self._handle_generation_failure(req, f"Validation fix failed: {str(fix_error)}")


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

# Firebase Cloud Function decorator - conditionally applied
def _firebase_decorator(func):
    """Apply Firebase decorator only if Firebase is available"""
    if FIREBASE_AVAILABLE:
        return https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540)(func)
    return func

@_firebase_decorator
def generate_planner_content(req):
    """Main HTTP handler for planner generation"""
    # Handle both Firebase Request and Flask request objects
    if FIREBASE_AVAILABLE:
        origin = req.headers.get("Origin")
    else:
        origin = req.headers.get("Origin") if hasattr(req, 'headers') else None
    
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
        
        # Validate request size and complexity to prevent timeouts
        if len(str(payload)) > 10000:  # 10KB limit for request payload
            return https_fn.Response(
                json.dumps({
                    "error": "Request too large",
                    "message": "Request payload is too large. Please simplify your requirements."
                }),
                status=400,
                headers={**_cors_headers(origin), "Content-Type": "application/json"}
            )
        
        parsed = GeneratePlannerRequest(**payload)
        
        # Additional validation for large plans that might cause timeouts
        if parsed.totalDays > 60:
            return https_fn.Response(
                json.dumps({
                    "error": "Plan too large",
                    "message": f"Plans with {parsed.totalDays} days may take too long to generate. Please try with 60 days or fewer."
                }),
                status=400,
                headers={**_cors_headers(origin), "Content-Type": "application/json"}
            )
        
        print(f"Processing {parsed.totalDays}-day {parsed.category} plan...")
        start_time = time.time()
        
        content = chat.generate(parsed)
        
        generation_time = time.time() - start_time
        print(f"Generated {parsed.totalDays}-day plan in {generation_time:.2f} seconds")
        
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