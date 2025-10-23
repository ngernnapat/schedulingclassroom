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

    def _generate_fallback_plan(self, req: GeneratePlannerRequest) -> dict:
        """Generate a basic fallback plan when AI response fails"""
        print(f"Generating fallback plan for {req.totalDays} days, category: {req.category}")
        
        fallback_days = []
        for day_num in range(1, req.totalDays + 1):
            # Create category-specific content
            if req.category == "learning":
                title = f"Day {day_num} - Learning Session"
                summary = "Continue your learning journey with focused study and practice."
                task1_text = "Study session: Review previous material and learn new concepts. Take notes and practice with examples."
                task2_text = "Practice and apply: Work on exercises or projects to reinforce your learning."
            elif req.category == "exercise":
                title = f"Day {day_num} - Workout Session"
                summary = "Maintain your fitness routine with a balanced workout."
                task1_text = "Warm-up and main exercise: Start with 5-10 minutes of light activity, then perform your main workout."
                task2_text = "Cool-down and recovery: Finish with stretching and light movement to aid recovery."
            elif req.category == "finance":
                title = f"Day {day_num} - Financial Planning"
                summary = "Continue building your financial knowledge and habits."
                task1_text = "Financial review: Check your accounts, review recent transactions, and update your budget."
                task2_text = "Learning and planning: Study financial concepts or plan your next financial goals."
            elif req.category == "health":
                title = f"Day {day_num} - Wellness Focus"
                summary = "Prioritize your health and well-being today."
                task1_text = "Health check: Monitor your vital signs, review your nutrition, and plan healthy meals."
                task2_text = "Wellness activity: Engage in stress management, meditation, or other wellness practices."
            elif req.category == "personal_development":
                title = f"Day {day_num} - Growth Session"
                summary = "Focus on personal growth and self-improvement."
                task1_text = "Self-reflection: Journal about your goals, progress, and areas for improvement."
                task2_text = "Skill development: Work on a specific skill or habit that contributes to your growth."
            elif req.category == "travel":
                title = f"Day {day_num} - Travel Planning"
                summary = "Continue planning and preparing for your travel goals."
                task1_text = "Research and planning: Research destinations, activities, or logistics for your trip."
                task2_text = "Preparation: Work on bookings, packing lists, or other travel preparations."
            else:
                title = f"Day {day_num} - Practice Session"
                summary = "Continue working on your goals with focused practice."
                task1_text = "Main activity: Spend time on your primary goal or skill development."
                task2_text = "Review and plan: Reflect on your progress and plan your next steps."
            
            fallback_day = {
                "id": uuid.uuid4().hex[:8],
                "dayNumber": day_num,
                "title": title,
                "summary": summary,
                "tasks": [
                    {
                        "id": uuid.uuid4().hex[:8],
                        "text": task1_text,
                        "done": False,
                        "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                        "note": "Customize this task based on your specific needs and progress",
                        "link": None
                    },
                    {
                        "id": uuid.uuid4().hex[:8],
                        "text": task2_text,
                        "done": False,
                        "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                        "note": "Adapt this task to your current level and goals",
                        "link": None
                    }
                ],
                "tips": "Stay consistent with your practice and adjust your approach based on what works best for you."
            }
            fallback_days.append(fallback_day)
        
        return {
            "planName": req.planName,
            "category": req.category,
            "totalDays": req.totalDays,
            "minutesPerDay": req.minutesPerDay,
            "coverImage": None,
            "coverImageUrl": None,
            "createdAt": {"seconds": int(time.time()), "nanoseconds": 0},
            "days": fallback_days,
            "warning": "Generated a basic plan structure due to AI response issues. You can customize the tasks as needed."
        }

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
            "1) Keep each day practical (2-4 tasks, which may include details from the detailPrompt). 2) Add brief tips when helpful. 3) Titles are short and motivating.\n"
            "4) Never invent unsafe or extreme advice; prefer safe defaults.\n"
            "5) CRITICAL: Output MUST be valid JSON matching the exact schema provided.\n"
            "6) Include ALL required fields: planName, category, totalDays, createdAt, days.\n"
            "7) ABSOLUTE REQUIREMENT: The 'days' array MUST contain EXACTLY the number of days specified in totalDays. If totalDays=30, you MUST generate exactly 30 days. If totalDays=7, you MUST generate exactly 7 days. NO MORE, NO LESS.\n"
            "8) TIME ALLOCATION: If minutesPerDay is specified, you MUST ensure that the sum of all task durations (duration_min) for each day equals exactly the specified minutesPerDay. Each task must have a duration_min value when minutesPerDay is provided.\n"
            "9) DAY NUMBERING: Each day must have a dayNumber field starting from 1 and incrementing sequentially (1, 2, 3, ..., totalDays).\n"
            "10) DETAILED TASK INSTRUCTIONS: Each task MUST include comprehensive, actionable instructions that users can follow without external links. Focus on providing clear, step-by-step guidance within the task description itself.\n"
            "   TASK QUALITY REQUIREMENTS:\n"
            "   ✓ Provide specific, actionable steps for each task\n"
            "   ✓ Include relevant tips, techniques, or methods\n"
            "   ✓ Give clear success criteria or what to expect\n"
            "   ✓ Include safety considerations where applicable\n"
            "   ✓ Make tasks self-contained and complete\n"
            "   ✓ Use the 'note' field for additional helpful details\n"
            "   \n"
            "   TASK DESCRIPTION EXAMPLES:\n"
            "   ✅ GOOD: 'Practice Python variables: Create 5 different variable types (string, integer, float, boolean, list). Write a simple program that uses each type and prints the results. Focus on proper naming conventions and data type understanding.'\n"
            "   ✅ GOOD: 'Morning cardio workout: Do 20 minutes of moderate-intensity exercise (brisk walking, jogging, or cycling). Start with 5-minute warm-up, maintain steady pace for 15 minutes, finish with 5-minute cool-down. Monitor your heart rate and stay hydrated.'\n"
            "   ❌ BAD: 'Learn Python' (too vague)\n"
            "   ❌ BAD: 'Do some exercise' (not specific enough)\n"
           
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
                # Fix the day numbering: chunk_start is the correct starting day number
                # day.dayNumber from the chunk should be 1, 2, 3, etc. for the chunk
                # We need to map it to the global day number
                day.dayNumber = chunk_start + (day.dayNumber - 1)
                all_days.append(day)
            
            # Add small delay between chunks to avoid rate limits
            if chunk_end < req.totalDays:
                time.sleep(1)
        
        # Validate that we have the correct number of days
        if len(all_days) != req.totalDays:
            raise PlannerGenerationError(
                f"Chunked generation failed: Expected {req.totalDays} days but got {len(all_days)}",
                f"Could not generate the complete {req.totalDays}-day plan. Please try again with fewer days or simpler requirements."
            )
        
        # Validate day numbering is sequential
        for i, day in enumerate(all_days):
            expected_day_num = i + 1
            if day.dayNumber != expected_day_num:
                print(f"Warning: Day {i+1} has incorrect dayNumber {day.dayNumber}, correcting to {expected_day_num}")
                day.dayNumber = expected_day_num
        
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
                print("Warning: Empty response from OpenAI, generating fallback plan...")
                # Generate fallback plan instead of raising error
                data = self._generate_fallback_plan(req)
            else:
                raw = response.choices[0].message.content
                print(f"DEBUG: Raw AI response: {raw[:500]}...")  # Log first 500 chars
                
                # Try to clean and parse the JSON response
                data = self._parse_json_response(raw)
                
                if not isinstance(data, dict):
                    print(f"Warning: Invalid response type: {type(data)}, generating fallback plan...")
                    data = self._generate_fallback_plan(req)
                else:
                    print(f"DEBUG: Parsed data keys: {list(data.keys())}")  # Log available keys
                    
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")
            print(f"DEBUG: Raw response that failed to parse: {raw}")
            print("Generating fallback plan due to JSON parsing error...")
            data = self._generate_fallback_plan(req)
        except Exception as e:
            print(f"DEBUG: Unexpected error parsing response: {e}")
            print(f"DEBUG: Raw response: {raw}")
            print("Generating fallback plan due to parsing error...")
            data = self._generate_fallback_plan(req)

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
                print("Generating fallback plan with basic structure...")
                
                # Generate a fallback plan with the requested number of days
                fallback_days = []
                for day_num in range(1, req.totalDays + 1):
                    fallback_day = {
                        "id": uuid.uuid4().hex[:8],
                        "dayNumber": day_num,
                        "title": f"Day {day_num} - {req.category.title()} Practice",
                        "summary": f"Continue your {req.category} journey with focused practice and learning.",
                        "tasks": [
                            {
                                "id": uuid.uuid4().hex[:8],
                                "text": f"Practice session: Spend time working on your {req.category} skills. Focus on consistency and gradual improvement.",
                                "done": False,
                                "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                                "note": "Adapt this task based on your specific needs and progress",
                                "link": None
                            },
                            {
                                "id": uuid.uuid4().hex[:8],
                                "text": f"Review and reflect: Take time to review your progress, identify areas for improvement, and plan your next steps.",
                                "done": False,
                                "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                                "note": "Use this time to assess your learning and adjust your approach",
                                "link": None
                            }
                        ],
                        "tips": "Stay consistent with your practice and don't be afraid to adjust your approach based on what you've learned."
                    }
                    fallback_days.append(fallback_day)
                
                # Update the data with fallback content
                data["days"] = fallback_days
                data["planName"] = data.get("planName", req.planName)
                data["category"] = data.get("category", req.category)
                data["totalDays"] = req.totalDays
                data["warning"] = "Generated a basic plan structure due to AI response format issues. You can customize the tasks as needed."
                
                print(f"Generated fallback plan with {len(fallback_days)} days")
            
            current_days = len(data.get("days", []))
            if current_days != req.totalDays:
                # Day count mismatch - this should not happen with proper AI generation
                # Try to fix it by generating additional days or trimming excess
                if current_days < req.totalDays:
                    # Generate additional days to match the request
                    missing_days = req.totalDays - current_days
                    print(f"Warning: Generated only {current_days} days instead of {req.totalDays}. Attempting to generate {missing_days} additional days.")
                    
                    # Create additional days with generic content
                    for day_num in range(current_days + 1, req.totalDays + 1):
                        additional_day = {
                            "id": uuid.uuid4().hex[:8],
                            "dayNumber": day_num,
                            "title": f"Day {day_num} - Continued Progress",
                            "summary": f"Continue your {req.category} journey with focused practice and reflection.",
                            "tasks": [
                                {
                                    "id": uuid.uuid4().hex[:8],
                                    "text": f"Review and practice: Spend time reviewing previous days' progress and continue building your skills. Focus on consistency and gradual improvement.",
                                    "done": False,
                                    "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                                    "note": "Adapt this task based on your specific needs and progress",
                                    "link": None
                                },
                                {
                                    "id": uuid.uuid4().hex[:8],
                                    "text": f"Reflection and planning: Take time to reflect on your progress, identify areas for improvement, and plan your next steps.",
                                    "done": False,
                                    "duration_min": req.minutesPerDay // 2 if req.minutesPerDay else None,
                                    "note": "Use this time to assess your learning and adjust your approach",
                                    "link": None
                                }
                            ],
                            "tips": "Stay consistent with your practice and don't be afraid to adjust your approach based on what you've learned."
                        }
                        data["days"].append(additional_day)
                    
                    print(f"Generated {missing_days} additional days to complete the {req.totalDays}-day plan.")
                    
                elif current_days > req.totalDays:
                    # Trim excess days
                    excess_days = current_days - req.totalDays
                    print(f"Warning: Generated {current_days} days instead of {req.totalDays}. Trimming {excess_days} excess days.")
                    data["days"] = data["days"][:req.totalDays]
                
                # Ensure totalDays matches the actual number of days
                data["totalDays"] = len(data["days"])
                
                # Add warning to the response data
                data["warning"] = f"Day count was adjusted from {current_days} to {req.totalDays} to match your request."
            
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
            print(f"Warning: Pydantic validation error: {errors}")
            print("Attempting to fix validation issues and generate fallback plan...")
            
            # Try to fix common validation issues
            try:
                # Ensure all required fields are present
                data.setdefault("planName", req.planName)
                data.setdefault("category", req.category)
                data.setdefault("totalDays", req.totalDays)
                data.setdefault("createdAt", {"seconds": int(time.time()), "nanoseconds": 0})
                
                # If days is still missing or invalid, use fallback
                if "days" not in data or not isinstance(data.get("days"), list) or len(data["days"]) == 0:
                    print("Days field is missing or invalid, using fallback generation...")
                    fallback_data = self._generate_fallback_plan(req)
                    data.update(fallback_data)
                
                # Try validation again
                validated = PlannerContent(**data)
                return validated
                
            except Exception as fix_error:
                print(f"Could not fix validation issues: {fix_error}")
                # Last resort: return a basic fallback plan
                fallback_data = self._generate_fallback_plan(req)
                validated = PlannerContent(**fallback_data)
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