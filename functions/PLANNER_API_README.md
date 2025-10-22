# Planner Content Generation API

This is a local API server for testing the `generate_planner_content.py` function. It provides a FastAPI interface to generate personalized daily plans for various categories like learning, exercise, travel, finance, health, and personal development.

## 🚀 Quick Start

### 1. Set up Environment

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

You have several options to set your OpenAI API key:

**Option A: Use the setup script (Recommended)**
```bash
python setup_api_key.py
```

**Option B: Set environment variable**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Option C: Create .env file**
```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 3. Start the API Server

**Option A: Use the startup script**
```bash
./start_planner_api.sh
```

**Option B: Run directly**
```bash
python generate_planner_content_api.py
```

The server will start on `http://localhost:8000`

### 4. Demo Without API Key

If you want to see the API structure without setting up an API key:

```bash
python demo_without_api.py
```

### 5. Access API Documentation

Open your browser and go to:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📋 Available Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `GET /categories` - List available plan categories
- `GET /examples` - Get example requests for testing
- `POST /generate` - Generate planner content (main endpoint)
- `POST /generate-raw` - Generate with raw JSON input
- `GET /test/{category}` - Quick test generation

### Example Usage

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Get Available Categories
```bash
curl http://localhost:8000/categories
```

#### 3. Quick Test Generation
```bash
curl "http://localhost:8000/test/learning?days=7&minutes=30"
```

#### 4. Full Generation Request
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "planName": "Python Learning Journey",
    "category": "learning",
    "totalDays": 7,
    "detailPrompt": "Learn Python programming from basics to intermediate",
    "minutesPerDay": 60,
    "language": "en"
  }'
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_planner_api.py
```

This will test all endpoints and show example outputs.

## 📊 Request Parameters

### Required Parameters
- `planName` (string): Name of the plan
- `category` (string): Type of planner content
- `totalDays` (integer): Number of days (1-90)

### Optional Parameters
- `detailPrompt` (string): User-specific requirements
- `minutesPerDay` (integer): Daily time allocation (10-480 minutes)
- `intensity` (string): Difficulty level ("easy", "moderate", "hard", "periodized")
- `language` (string): Output language ("en" or "th")
- `startDate` (string): Preferred start date (YYYY-MM-DD)
- `timeOfDay` (string): Preferred time ("morning", "afternoon", "evening", "flexible")

### Available Categories
- `learning` - Skill acquisition and knowledge development
- `exercise` - Physical fitness and health
- `travel` - Trip planning and itinerary
- `finance` - Financial management and literacy
- `health` - Holistic wellness and healthy habits
- `personal_development` - Self-improvement and growth
- `other` - Custom plan based on specific needs

## 📝 Example Requests

### Learning Plan
```json
{
  "planName": "Python Programming Basics",
  "category": "learning",
  "totalDays": 14,
  "detailPrompt": "Beginner Python programming with focus on data structures and algorithms",
  "minutesPerDay": 60,
  "intensity": "moderate",
  "language": "en"
}
```

### Exercise Plan
```json
{
  "planName": "30-Day Fitness Challenge",
  "category": "exercise",
  "totalDays": 30,
  "detailPrompt": "Home workout routine for beginners, no equipment needed",
  "minutesPerDay": 45,
  "intensity": "moderate",
  "language": "en"
}
```

### Travel Plan
```json
{
  "planName": "Thailand Adventure",
  "category": "travel",
  "totalDays": 14,
  "detailPrompt": "Backpacking trip through Bangkok, Chiang Mai, and Phuket",
  "language": "en"
}
```

### Thai Language Learning
```json
{
  "planName": "เรียนภาษาไทย 30 วัน",
  "category": "learning",
  "totalDays": 30,
  "detailPrompt": "Basic Thai language learning for English speakers",
  "minutesPerDay": 30,
  "language": "th"
}
```

## 🔧 Response Format

### Successful Response
```json
{
  "success": true,
  "data": {
    "planName": "Python Learning Journey",
    "category": "learning",
    "totalDays": 7,
    "minutesPerDay": 60,
    "createdAt": {
      "seconds": 1703123456,
      "nanoseconds": 0
    },
    "days": [
      {
        "id": "abc12345",
        "dayNumber": 1,
        "title": "Python Basics Introduction",
        "summary": "Get started with Python fundamentals",
        "tasks": [
          {
            "id": "task123",
            "text": "Install Python and set up development environment",
            "done": false,
            "duration_min": 20,
            "note": "Use Python 3.8 or later"
          }
        ],
        "tips": "Take breaks every 20 minutes to avoid fatigue"
      }
    ]
  },
  "message": "Successfully generated 7-day learning plan: Python Learning Journey"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Planner generation failed",
  "message": "User-friendly error message",
  "details": {
    "technical_details": "Technical error details"
  }
}
```

## 🛠️ Development

### Running in Development Mode
The server runs with auto-reload enabled by default. Any changes to the code will automatically restart the server.

### Adding New Features
1. Modify `generate_planner_content_api.py` to add new endpoints
2. Update the test script `test_planner_api.py` to test new features
3. Update this README with new endpoint documentation

### Debugging
- Check the console output for detailed error messages
- Use the `/health` endpoint to verify the server is running
- Check the interactive docs at `/docs` for request/response schemas

## 🔒 Security Notes

- The API runs with CORS enabled for all origins (suitable for local development)
- In production, restrict CORS origins to your actual domains
- The OpenAI API key should be kept secure and not exposed in client-side code
- Consider adding authentication/rate limiting for production use

## 📞 Support

If you encounter issues:
1. Check that the OpenAI API key is set correctly
2. Verify all dependencies are installed
3. Check the server logs for detailed error messages
4. Test with the provided test script to isolate issues
