#!/bin/bash

# Start Planner Content Generation API
# This script sets up the environment and starts the API server

echo "🚀 Starting Planner Content Generation API..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "generate_planner_content_api.py" ]; then
    echo "❌ Error: generate_planner_content_api.py not found!"
    echo "Please run this script from the functions directory."
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY environment variable not set!"
    echo ""
    echo "To set up your API key, you can:"
    echo "1. Run the setup script: python setup_api_key.py"
    echo "2. Set environment variable: export OPENAI_API_KEY='your-api-key-here'"
    echo "3. Create a .env file with: OPENAI_API_KEY=your-api-key-here"
    echo ""
    echo "You can still start the server, but generation requests will fail."
    echo ""
    
    # Ask if user wants to run setup
    read -p "Do you want to run the API key setup now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔧 Running API key setup..."
        python setup_api_key.py
        echo ""
    fi
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi, uvicorn, openai, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Installing..."
    pip install fastapi uvicorn openai pydantic
fi

echo "✅ Dependencies OK"
echo ""

# Start the server
echo "🌐 Starting API server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo "📋 Examples: http://localhost:8000/examples"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python generate_planner_content_api.py
