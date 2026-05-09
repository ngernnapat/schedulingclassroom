#!/bin/bash

# Firebase Functions Deployment Script
# This script helps deploy the functions with proper dependency management

set -e  # Exit on any error

echo "🚀 Starting Firebase Functions Deployment..."

# Check if we're in the functions directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the functions directory"
    exit 1
fi

# Check if firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Error: Firebase CLI is not installed. Please install it first:"
    echo "npm install -g firebase-tools"
    exit 1
fi

# Check if we're logged in to Firebase
if ! firebase projects:list &> /dev/null; then
    echo "❌ Error: Not logged in to Firebase. Please run:"
    echo "firebase login"
    exit 1
fi

echo "📦 Installing/updating dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies with updated setuptools
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for pkg_resources deprecation warning
echo "🔍 Checking for deprecation warnings..."
python -c "
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

# Capture warnings
stderr = StringIO()
with redirect_stderr(stderr):
    import google
    import firebase_admin
    import openai

warnings_output = stderr.getvalue()
if 'pkg_resources is deprecated' in warnings_output:
    print('⚠️  pkg_resources deprecation warning detected')
    print('This is expected and will be resolved in future updates')
else:
    print('✅ No deprecation warnings detected')
"

echo "🧪 Running tests..."
if [ -f "test_planner_fix.py" ]; then
    python test_planner_fix.py
    if [ $? -eq 0 ]; then
        echo "✅ Tests passed!"
    else
        echo "❌ Tests failed! Please fix the issues before deploying."
        exit 1
    fi
else
    echo "⚠️  No test file found, skipping tests"
fi

echo "🚀 Deploying to Firebase Functions..."

# Deploy with specific configuration
firebase deploy --only functions

echo "✅ Deployment completed successfully!"

# Allow public access to all functions (Firebase Functions v2 run on Cloud Run)
echo ""
echo "🔓 Allowing public access to all functions..."
if command -v gcloud &> /dev/null; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "schedule-optimization-d83ea")
    REGION="us-central1"
    # List Cloud Run services (each deployed function is a service) and add allUsers invoker to each
    FAILED=0
    while IFS= read -r SERVICE; do
        [ -z "$SERVICE" ] && continue
        if gcloud run services add-iam-policy-binding "$SERVICE" \
            --region="${REGION}" \
            --member="allUsers" \
            --role="roles/run.invoker" \
            --project="${PROJECT_ID}" --quiet 2>/dev/null; then
            echo "   ✅ ${SERVICE}"
        else
            echo "   ⚠️  ${SERVICE} (skipped or already public)"
            FAILED=$((FAILED + 1))
        fi
    done < <(gcloud run services list --region="${REGION}" --project="${PROJECT_ID}" --format="value(metadata.name)" 2>/dev/null)
    if [ "$FAILED" -gt 0 ]; then
        echo "⚠️  Some services could not be updated (may already allow public access)."
    fi
    echo "✅ Public access step completed."
else
    echo "⚠️  gcloud CLI not found; skipping public access. Run allow_public_access.sh to enable."
fi

# Clean up
deactivate

echo ""
echo "🎉 All done! Your Firebase Functions are now deployed."
echo ""
echo "📝 Notes:"
echo "- The pkg_resources deprecation warning is expected and harmless"
echo "- It will be resolved in future updates of the Google Cloud libraries"
echo "- Your functions should work correctly despite the warning"