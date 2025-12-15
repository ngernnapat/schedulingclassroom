#!/bin/bash

# Deployment script for optimized generate-planner-content service
# This script applies timeout and performance optimizations to prevent 504 errors

set -e

echo "🚀 Deploying optimized generate-planner-content service..."

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Get current project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID found. Please run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"

# Deploy the function with optimized settings
echo "🔧 Deploying with optimized configuration..."

# Deploy using gcloud functions deploy with optimized settings
gcloud functions deploy generate-planner-content \
    --runtime python312 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 2048MB \
    --timeout 540s \
    --max-instances 5 \
    --min-instances 1 \
    --cpu 2 \
    --gen2 \
    --source . \
    --entry-point generate_planner_content \
    --set-env-vars "PYTHONUNBUFFERED=1" \
    --region us-central1

echo "✅ Deployment completed successfully!"

# Test the deployment
echo "🧪 Testing the deployment..."

# Wait a moment for the service to be ready
sleep 10

# Test with a simple request
echo "📡 Testing with a simple 7-day plan..."

curl -X POST \
    "https://us-central1-$PROJECT_ID.cloudfunctions.net/generate-planner-content" \
    -H "Content-Type: application/json" \
    -d '{
        "planName": "Test Plan",
        "category": "learning",
        "totalDays": 7,
        "detailPrompt": "Basic test plan",
        "language": "en"
    }' \
    --max-time 60 \
    --connect-timeout 10

echo ""
echo "🎉 Deployment and test completed!"
echo ""
echo "📊 Optimizations applied:"
echo "  • Memory: 2048MB (increased from 1024MB)"
echo "  • Timeout: 540 seconds (9 minutes)"
echo "  • Max instances: 5 (increased from 3)"
echo "  • Min instances: 1 (reduces cold starts)"
echo "  • CPU: 2 cores"
echo "  • Request validation: Prevents overly large requests"
echo "  • Chunk optimization: Larger chunks, faster processing"
echo "  • Timeout monitoring: Prevents long-running requests"
echo ""
echo "🔍 Monitor the service with:"
echo "  gcloud functions logs read generate-planner-content --limit 50"
echo ""
echo "📈 Check metrics with:"
echo "  gcloud functions describe generate-planner-content"
