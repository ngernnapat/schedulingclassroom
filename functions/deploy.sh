#!/bin/bash

set -e

echo "🚀 Starting deployment of School Schedule Optimization Functions..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the functions directory."
    exit 1
fi


print_status "Cleaning up previous deployment artifacts..."
rm -rf __pycache__ *.pyc .pytest_cache
rm -f firebase-debug.log firebase-debug.*.log

if [ -d "venv" ]; then
    print_status "Removing existing virtual environment..."
    rm -rf venv
fi

print_status "Creating new virtual environment..."
python3.11 -m venv venv

print_status "Activating virtual environment..."
source venv/bin/activate

print_status "Upgrading pip..."
pip install --upgrade pip

print_status "Installing dependencies..."
pip install -r requirements.txt

print_status "Verifying common dependencies (excluding firebase_functions)..."
python -c "
import firebase_admin
import pandas
import pulp
import ortools
import plotly
import numpy
import requests
print('✅ Common dependencies installed successfully')
"
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

print_warning "⚠️ Skipping firebase_functions check (only works in Firebase runtime)"

print_status "Testing local import of health_check and get_schedule_info..."
python -c "
from main import health_check, get_schedule_info
print('✅ Functions imported successfully')
"

print_status "Deploying to Firebase..."
firebase deploy --only functions

print_success "Deployment completed successfully!"
nvm use 20
print_status "Retrieving function URLs..."
firebase functions:list

print_success "🎉 Deployment finished! Your functions are now live."
print_status "You can test the endpoints using:"
echo "  - Health check: curl https://school-schedule-optimization-functions.cloudfunctions.net/health_check"
echo "  - Schedule info: curl https://school-schedule-optimization-functions.cloudfunctions.net/get_schedule_info"
echo "  - Generate schedule: POST to https://school-schedule-optimization-functions.cloudfunctions.net/generate_schedule"