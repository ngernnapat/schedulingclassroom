#!/bin/bash

# Environment Validation Script for School Schedule Optimization Functions
# This script validates the deployment environment before running the main deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/deploy.config"

# Load configuration if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    # Default values if config file doesn't exist
    PYTHON_VERSION="3.11"
    NODE_VERSION="20"
    PROJECT_ID="school-schedule-optimization-functions"
fi

# Validation results
validation_results=()
validation_status=()

# Check command availability
check_command() {
    local cmd=$1
    local name=${2:-$1}
    
    if command -v "$cmd" &> /dev/null; then
        local version
        version=$($cmd --version 2>/dev/null || echo "unknown version")
        log_success "$name: $version"
        validation_results+=("$name")
        validation_status+=("OK")
        return 0
    else
        log_error "$name: Not found"
        validation_results+=("$name")
        validation_status+=("FAIL")
        return 1
    fi
}

# Check Python version
check_python_version() {
    log_step "Checking Python version..."
    
    if ! check_command "python3" "Python 3"; then
        return 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [[ "$python_version" == "$PYTHON_VERSION" ]]; then
        log_success "Python version matches requirement: $python_version"
        validation_results+=("Python Version")
        validation_status+=("OK")
    else
        log_warning "Python version $python_version found, but $PYTHON_VERSION is recommended"
        validation_results+=("Python Version")
        validation_status+=("WARN")
    fi
}

# Check Node.js and npm
check_node_environment() {
    log_step "Checking Node.js environment..."
    
    check_command "node" "Node.js"
    check_command "npm" "npm"
    
    if command -v nvm &> /dev/null; then
        log_info "nvm found - Node version manager available"
        validation_results+=("nvm")
        validation_status+=("OK")
    else
        log_warning "nvm not found - consider installing for better Node.js management"
        validation_results+=("nvm")
        validation_status+=("WARN")
    fi
}

# Check Firebase CLI
check_firebase_cli() {
    log_step "Checking Firebase CLI..."
    
    if check_command "firebase" "Firebase CLI"; then
        # Check if logged in
        if firebase projects:list &> /dev/null; then
            log_success "Firebase CLI authenticated"
            validation_results+=("Firebase Auth")
            validation_status+=("OK")
        else
            log_error "Firebase CLI not authenticated. Run: firebase login"
            validation_results+=("Firebase Auth")
            validation_status+=("FAIL")
            return 1
        fi
    else
        return 1
    fi
}

# Check project files
check_project_files() {
    log_step "Checking project files..."
    
    local required_files=(
        "main.py"
        "requirements.txt"
        "school_scheduler.py"
        "planner_utils.py"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$file" ]]; then
            log_success "Found $file"
            validation_results+=("$file")
            validation_status+=("OK")
        else
            log_error "Missing $file"
            validation_results+=("$file")
            validation_status+=("FAIL")
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing required files: ${missing_files[*]}"
        return 1
    fi
}

# Check Firebase configuration
check_firebase_config() {
    log_step "Checking Firebase configuration..."
    
    local project_root
    project_root=$(dirname "$SCRIPT_DIR")
    
    if [[ -f "$project_root/firebase.json" ]]; then
        log_success "Found firebase.json"
        validation_results+=("firebase.json")
        validation_status+=("OK")
    else
        log_error "Missing firebase.json in project root"
        validation_results+=("firebase.json")
        validation_status+=("FAIL")
        return 1
    fi
    
    if [[ -f "$project_root/.firebaserc" ]]; then
        log_success "Found .firebaserc"
        validation_results+=(".firebaserc")
        validation_status+=("OK")
    else
        log_warning "Missing .firebaserc - Firebase project may not be initialized"
        validation_results+=(".firebaserc")
        validation_status+=("WARN")
    fi
}

# Check system resources
check_system_resources() {
    log_step "Checking system resources..."
    
    # Check available disk space (at least 1GB)
    local available_space
    available_space=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    available_space_mb=$((available_space / 1024))
    
    if [[ $available_space_mb -gt 1024 ]]; then
        log_success "Available disk space: ${available_space_mb}MB"
        validation_results+=("Disk Space")
        validation_status+=("OK")
    else
        log_warning "Low disk space: ${available_space_mb}MB (recommend >1GB)"
        validation_results+=("Disk Space")
        validation_status+=("WARN")
    fi
    
    # Check available memory (at least 2GB)
    local total_memory
    total_memory=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    total_memory_gb=$((total_memory / 1024 / 1024 / 1024))
    
    if [[ $total_memory_gb -gt 2 ]]; then
        log_success "Total memory: ${total_memory_gb}GB"
        validation_results+=("Memory")
        validation_status+=("OK")
    else
        log_warning "Low memory: ${total_memory_gb}GB (recommend >2GB)"
        validation_results+=("Memory")
        validation_status+=("WARN")
    fi
}

# Check network connectivity
check_network() {
    log_step "Checking network connectivity..."
    
    if ping -c 1 google.com &> /dev/null; then
        log_success "Internet connectivity: OK"
        validation_results+=("Internet")
        validation_status+=("OK")
    else
        log_error "No internet connectivity"
        validation_results+=("Internet")
        validation_status+=("FAIL")
        return 1
    fi
    
    # Check if we can reach Firebase
    if curl -s --connect-timeout 5 https://firebase.google.com &> /dev/null; then
        log_success "Firebase connectivity: OK"
        validation_results+=("Firebase Connectivity")
        validation_status+=("OK")
    else
        log_warning "Cannot reach Firebase (may be blocked by firewall)"
        validation_results+=("Firebase Connectivity")
        validation_status+=("WARN")
    fi
}

# Print validation summary
print_summary() {
    echo ""
    log_step "Validation Summary:"
    echo "===================="
    
    local total_checks=0
    local passed_checks=0
    local failed_checks=0
    local warnings=0
    
    for i in "${!validation_results[@]}"; do
        local status="${validation_status[$i]}"
        total_checks=$((total_checks + 1))
        
        case "$status" in
            "OK")
                echo -e "${GREEN}✓${NC} ${validation_results[$i]}"
                passed_checks=$((passed_checks + 1))
                ;;
            "WARN")
                echo -e "${YELLOW}⚠${NC} ${validation_results[$i]}"
                warnings=$((warnings + 1))
                ;;
            "FAIL")
                echo -e "${RED}✗${NC} ${validation_results[$i]}"
                failed_checks=$((failed_checks + 1))
                ;;
        esac
    done
    
    echo ""
    echo "Results: $passed_checks passed, $warnings warnings, $failed_checks failed"
    
    if [[ $failed_checks -gt 0 ]]; then
        log_error "Validation failed! Please fix the issues above before deploying."
        return 1
    elif [[ $warnings -gt 0 ]]; then
        log_warning "Validation completed with warnings. Deployment may proceed but issues should be addressed."
        return 0
    else
        log_success "All validations passed! Environment is ready for deployment."
        return 0
    fi
}

# Main validation function
main() {
    echo "🔍 Validating deployment environment for School Schedule Optimization Functions..."
    echo "📁 Script directory: $SCRIPT_DIR"
    echo ""
    
    # Run all checks
    check_python_version
    check_node_environment
    check_firebase_cli
    check_project_files
    check_firebase_config
    check_system_resources
    check_network
    
    # Print summary
    print_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h  Show this help message"
            echo ""
            echo "This script validates the deployment environment before running the main deployment."
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@" 