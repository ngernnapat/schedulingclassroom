#!/bin/bash

# Test Script for Deployment System
# This script tests the deployment system without actually deploying

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
TEST_RESULTS=()

# Test script existence
test_script_existence() {
    local scripts=("validate_env.sh" "deploy.sh" "rollback.sh")
    local all_exist=true
    
    for script in "${scripts[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            log_error "Missing script: $script"
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == "true" ]]; then
        log_success "✓ All deployment scripts exist"
        TEST_RESULTS+=("PASS: Script existence")
        return 0
    else
        TEST_RESULTS+=("FAIL: Script existence")
        return 1
    fi
}

# Test script permissions
test_script_permissions() {
    local scripts=("validate_env.sh" "deploy.sh" "rollback.sh")
    local all_executable=true
    
    for script in "${scripts[@]}"; do
        if [[ ! -x "$SCRIPT_DIR/$script" ]]; then
            log_error "Script not executable: $script"
            all_executable=false
        fi
    done
    
    if [[ "$all_executable" == "true" ]]; then
        log_success "✓ All scripts are executable"
        TEST_RESULTS+=("PASS: Script permissions")
        return 0
    else
        TEST_RESULTS+=("FAIL: Script permissions")
        return 1
    fi
}

# Test script syntax
test_script_syntax() {
    local scripts=("validate_env.sh" "deploy.sh" "rollback.sh")
    local all_valid=true
    
    for script in "${scripts[@]}"; do
        if ! bash -n "$SCRIPT_DIR/$script" 2>/dev/null; then
            log_error "Syntax error in: $script"
            all_valid=false
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_success "✓ All scripts have valid syntax"
        TEST_RESULTS+=("PASS: Script syntax")
        return 0
    else
        TEST_RESULTS+=("FAIL: Script syntax")
        return 1
    fi
}

# Test help commands
test_help_commands() {
    local scripts=("validate_env.sh" "deploy.sh" "rollback.sh")
    local all_help_work=true
    
    for script in "${scripts[@]}"; do
        if ! "$SCRIPT_DIR/$script" --help >/dev/null 2>&1; then
            log_error "Help command failed for: $script"
            all_help_work=false
        fi
    done
    
    if [[ "$all_help_work" == "true" ]]; then
        log_success "✓ All help commands work"
        TEST_RESULTS+=("PASS: Help commands")
        return 0
    else
        TEST_RESULTS+=("FAIL: Help commands")
        return 1
    fi
}

# Test rollback commands
test_rollback_commands() {
    local commands=("list" "create" "cleanup")
    local all_commands_work=true
    
    for cmd in "${commands[@]}"; do
        if ! "$SCRIPT_DIR/rollback.sh" "$cmd" >/dev/null 2>&1; then
            log_error "Rollback command failed: $cmd"
            all_commands_work=false
        fi
    done
    
    if [[ "$all_commands_work" == "true" ]]; then
        log_success "✓ All rollback commands work"
        TEST_RESULTS+=("PASS: Rollback commands")
        return 0
    else
        TEST_RESULTS+=("FAIL: Rollback commands")
        return 1
    fi
}

# Test configuration file
test_config_file() {
    if [[ -f "$SCRIPT_DIR/deploy.config" ]]; then
        log_success "✓ Configuration file exists"
        TEST_RESULTS+=("PASS: Configuration file")
        return 0
    else
        log_warning "⚠ Configuration file missing (optional)"
        TEST_RESULTS+=("WARN: Configuration file")
        return 0
    fi
}

# Test required files
test_required_files() {
    local required_files=("main.py" "requirements.txt" "school_scheduler.py" "planner_utils.py")
    local all_exist=true
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
            log_error "Missing required file: $file"
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == "true" ]]; then
        log_success "✓ All required files exist"
        TEST_RESULTS+=("PASS: Required files")
        return 0
    else
        TEST_RESULTS+=("FAIL: Required files")
        return 1
    fi
}

# Test Firebase configuration
test_firebase_config() {
    local project_root
    project_root=$(dirname "$SCRIPT_DIR")
    
    if [[ -f "$project_root/firebase.json" ]]; then
        log_success "✓ Firebase configuration exists"
        TEST_RESULTS+=("PASS: Firebase config")
        return 0
    else
        log_error "Missing Firebase configuration"
        TEST_RESULTS+=("FAIL: Firebase config")
        return 1
    fi
}

# Print test summary
print_summary() {
    echo ""
    log_step "Test Summary:"
    echo "=============="
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local warnings=0
    
    for result in "${TEST_RESULTS[@]}"; do
        total_tests=$((total_tests + 1))
        
        case "$result" in
            PASS:*)
                echo -e "${GREEN}✓${NC} ${result#PASS: }"
                passed_tests=$((passed_tests + 1))
                ;;
            WARN:*)
                echo -e "${YELLOW}⚠${NC} ${result#WARN: }"
                warnings=$((warnings + 1))
                ;;
            FAIL:*)
                echo -e "${RED}✗${NC} ${result#FAIL: }"
                failed_tests=$((failed_tests + 1))
                ;;
        esac
    done
    
    echo ""
    echo "Results: $passed_tests passed, $warnings warnings, $failed_tests failed"
    
    if [[ $failed_tests -gt 0 ]]; then
        log_error "Some tests failed! Please fix the issues before deploying."
        return 1
    elif [[ $warnings -gt 0 ]]; then
        log_warning "Tests completed with warnings. Deployment should work but issues should be addressed."
        return 0
    else
        log_success "All tests passed! Deployment system is ready."
        return 0
    fi
}

# Main test function
main() {
    echo "🧪 Testing Deployment System for School Schedule Optimization Functions"
    echo "📁 Test directory: $SCRIPT_DIR"
    echo ""
    
    # Run all tests
    test_script_existence
    test_script_permissions
    test_script_syntax
    test_help_commands
    test_rollback_commands
    test_config_file
    test_required_files
    test_firebase_config
    
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
            echo "This script tests the deployment system without actually deploying."
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