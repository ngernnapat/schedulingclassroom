#!/bin/bash

# Rollback Script for School Schedule Optimization Functions
# This script allows safe rollback to previous deployments

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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$SCRIPT_DIR/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Check if running from correct directory
check_environment() {
    if [[ ! -f "$SCRIPT_DIR/main.py" ]]; then
        log_error "main.py not found. Please run this script from the functions directory."
        exit 1
    fi
    
    if ! command -v firebase &> /dev/null; then
        log_error "Firebase CLI is not installed"
        exit 1
    fi
}

# List available backups
list_backups() {
    log_step "Available backups:"
    
    if [[ ! -d "$BACKUP_DIR" ]] || [[ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]]; then
        log_warning "No backups found"
        return 1
    fi
    
    local backup_count=0
    for backup in "$BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$backup" ]]; then
            local backup_name=$(basename "$backup" .tar.gz)
            local backup_date=$(echo "$backup_name" | cut -d'_' -f1-2)
            local backup_time=$(echo "$backup_name" | cut -d'_' -f3-4)
            local size=$(du -h "$backup" | cut -f1)
            
            echo "  $backup_count: $backup_date $backup_time ($size)"
            backup_count=$((backup_count + 1))
        fi
    done
    
    return 0
}

# Create backup of current deployment
create_backup() {
    log_step "Creating backup of current deployment..."
    
    local backup_file="$BACKUP_DIR/deployment_${TIMESTAMP}.tar.gz"
    
    log_info "Creating backup: $backup_file"
    
    # Create backup of current functions
    tar -czf "$backup_file" \
        --exclude="venv" \
        --exclude="venv_backup" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude=".git" \
        --exclude="backups" \
        --exclude=".DS_Store" \
        -C "$SCRIPT_DIR" .
    
    if [[ -f "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Backup created successfully: $backup_file ($size)"
        return 0
    else
        log_error "Failed to create backup"
        return 1
    fi
}

# Restore from backup
restore_backup() {
    local backup_index=$1
    
    log_step "Restoring from backup..."
    
    # Get backup file by index
    local backup_files=()
    while IFS= read -r -d '' file; do
        backup_files+=("$file")
    done < <(find "$BACKUP_DIR" -name "*.tar.gz" -print0 | sort -z)
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        log_error "No backups found"
        return 1
    fi
    
    if [[ $backup_index -ge ${#backup_files[@]} ]] || [[ $backup_index -lt 0 ]]; then
        log_error "Invalid backup index: $backup_index"
        log_info "Available backups: 0-$(( ${#backup_files[@]} - 1 ))"
        return 1
    fi
    
    local backup_file="${backup_files[$backup_index]}"
    local backup_name=$(basename "$backup_file" .tar.gz)
    
    log_warning "This will overwrite current deployment with backup: $backup_name"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Rollback cancelled"
        return 0
    fi
    
    # Create backup of current state before restoring
    create_backup
    
    log_info "Restoring from: $backup_file"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$SCRIPT_DIR"
    
    log_success "Backup restored successfully"
    
    # Deploy restored version
    log_info "Deploying restored version..."
    cd "$PROJECT_ROOT"
    firebase deploy --only functions
    
    log_success "Rollback completed successfully!"
}

# Delete old backups
cleanup_backups() {
    local keep_count=${1:-5}
    
    log_step "Cleaning up old backups (keeping $keep_count most recent)..."
    
    # Get list of backups sorted by modification time (oldest first)
    local backup_files=()
    while IFS= read -r -d '' file; do
        backup_files+=("$file")
    done < <(find "$BACKUP_DIR" -name "*.tar.gz" -print0 | xargs -0 ls -t | head -n $((keep_count + 1)))
    
    if [[ ${#backup_files[@]} -le $keep_count ]]; then
        log_info "No old backups to clean up"
        return 0
    fi
    
    local deleted_count=0
    for ((i=keep_count; i<${#backup_files[@]}; i++)); do
        local backup_file="${backup_files[$i]}"
        local backup_name=$(basename "$backup_file")
        
        log_info "Deleting old backup: $backup_name"
        rm "$backup_file"
        deleted_count=$((deleted_count + 1))
    done
    
    log_success "Cleaned up $deleted_count old backups"
}

# Show backup details
show_backup_details() {
    local backup_index=$1
    
    # Get backup file by index
    local backup_files=()
    while IFS= read -r -d '' file; do
        backup_files+=("$file")
    done < <(find "$BACKUP_DIR" -name "*.tar.gz" -print0 | sort -z)
    
    if [[ $backup_index -ge ${#backup_files[@]} ]] || [[ $backup_index -lt 0 ]]; then
        log_error "Invalid backup index: $backup_index"
        return 1
    fi
    
    local backup_file="${backup_files[$backup_index]}"
    local backup_name=$(basename "$backup_file" .tar.gz)
    
    log_step "Backup details for index $backup_index:"
    echo "  File: $(basename "$backup_file")"
    echo "  Size: $(du -h "$backup_file" | cut -f1)"
    echo "  Created: $(stat -f "%Sm" "$backup_file" 2>/dev/null || stat -c "%y" "$backup_file")"
    
    # Show contents
    log_info "Contents:"
    tar -tzf "$backup_file" | head -20 | sed 's/^/    /'
    
    if [[ $(tar -tzf "$backup_file" | wc -l) -gt 20 ]]; then
        echo "    ... and $(($(tar -tzf "$backup_file" | wc -l) - 20)) more files"
    fi
}

# Main function
main() {
    echo "🔄 School Schedule Optimization Functions - Rollback Utility"
    echo "📁 Backup directory: $BACKUP_DIR"
    echo ""
    
    check_environment
    
    case "${1:-}" in
        "list"|"ls")
            list_backups
            ;;
        "create"|"backup")
            create_backup
            ;;
        "restore"|"rollback")
            if [[ -z "${2:-}" ]]; then
                log_error "Please specify backup index to restore"
                echo "Usage: $0 restore <backup_index>"
                echo "Use '$0 list' to see available backups"
                exit 1
            fi
            restore_backup "$2"
            ;;
        "cleanup")
            cleanup_backups "${2:-5}"
            ;;
        "details"|"info")
            if [[ -z "${2:-}" ]]; then
                log_error "Please specify backup index to show details"
                echo "Usage: $0 details <backup_index>"
                echo "Use '$0 list' to see available backups"
                exit 1
            fi
            show_backup_details "$2"
            ;;
        "help"|"--help"|"-h"|"")
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  list, ls                    List available backups"
            echo "  create, backup              Create backup of current deployment"
            echo "  restore, rollback <index>   Restore from backup by index"
            echo "  cleanup [count]             Clean up old backups (default: keep 5)"
            echo "  details, info <index>       Show backup details"
            echo "  help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 list                     # List all backups"
            echo "  $0 create                   # Create new backup"
            echo "  $0 restore 0                # Restore from backup index 0"
            echo "  $0 cleanup 3                # Keep only 3 most recent backups"
            echo "  $0 details 1                # Show details of backup index 1"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 