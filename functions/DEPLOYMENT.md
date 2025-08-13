# Deployment Guide for School Schedule Optimization Functions

This guide covers the enhanced deployment system for the School Schedule Optimization Functions, including validation, deployment, and rollback capabilities.

## 📋 Overview

The deployment system consists of three main scripts:

1. **`validate_env.sh`** - Validates the deployment environment
2. **`deploy.sh`** - Main deployment script with enhanced features
3. **`rollback.sh`** - Safe rollback and backup management

## 🚀 Quick Start

### 1. Validate Environment
Before deploying, validate your environment:

```bash
cd functions
./validate_env.sh
```

This will check:
- Python version and dependencies
- Node.js and Firebase CLI
- Project files and configuration
- System resources and network connectivity

### 2. Deploy Functions
Run the enhanced deployment script:

```bash
./deploy.sh
```

For verbose output with debug information:

```bash
./deploy.sh --debug
```

### 3. Rollback if Needed
If you need to rollback to a previous deployment:

```bash
./rollback.sh list          # List available backups
./rollback.sh restore 0     # Restore from backup index 0
```

## 📁 Script Details

### `validate_env.sh`

Validates the deployment environment before running the main deployment.

**Usage:**
```bash
./validate_env.sh [--help]
```

**Checks performed:**
- ✅ Python 3.11+ availability
- ✅ Node.js and npm installation
- ✅ Firebase CLI installation and authentication
- ✅ Required project files presence
- ✅ Firebase configuration files
- ✅ System resources (disk space, memory)
- ✅ Network connectivity

**Output:**
- Green ✓ for passed checks
- Yellow ⚠ for warnings
- Red ✗ for failed checks

### `deploy.sh`

Enhanced deployment script with better error handling and user experience.

**Usage:**
```bash
./deploy.sh [--debug] [--help]
```

**Features:**
- 🔧 Automatic environment validation
- 🧹 Cleanup of previous artifacts
- 🐍 Virtual environment management
- 📦 Dependency verification
- 🔍 Function import testing
- 🚀 Firebase deployment
- 📋 Function URL generation

**Options:**
- `--debug`: Enable verbose output and debug mode
- `--help`: Show help information

**Process:**
1. Pre-flight checks (Python, Node.js, Firebase CLI)
2. Cleanup previous artifacts
3. Create and activate virtual environment
4. Install and verify dependencies
5. Test function imports
6. Deploy to Firebase
7. Display function URLs

### `rollback.sh`

Safe rollback and backup management system.

**Usage:**
```bash
./rollback.sh <command> [options]
```

**Commands:**
- `list, ls` - List available backups
- `create, backup` - Create backup of current deployment
- `restore, rollback <index>` - Restore from backup by index
- `cleanup [count]` - Clean up old backups (default: keep 5)
- `details, info <index>` - Show backup details
- `help` - Show help information

**Examples:**
```bash
./rollback.sh list                    # List all backups
./rollback.sh create                  # Create new backup
./rollback.sh restore 0               # Restore from backup index 0
./rollback.sh cleanup 3               # Keep only 3 most recent backups
./rollback.sh details 1               # Show details of backup index 1
```

## ⚙️ Configuration

### `deploy.config`

Configuration file for deployment settings:

```bash
# Project Configuration
PROJECT_NAME="School Schedule Optimization Functions"
PROJECT_ID="school-schedule-optimization-functions"

# Python Configuration
PYTHON_VERSION="3.11"
PYTHON_MIN_VERSION="3.11"

# Firebase Configuration
FIREBASE_RUNTIME="python311"
FIREBASE_REGION="us-central1"

# Function Configuration
MAX_INSTANCES=5
DEFAULT_TIMEOUT=300

# Dependencies to verify
REQUIRED_DEPENDENCIES=(
    "firebase_admin"
    "pandas"
    "pulp"
    "ortools"
    "plotly"
    "numpy"
    "requests"
    "fastapi"
    "pydantic"
    "python-dotenv"
    "openai"
    "langdetect"
)
```

## 🔧 Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python3 --version
   ```

2. **Node.js 20+**
   ```bash
   node --version
   npm --version
   ```

3. **Firebase CLI**
   ```bash
   npm install -g firebase-tools
   firebase login
   ```

4. **Git** (for version control)
   ```bash
   git --version
   ```

### Required Files

Ensure these files exist in your project:

```
functions/
├── main.py              # Main function code
├── requirements.txt     # Python dependencies
├── school_scheduler.py  # Scheduler implementation
├── planner_utils.py     # Utility functions
└── deploy.sh           # Deployment script

firebase.json           # Firebase configuration
.firebaserc            # Firebase project settings
```

## 🚨 Troubleshooting

### Common Issues

1. **Python version mismatch**
   ```bash
   # Install Python 3.11
   brew install python@3.11  # macOS
   sudo apt install python3.11  # Ubuntu
   ```

2. **Firebase CLI not authenticated**
   ```bash
   firebase login
   firebase projects:list  # Verify authentication
   ```

3. **Missing dependencies**
   ```bash
   # Check requirements.txt
   cat requirements.txt
   
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **Deployment fails**
   ```bash
   # Check Firebase project
   firebase use --add
   
   # Check function logs
   firebase functions:log
   ```

### Debug Mode

Enable debug mode for verbose output:

```bash
./deploy.sh --debug
```

This will show:
- Detailed dependency installation
- Function import testing
- Firebase deployment logs
- Environment information

## 📊 Monitoring

### Check Function Status

```bash
# List deployed functions
firebase functions:list

# Check function logs
firebase functions:log

# Monitor function performance
firebase functions:config:get
```

### Test Endpoints

After deployment, test your functions:

```bash
# Health check
curl https://your-project-id.cloudfunctions.net/health_check

# Get schedule info
curl https://your-project-id.cloudfunctions.net/get_schedule_info

# Generate schedule (POST request)
curl -X POST https://your-project-id.cloudfunctions.net/generate_schedule \
  -H "Content-Type: application/json" \
  -d '{"n_teachers": 13, "grades": ["P1", "P2", "P3"]}'
```

## 🔄 Backup and Rollback

### Automatic Backups

The deployment script automatically creates backups before deploying:

```bash
# List backups
./rollback.sh list

# Create manual backup
./rollback.sh create

# Restore from backup
./rollback.sh restore 0
```

### Backup Management

```bash
# Clean up old backups (keep 5 most recent)
./rollback.sh cleanup

# Keep only 3 backups
./rollback.sh cleanup 3

# Show backup details
./rollback.sh details 0
```

## 📈 Best Practices

1. **Always validate environment first**
   ```bash
   ./validate_env.sh
   ```

2. **Create backups before major changes**
   ```bash
   ./rollback.sh create
   ```

3. **Use debug mode for troubleshooting**
   ```bash
   ./deploy.sh --debug
   ```

4. **Monitor function logs after deployment**
   ```bash
   firebase functions:log
   ```

5. **Test endpoints after deployment**
   ```bash
   curl https://your-project-id.cloudfunctions.net/health_check
   ```

## 🆘 Support

If you encounter issues:

1. Run validation: `./validate_env.sh`
2. Check logs: `firebase functions:log`
3. Enable debug mode: `./deploy.sh --debug`
4. Review this documentation
5. Check Firebase console for function status

## 📝 Changelog

### Version 2.0 (Enhanced)
- ✅ Added environment validation script
- ✅ Enhanced deployment script with better error handling
- ✅ Added rollback and backup management
- ✅ Improved logging and user experience
- ✅ Added configuration file support
- ✅ Added debug mode for troubleshooting
- ✅ Added comprehensive documentation

### Version 1.0 (Original)
- Basic deployment script
- Simple error handling
- Manual validation required 