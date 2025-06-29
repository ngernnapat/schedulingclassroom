# School Schedule Optimization - Firebase Functions

A serverless API for generating optimized school schedules using constraint programming and optimization algorithms.

## 🚀 Features

- **Smart Schedule Generation**: Uses OR-Tools constraint programming to create optimal schedules
- **Flexible Constraints**: Supports PE classes, homeroom periods, lunch breaks, and teacher availability
- **RESTful API**: Clean HTTP endpoints for easy integration
- **Error Handling**: Comprehensive validation and error reporting
- **CORS Support**: Ready for web applications
- **Health Monitoring**: Built-in health check and monitoring endpoints

## 📋 API Endpoints

### 1. Health Check
```
GET /health_check
```
Returns service status and availability information.

**Response:**
```json
{
  "status": "healthy",
  "service": "school-schedule-optimizer",
  "version": "1.0.0",
  "scheduler_available": true,
  "python_version": "3.11.x",
  "environment": "production"
}
```

### 2. Schedule Information
```
GET /get_schedule_info
```
Returns API documentation and parameter information.

### 3. Generate Schedule
```
POST /generate_schedule
```
Generates an optimized school schedule based on provided parameters.

**Request Body:**
```json
{
  "n_teachers": 13,
  "grades": ["P1", "P2", "P3", "P4", "P5", "P6", "M1", "M2", "M3"],
  "pe_teacher": "T13",
  "pe_grades": ["P4", "P5", "P6", "M1", "M2", "M3"],
  "pe_day": 3,
  "n_pe_periods": 6,
  "start_hour": 8,
  "n_hours": 8,
  "lunch_hour": 5,
  "days_per_week": 5,
  "enable_pe_constraints": false,
  "homeroom_mode": 1
}
```

**Response:**
```json
{
  "success": true,
  "message": "Schedule generated successfully",
  "schedule": [...],
  "homeroom": [...],
  "parameters": {...},
  "metadata": {
    "total_assignments": 45,
    "homeroom_assignments": 9
  }
}
```

## 🛠️ Installation & Deployment

### Prerequisites
- Node.js and npm
- Python 3.11
- Firebase CLI (`npm install -g firebase-tools`)

### Quick Deployment

1. **Clone and navigate to the functions directory:**
   ```bash
   cd functions
   ```

2. **Run the deployment script:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Test the deployment:**
   ```bash
   python test_function.py
   ```

### Manual Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Deploy to Firebase:**
   ```bash
   firebase deploy --only functions
   ```

## 📦 Dependencies

- `firebase-functions==0.1.0` - Firebase Functions framework
- `firebase-admin==6.2.0` - Firebase Admin SDK
- `pandas==2.0.3` - Data manipulation
- `pulp==2.7.0` - Linear programming
- `ortools==9.7.2997` - Google OR-Tools for optimization
- `plotly==5.15.0` - Data visualization
- `numpy==1.24.3` - Numerical computing
- `requests==2.31.0` - HTTP requests
- `setuptools<81.0.0` - Package management (compatibility fix)

## 🔧 Configuration

### Firebase Configuration (`firebase.json`)
```json
{
  "functions": [
    {
      "source": "functions",
      "codebase": "default",
      "ignore": [
        "venv",
        ".git",
        "firebase-debug.log",
        "firebase-debug.*.log",
        "*.local",
        "__pycache__",
        "*.pyc"
      ],
      "runtime": "python311"
    }
  ]
}
```

### Environment Variables
The functions use default Firebase configuration. For custom settings, add environment variables in the Firebase console.

## 🧪 Testing

### Local Testing
```bash
python test_function.py
```

### Manual Testing
```bash
# Health check
curl https://your-project.cloudfunctions.net/health_check

# Get schedule info
curl https://your-project.cloudfunctions.net/get_schedule_info

# Generate schedule
curl -X POST https://your-project.cloudfunctions.net/generate_schedule \
  -H "Content-Type: application/json" \
  -d '{"n_teachers": 5, "grades": ["P1", "P2", "P3"]}'
```

## 📊 Performance & Limits

- **Max Instances**: 5 concurrent instances per function
- **Timeout**: 60 seconds for schedule generation
- **Memory**: 512MB allocated
- **Max Teachers**: 50
- **Max Grades**: 20
- **Max Hours per Day**: 10
- **Max Days per Week**: 7

## 🔍 Troubleshooting

### Common Issues

1. **Deployment Failures**
   - Check Python version compatibility (use Python 3.11)
   - Verify all dependencies are installed
   - Check Firebase project permissions

2. **Import Errors**
   - Ensure `school_scheduler.py` is in the functions directory
   - Check all required packages are in `requirements.txt`

3. **Timeout Errors**
   - Reduce the number of teachers or grades
   - Simplify constraints
   - Check for conflicting constraints

### Debugging

1. **Check logs:**
   ```bash
   firebase functions:log
   ```

2. **Test locally:**
   ```bash
   firebase emulators:start --only functions
   ```

3. **Verify deployment:**
   ```bash
   firebase functions:list
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Firebase Functions documentation
3. Open an issue in the repository

---

**Note**: This API is designed for educational institutions and should be used in accordance with your school's scheduling policies and requirements. 