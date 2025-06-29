# Quick Start Guide - School Schedule Optimization

Get your Google Cloud Function up and running in 5 minutes! 🚀

## Prerequisites

- Google Cloud account with billing enabled
- Firebase CLI installed: `npm install -g firebase-tools`
- Python 3.9+ installed

## Step 1: Login to Firebase

```bash
firebase login
```

## Step 2: Navigate to Functions Directory

```bash
cd functions
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Deploy the Function

```bash
# Option A: Use the deployment script
./deploy.sh

# Option B: Deploy manually
firebase deploy --only functions
```

## Step 5: Test Your Function

After deployment, you'll get URLs like:
- `https://your-project.cloudfunctions.net/health_check`
- `https://your-project.cloudfunctions.net/generate_schedule`

### Quick Test with cURL

```bash
# Health check
curl https://your-project.cloudfunctions.net/health_check

# Generate a simple schedule
curl -X POST https://your-project.cloudfunctions.net/generate_schedule \
  -H "Content-Type: application/json" \
  -d '{
    "n_teachers": 5,
    "grades": ["P1", "P2", "P3"],
    "pe_teacher": "T5",
    "pe_grades": ["P2", "P3"],
    "pe_day": 3,
    "n_pe_periods": 2,
    "start_hour": 8,
    "n_hours": 6,
    "lunch_hour": 4,
    "days_per_week": 5,
    "enable_pe_constraints": false,
    "homeroom_mode": 1
  }'
```

### Test with Python Client

```bash
# Update the base_url in client_example.py with your function URL
python client_example.py
```

## 🎉 You're Done!

Your school schedule optimization API is now live and ready to use!

## Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- Check out the [client_example.py](client_example.py) for more usage examples
- Customize the parameters for your specific school needs

## Need Help?

- Check the [troubleshooting section](README.md#troubleshooting) in the main README
- View function logs: `firebase functions:log`
- Make sure all dependencies are installed correctly

## Common Issues

1. **Import Error**: Make sure `school_scheduler.py` is in the functions directory
2. **Deployment Fails**: Check that you're logged into Firebase and have the right project selected
3. **Function Times Out**: Complex schedules may take longer; consider reducing the problem size for testing

Happy scheduling! 📚✨ 