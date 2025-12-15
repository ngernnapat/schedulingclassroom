# 504 Gateway Timeout Fixes for generate-planner-content

## Problem Analysis

The `generate-planner-content` service was experiencing 504 Gateway Timeout errors due to several factors:

1. **Insufficient Resources**: Only 1024MB memory and default timeout (5 minutes)
2. **Sequential Processing**: Chunked generation processed chunks one by one
3. **Large Plan Requests**: No limits on plan size or request complexity
4. **Inefficient Chunking**: Small chunk sizes requiring many API calls
5. **No Timeout Monitoring**: No internal timeout checks

## Implemented Solutions

### 1. Cloud Run Configuration Optimizations

#### Memory and CPU Increases
```python
# Before
@https_fn.on_request(memory=1024, max_instances=3)

# After  
@https_fn.on_request(memory=2048, max_instances=5, timeout_sec=540)  # 9 minutes
```

#### Resource Allocation
- **Memory**: Increased from 1024MB to 2048MB
- **Timeout**: Increased from 5 minutes to 9 minutes (540 seconds)
- **Max Instances**: Increased from 3 to 5
- **Min Instances**: Set to 1 to reduce cold starts
- **CPU**: Increased to 2 cores

### 2. Request Validation and Limits

#### Payload Size Validation
```python
# Validate request size and complexity to prevent timeouts
if len(str(payload)) > 10000:  # 10KB limit for request payload
    return error_response("Request too large")
```

#### Plan Size Limits
```python
# Additional validation for large plans that might cause timeouts
if parsed.totalDays > 60:
    return error_response(f"Plans with {parsed.totalDays} days may take too long")
```

### 3. Chunking Optimizations

#### Larger Chunk Sizes
```python
# Before: Small chunks requiring many API calls
elif req.totalDays <= 30:
    analysis["optimal_chunk_size"] = 10
else:
    analysis["optimal_chunk_size"] = 15

# After: Larger chunks reducing API calls
elif req.totalDays <= 30:
    analysis["optimal_chunk_size"] = 15  # Increased from 10
else:
    analysis["optimal_chunk_size"] = 20  # Increased from 15
```

#### Reduced Delays
```python
# Before: 1 second delay between chunks
time.sleep(1)

# After: 0.5 second delay
time.sleep(0.5)  # Reduced for faster processing
```

### 4. Timeout Monitoring

#### Internal Timeout Checks
```python
max_generation_time = 480  # 8 minutes max (leave 1 minute buffer)

# Check if we're approaching timeout
elapsed_time = time.time() - generation_start_time
if elapsed_time > max_generation_time:
    raise PlannerGenerationError("Generation timeout")
```

#### Progress Logging
```python
print(f"Generating chunk {chunk_idx}/{len(chunks)}: {chunk.phase_name} - {elapsed_time:.2f}s elapsed")
```

### 5. Error Handling Improvements

#### Reduced Retry Backoff
```python
# Before: Exponential backoff
time.sleep(2 ** retry)

# After: Fixed 1 second delay
time.sleep(1)  # Reduced from exponential backoff
```

#### Better Error Messages
```python
# User-friendly timeout messages
"Plan generation is taking too long. Please try with fewer days or simpler requirements."
```

## Performance Improvements

### Before Optimization
- **Memory**: 1024MB
- **Timeout**: 5 minutes (300 seconds)
- **Chunk Size**: 10-15 days
- **Processing**: Sequential with 1s delays
- **Max Plans**: 90 days
- **Error Handling**: Generic fallback plans

### After Optimization
- **Memory**: 2048MB (2x increase)
- **Timeout**: 9 minutes (540 seconds)
- **Chunk Size**: 15-20 days (fewer API calls)
- **Processing**: Optimized with 0.5s delays
- **Max Plans**: 60 days (prevent timeouts)
- **Error Handling**: Proper timeout detection

## Deployment Configuration

### Cloud Run Service Settings
```yaml
annotations:
  run.googleapis.com/timeout: "540s"
  run.googleapis.com/memory: "2Gi"
  run.googleapis.com/cpu: "2"
  autoscaling.knative.dev/minScale: "1"
  autoscaling.knative.dev/maxScale: "5"
  autoscaling.knative.dev/target: "10"
```

### Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Monitoring and Debugging

### Logging Improvements
- **Generation Time Tracking**: Logs total generation time
- **Chunk Progress**: Logs each chunk completion time
- **Timeout Warnings**: Alerts when approaching timeout limits
- **Request Validation**: Logs rejected requests with reasons

### Metrics to Monitor
- **Request Duration**: Should be under 8 minutes
- **Memory Usage**: Should stay under 2GB
- **Error Rate**: Should decrease significantly
- **Cold Start Time**: Should improve with min instances

## Testing Results

### Expected Improvements
- **504 Errors**: Should be eliminated for plans ≤ 60 days
- **Response Time**: Faster processing due to larger chunks
- **Success Rate**: Higher success rate for complex plans
- **User Experience**: Clear error messages instead of timeouts

### Validation Steps
1. **Deploy** using `deploy_optimized.sh`
2. **Test** with various plan sizes (7, 14, 30, 45, 60 days)
3. **Monitor** logs for timeout warnings
4. **Verify** no 504 errors for valid requests

## Rollback Plan

If issues occur, rollback by:
1. Reverting to previous memory/timeout settings
2. Restoring original chunk sizes
3. Removing request validation limits
4. Using the original deployment configuration

## Future Optimizations

### Potential Improvements
1. **Parallel Chunk Processing**: Process multiple chunks simultaneously
2. **Caching**: Cache common plan patterns
3. **Streaming Responses**: Stream results as chunks complete
4. **Load Balancing**: Distribute large plans across multiple instances

### Monitoring Recommendations
1. Set up alerts for generation times > 6 minutes
2. Monitor memory usage patterns
3. Track error rates by plan size
4. Analyze user request patterns

## Conclusion

These optimizations address the root causes of 504 Gateway Timeout errors:

- **Resource Constraints**: Increased memory, CPU, and timeout limits
- **Processing Efficiency**: Larger chunks and reduced delays
- **Request Validation**: Prevents overly complex requests
- **Timeout Monitoring**: Proactive timeout detection and handling
- **Better Error Handling**: Clear user feedback instead of generic timeouts

The service should now handle complex, long-term plans without timing out while maintaining the quality improvements from the intelligent chunking system.
