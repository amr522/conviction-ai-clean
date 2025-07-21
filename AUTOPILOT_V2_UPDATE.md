# SageMaker Autopilot V2 Update - Summary

## Changes Made

1. **Added V2 API Support**:
   - Added `--use-automl-v2` flag to enable SageMaker Autopilot V2 API
   - Implemented proper configuration for both V1 and V2 API calls
   - Added fallback from V2 to V1 API when appropriate

2. **Implemented Time-based Split Support**:
   - Added `--split-type` parameter with support for both RANDOM and TIMESTAMP splits
   - Added `--timestamp-col` parameter to specify the column containing timestamps
   - Automatically enables V2 API when TIMESTAMP split is requested

3. **Enhanced Job Status Monitoring**:
   - Updated job status polling to support both V1 and V2 APIs
   - Improved status reporting with elapsed time tracking
   - Added better error handling for failed jobs

4. **Improved Model Deployment**:
   - Updated endpoint deployment logic to work consistently with both APIs
   - Enhanced endpoint information tracking to include API version used
   - Added more detailed status updates during endpoint creation

5. **Comprehensive Testing**:
   - Created new test suite for the combined functionality
   - Added tests for V1 API, V2 API, timestamp splits, and fallback scenarios
   - Added tests for argument parsing and validation

## Key Features

- **Dual API Support**: Seamlessly works with both V1 and V2 SageMaker Autopilot APIs
- **Time Series Support**: Full support for timestamp-based splits for time series data
- **Automatic Format Conversion**: Converts Parquet data to CSV format required by SageMaker
- **Robust Error Handling**: Graceful fallback from V2 to V1 API when needed
- **Comprehensive Logging**: Detailed status updates throughout the process
- **Enhanced Configuration**: Additional command-line parameters for fine-tuning

## Usage

```
python run_sagemaker_autopilot.py --input-s3-uri s3://bucket/data/ --role-arn <role-arn> \
  --problem-type Regression --target-column return --use-automl-v2 \
  --split-type TIMESTAMP --timestamp-col date
```

For standard random splits (default):
```
python run_sagemaker_autopilot.py --input-s3-uri s3://bucket/data/ --role-arn <role-arn> \
  --problem-type Regression --target-column return
```

## API Version Selection Logic

- **V2 API** is used when:
  - `--use-automl-v2` flag is present
  - `--split-type TIMESTAMP` is specified (V2 is automatically enabled)

- **V1 API** is used when:
  - `--use-automl-v2` flag is not present
  - V2 API fails and TIMESTAMP split was not requested

## Next Steps

- Add support for additional V2 API features
- Implement hyperparameter optimization configuration
- Add support for custom ML algorithms
- Enhance model evaluation with additional metrics
