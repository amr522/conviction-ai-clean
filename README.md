# Conviction-AI Machine Learning Pipeline

[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](https://github.com/amr522/conviction-ai-clean/releases/tag/v2.1.0)
[![Coverage Status](https://img.shields.io/badge/coverage-50%25-yellow.svg)](htmlcov/index.html)

This repository contains the Conviction-AI machine learning pipeline, which automates data processing, model training, and deployment to AWS.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Repository Structure](#repository-structure)
- [AWS Pipeline](#aws-pipeline)
- [Running Tests Locally](#running-tests-locally)
- [CI/CD Pipeline](#cicd-pipeline)
- [Slack Notifications](#sla## 🔍 Data Validation Framework

We've implemented a comprehensive framework for validating data quality and preventing forward-looking bias in our machine learning pipeline. This includes:

- Feature validation with Spark Window functions and pandas
- Great Expectations integration for ETL validation
- Time-series split and early stopping for all training scripts

For details, see [DATA_VALIDATION.md](DATA_VALIDATION.md).

## 📈 Volatility Prediction Features

The Conviction-AI pipeline now includes advanced volatility prediction features that enhance the existing implied volatility (IV) change prediction capabilities. These new features enable more sophisticated volatility trading strategies.

### Volatility Target Columns

The pipeline generates and uses the following volatility-related target columns:

- **sigma_forecast_5d**: 5-day volatility forecast using EGARCH models, providing forward-looking volatility estimates
- **ivhv_spread_tplus1**: The spread between implied volatility (IV) and historical volatility (HV) for t+1, a key mean-reversion indicator

### Multi-Target Training

The machine learning models can now be trained with multiple target columns simultaneously:

1. **RandomForestRegressor with MultiOutputRegressor**: Train a single model to predict multiple volatility targets
2. **LightGBM Blender**: Enhanced to support volatility targets with fallback mechanism to iv_change_5d
3. **PatchTST with Multi-Target Support**: Deep learning time series model updated for multi-target prediction

### Using Volatility Targets

To use the volatility targets in your training and prediction workflows:

1. **Create holdout dataset with volatility targets**:
   ```bash
   python create_holdout_csv.py --include-volatility-targets
   ```

2. **Train models with multiple volatility targets**:
   ```bash
   python train_patchtst_tail.py --target-columns "tail_event,sigma_forecast_5d,ivhv_spread_tplus1"
   ```

3. **Run SageMaker Autopilot with volatility target**:
   ```bash
   python run_sagemaker_autopilot.py --target-column sigma_forecast_5d
   ```

### Volatility Targets Demo

For a comprehensive demonstration of the volatility targets workflow, run the included Jupyter notebook:

```bash
jupyter notebook volatility_targets_demo.ipynb
```

The notebook provides:
- Data loading and exploration of volatility features
- Training workflows for multi-target models
- Visualization of feature importance for volatility prediction
- Performance evaluation across different volatility targets

## 📚 Additional Documentationtions)
- [AWS Resources](#aws-resources)
- [Data Validation Framework](#data-validation-framework)
- [Volatility Prediction Features](#volatility-prediction-features)
- [Additional Documentation](#additional-documentation)

The pipeline generates and uses the following volatility-related target columns:

- **sigma_forecast_5d**: 5-day volatility forecast using EGARCH models, providing forward-looking volatility estimates
- **ivhv_spread_tplus1**: The spread between implied volatility (IV) and historical volatility (HV) for t+1, a key mean-reversion indicator

### Multi-Target Training

The machine learning models can now be trained with multiple target columns simultaneously:

1. **RandomForestRegressor with MultiOutputRegressor**: Train a single model to predict multiple volatility targets
2. **LightGBM Blender**: Enhanced to support volatility targets with fallback mechanism to iv_change_5d
3. **PatchTST with Multi-Target Support**: Deep learning time series model updated for multi-target prediction

### Using Volatility Targets

To use the volatility targets in your training and prediction workflows:

1. **Create holdout dataset with volatility targets**:
   ```bash
   python create_holdout_csv.py --include-volatility-targets
   ```

2. **Train models with multiple volatility targets**:
   ```bash
   python train_patchtst_tail.py --target-columns "tail_event,sigma_forecast_5d,ivhv_spread_tplus1"
   ```

3. **Run SageMaker Autopilot with volatility target**:
   ```bash
   python run_sagemaker_autopilot.py --target-column sigma_forecast_5d
   ```

### Volatility Targets Demo

For a comprehensive demonstration of the volatility targets workflow, run the included Jupyter notebook:

```bash
jupyter notebook volatility_targets_demo.ipynb
```

The notebook provides:
- Data loading and exploration of volatility features
- Training workflows for multi-target models
- Visualization of feature importance for volatility prediction
- Performance evaluation across different volatility targets

## 📚 Additional Documentationicd-pipeline)
- [Slack Notifications](#slack-notifications)
- [AWS Resources](#aws-resources)
- [Data Validation Framework](#data-validation-framework)
- [Additional Documentation](#additional-documentation)

## 🔧 Environment Setup

### Local Development Setup

```bash
./scripts/setup-env.sh
source .venv/bin/activate
```

Then you can run:
```bash
./scripts/evaluate_pipeline.sh 2025-07-27
```

### DevContainer Development Environment

We provide a VS Code DevContainer with all dependencies installed.

1. Open this project in VS Code.
2. When prompted, reopen in DevContainer.
3. The `scripts/setup-env.sh` will run automatically to create a virtualenv and install dependencies.

You'll have:
- A consistent Python environment (`.venv`)
- Recommended extensions installed automatically
- AutoSave and FormatOnSave enabled

### Code Quality Setup

Pre-commit hooks enforce code quality standards:

```bash
pip install pre-commit
pre-commit install
```

This enables automatic code formatting, import sorting, type checking, and commit message validation on every commit.

### Type Checking

Run mypy locally for comprehensive type checking:

```bash
mypy --config-file=mypy.ini src/
```

### Test Coverage

Run tests with coverage measurement:

```bash
# Install test dependencies first
./scripts/install-test-deps.sh

# Run tests with coverage
coverage run -m pytest
coverage report --fail-under=80
```

Generate HTML coverage report:

```bash
coverage html
```

Tests must meet a minimum of 80% code coverage to pass.

### Prerequisites

- Python 3.9+
- AWS CLI installed and configured
- Appropriate AWS IAM permissions for S3, Glue, and SageMaker
- Git

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/conviction-ai-clean.git
   cd conviction-ai-clean
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   Create a `.env` file with the following required variables:
   ```
   # AWS credentials (or use IAM roles)
   AWS_ACCESS_KEY_ID=your_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1

   # S3 bucket for data storage
   S3_BUCKET_NAME=your-bucket-name

   # Slack webhook for notifications
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
   ```

4. For Slack notifications integration:
   - Create a Slack App in your workspace
   - Enable Incoming Webhooks
   - Create a webhook for your desired channel
   - Add the webhook URL to your `.env` file

## 📁 Repository Structure

The repository is organized into the following key directories:

```
conviction-ai-clean/
├── aws_pipeline/            # AWS SageMaker and pipeline components
│   ├── model_analysis.py    # Tools for model evaluation and analysis
│   ├── run_aws_pipeline.sh  # Main script to execute the full AWS pipeline
│   ├── setup_aws_env.sh     # Setup script for AWS environment
│   └── README.md            # AWS pipeline documentation
├── data/                    # Data directory (not tracked in git)
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── predictions/         # Model predictions
├── tests/                   # Test suite
├── .env                     # Environment variables (not tracked in git)
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # Main documentation
```

## ☁️ AWS Pipeline

The `aws_pipeline` directory contains all components for the AWS SageMaker and data processing pipeline. This includes scripts for model training, deployment, and analysis.

### Setup and Configuration

To set up the AWS environment:

1. Navigate to the aws_pipeline directory:
   ```bash
   cd aws_pipeline
   ```

2. Make the setup script executable and run it:
   ```bash
   chmod +x setup_aws_env.sh
   ./setup_aws_env.sh
   ```

### Running the Pipeline

To execute the full AWS pipeline:

1. Make the pipeline script executable:
   ```bash
   chmod +x run_aws_pipeline.sh
   ```

2. Run the pipeline:
   ```bash
   ./run_aws_pipeline.sh
   ```

### Model Analysis

To analyze a deployed SageMaker model for data leakage and overfitting:

```bash
python aws_pipeline/model_analysis.py
```

The script will:
- Connect to your AWS account
- Load data from S3
- Evaluate model predictions
- Generate analysis reports for data leakage and model performance

## 📈 Volatility Prediction Features

The Conviction-AI pipeline now includes advanced volatility prediction capabilities with multiple volatility targets. To use these features:

### Running Inference with Multi-Target Models

To generate predictions from a deployed multi-target volatility model:

```bash
# Basic inference
python inference_from_sagemaker.py --endpoint-name your-endpoint-name --input-file data/your_input.csv

# Multi-target volatility inference
python inference_from_sagemaker.py --endpoint-name your-endpoint-name --input-file data/your_input.csv --target-columns iv_change_5d,spread_change_5d,vrp_change_5d
```

The `--target-columns` parameter accepts a comma-separated list of target columns that the model has been trained to predict. The output CSV file will include a prediction column for each target.

### Multi-Target Evaluation

To evaluate a deployed multi-target volatility model using holdout data:

```bash
python evaluate_endpoint_rmse.py \
  --holdout-csv s3://your-bucket/holdout.csv \
  --target-columns iv_change_5d,spread_change_5d,vrp_change_5d \
  --endpoint-name my-endpoint \
  --output-file predictions.csv \
  --output-json eval_metrics.json
```

This will:
1. Load holdout data from the specified CSV file
2. Send features to the SageMaker endpoint for prediction
3. Calculate RMSE, MAE, and R² metrics for each target column
4. Save results to a JSON file with the following structure:

```json
{
  "iv_change_5d": {
    "rmse": 0.0123,
    "mae": 0.0098,
    "r2": 0.876
  },
  "spread_change_5d": {
    "rmse": 0.0056,
    "mae": 0.0041,
    "r2": 0.921
  },
  "vrp_change_5d": {
    "rmse": 0.0078,
    "mae": 0.0062,
    "r2": 0.894
  }
}
```

The JSON report provides the following metrics for each target:
- **RMSE (Root Mean Squared Error)**: Lower values indicate better fit
- **MAE (Mean Absolute Error)**: Lower values indicate better fit
- **R² (Coefficient of Determination)**: Values closer to 1.0 indicate better fit

The script also creates a CSV file with the original data plus prediction columns for each target, which can be used for further analysis or visualization.

## 🧪 Running Tests Locally

The project includes a comprehensive test suite to validate functionality. To run tests locally:

### Install test dependencies:

```bash
pip install pytest pytest-xdist coverage
```

## 🔍 Data Validation

The pipeline includes comprehensive data validation to ensure data quality and prevent forward-looking bias:

### Running Data Validation

Before training models, validate the processed dataset:

```bash
python validate_option_features.py --s3-bucket convictionai-data --output-report validation_report.json
```

### Validation Checks

The validation suite performs the following checks:

1. **Row Count Validation**: Ensures sufficient data for training (minimum 1,000 records)
2. **Schema Validation**: Verifies all expected columns are present with correct data types
3. **Null Value Checks**: Validates null percentages are within acceptable limits (<5% error threshold, <1% warning threshold)
4. **Range Validation**: Ensures financial features are within realistic bounds:
   - Delta: [-1.0, 1.0]
   - IV Percentile: [0.0, 1.0]
   - Time to Expiry: [0.0, 365.0] days
   - Sentiment: [-1.0, 1.0]
5. **Lag Integrity**: Verifies that lagged features have proper temporal structure
6. **Forward-Looking Bias Detection**: Identifies potential data leakage issues
7. **Data Consistency**: Validates logical relationships (e.g., expiry > timestamp)

### Validation Report

The validation generates a comprehensive report with:
- Individual validation results (PASS/FAIL)
- Overall success rate
- Detailed error messages for failed validations
- Recommendations for fixing issues

Example validation output:
```
VALIDATION SUMMARY
=====================================
row_count...................... ✅ PASS
schema......................... ✅ PASS
null_values.................... ✅ PASS
ranges......................... ✅ PASS
lag_integrity.................. ✅ PASS
forward_looking................ ✅ PASS
consistency.................... ✅ PASS

Overall Success Rate: 100.0%
Passed: 7/7

🎉 Dataset validation SUCCESSFUL! Ready for training.
```

## 🛡️ Anti-Leakage Safeguards

To prevent forward-looking bias and ensure robust model performance:

### 1. Feature Lagging

All potentially forward-looking features are automatically lagged by one period:

**News Features** (lagged):
- `news_count`, `avg_sentiment`, `sum_sentiment`
- `news_volatility`, `news_ext_count`, `news_ext_avg_sentiment`

**FRED Economic Features** (lagged):
- `fed_event_flag`, `fed_surprise_mean`, `fed_surprise_sum`
- `fed_actual_mean`, `fed_forecast_mean`

**Treasury Auction Features** (lagged):
- `auction_flag`, `auction_amount`, `auction_yield`

### 2. Time-Series Training Splits

All training scripts use proper time-series validation:

- **Cross-Validation**: 5-fold time-series split (no random shuffling)
- **Final Validation**: Last 10% of timeline reserved for validation
- **Early Stopping**: 50 rounds to prevent overfitting

### 3. Great Expectations in ETL

The ETL pipeline includes built-in validation that fails the job if:
- Forward-looking features are detected
- Future timestamps are present
- Duplicate primary keys exist
- Required columns are missing
- Data ranges are unrealistic

### 4. Testing Anti-Leakage

Run the feature lagging validation:
```bash
python validate_feature_lagging.py
```

This verifies that:
- Features are properly shifted by 1 period per symbol
- First row per symbol has NaN for lagged features
- Non-lagged features remain unchanged

### Best Practices

1. **Always run validation** before training: `python validate_option_features.py`
2. **Check validation reports** for any warnings or errors
3. **Use time-series splits** - never random shuffling for temporal data
4. **Monitor for data leakage** in production using the validation suite
5. **Test regularly** with `pytest tests/test_validate_option_features.py`

### Run all tests:

```bash
python -m pytest tests/ -v
```

### Run tests in parallel:

```bash
python -m pytest tests/ -n auto
```

### Generate a coverage report:

```bash
coverage run -m pytest tests/
coverage report
coverage html  # Creates an HTML report in htmlcov/
```

Tests must meet a minimum of 80% code coverage to pass.

## 🚀 CI/CD Pipeline

The CI/CD pipeline is implemented using GitHub Actions and defined in `.github/workflows/aws_ml_pipeline.yml`. It automates testing, code quality checks, and deployment to AWS.

### Pipeline Triggers

The pipeline runs automatically:
- On every push to the `main` branch
- On a daily schedule at 03:00 UTC

### Pipeline Workflow

The pipeline includes the following key steps:

1. **Setup Environment**:
   - Checkout code
   - Configure AWS credentials
   - Set up Python 3.9
   - Install dependencies

2. **Code Quality Checks**:
   - Linting with flake8
   - Static code analysis

3. **Run Tests**:
   - Execute tests in parallel using pytest-xdist
   - Generate and check code coverage
   - Fail if coverage is below 80%
   - Upload coverage report as an artifact

4. **Slack Notification (Tests)**:
   - Notify team on test success or failure
   - Include repository, workflow, and run details

5. **Data Processing**:
   - Reorganize S3 bucket
   - Start AWS Glue ETL job
   - Poll for job completion

6. **Model Training & Deployment**:
   - Run SageMaker Autopilot job
   - Deploy trained model to endpoint
   - Log results and endpoint URL

7. **Final Notification**:
   - Send comprehensive Slack notification with pipeline results
   - Include links to deployed resources

### Concurrency Control

The pipeline uses GitHub Actions concurrency controls to prevent overlapping runs, ensuring that only one instance runs at a time.

## 💬 Slack Notifications

The pipeline integrates with Slack to provide real-time notifications about pipeline status.

### Notification Types

1. **Test Results Notification**:
   - Sent immediately after tests complete
   - Indicates test success or failure
   - Includes repository and workflow details
   - For failures, includes a link to GitHub Actions logs

2. **Pipeline Completion Notification**:
   - Sent when the entire pipeline completes
   - On success: includes Glue job, AutoML job, and endpoint details
   - On failure: includes link to GitHub Actions logs for troubleshooting

### Setting Up Slack Notifications

1. Create a Slack App in your workspace
2. Enable Incoming Webhooks
3. Create a webhook for your channel
4. Add the webhook URL as a GitHub secret named `SLACK_WEBHOOK_URL`

## ☁️ AWS Resources

The pipeline interacts with the following AWS resources:

- **S3 Bucket**: Stores raw and processed data
- **AWS Glue**: Runs ETL jobs to transform and prepare data
- **SageMaker Autopilot**: Automatically trains and tunes machine learning models
- **SageMaker Endpoint**: Hosts the deployed model for inference

### AWS Pipeline Components

The `aws_pipeline` directory contains all the necessary components for AWS interaction:

- **run_sagemaker_autopilot.py**: Launches and monitors SageMaker Autopilot jobs
- **cleanup_sagemaker_resources.py**: Manages SageMaker resource cleanup
- **model_analysis.py**: Analyzes model performance and checks for data leakage
- **run_aws_pipeline.sh**: Main script that orchestrates the entire AWS workflow
- **setup_aws_env.sh**: Sets up required environment variables and dependencies

### Infrastructure as Code

The project includes CloudFormation templates to provision all required AWS resources:

#### Deploying the Infrastructure:

You can deploy the infrastructure using the AWS pipeline scripts:

```bash
# Set up the environment first
cd aws_pipeline
./setup_aws_env.sh

# Deploy the infrastructure and run the pipeline
./run_aws_pipeline.sh
```

#### Cleanup Resources:

To clean up AWS resources after use:

```bash
python aws_pipeline/cleanup_sagemaker_resources.py
```

## � Data Validation Framework

We've implemented a comprehensive framework for validating data quality and preventing forward-looking bias in our machine learning pipeline. This includes:

- Feature validation with Spark Window functions and pandas
- Great Expectations integration for ETL validation
- Time-series split and early stopping for all training scripts

For details, see [DATA_VALIDATION.md](DATA_VALIDATION.md).

## �📚 Additional Documentation

- For AWS pipeline details, see [aws_pipeline/README.md](aws_pipeline/README.md)
- For GPU training information, see [GPU_TRAINING_INSTRUCTIONS.md](GPU_TRAINING_INSTRUCTIONS.md)
- For model performance analysis, see [MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md)
- **For Step Functions + SageMaker Pipeline integration, see [STEP_FUNCTIONS_INTEGRATION.md](STEP_FUNCTIONS_INTEGRATION.md)**

## 📖 Tutorials

- [Autopilot_V2_TimeSeries_Split_Tutorial.ipynb](Autopilot_V2_TimeSeries_Split_Tutorial.ipynb) - Tutorial on using SageMaker Autopilot V2 with time-series splits

## 🚀 Latest AutoML V2 Run

**Job Details:**
- **Job Name**: `conviction-automl-20250721153332`
- **Best Candidate RMSE**: `0.0472`
- **Deployed Endpoint**: `conviction-ai-endpoint-20250721153332`
- **Date**: `2025-07-21`
- **API Version**: `V2 (CreateAutoMLJobV2)`
- **Duration**: `29 minutes (1749 seconds)`
- **Total Candidates**: `40 models evaluated`

**Performance Results:**
The AutoML V2 job successfully completed with excellent performance metrics. The best performing model achieved a validation RMSE of 0.0472, demonstrating strong predictive accuracy.

**Metrics Artifact:**
- S3 Location: `s3://sagemaker-us-east-1-773934887314/conviction-ai/automl-out/conviction-automl-20250721153332/`
- Candidate metrics available in job output folder

**Resource Cleanup:**
After testing or when no longer needed, clean up the deployed resources to avoid unnecessary costs:

```bash
# Remove endpoint after hours to save cost
python cleanup_sagemaker_resources.py \
   --prefix conviction-automl-20250721153332 \
   --include-endpoints \
   --dry-run  # Remove --dry-run to actually delete

# Or clean up all conviction resources older than 7 days
python cleanup_sagemaker_resources.py \
   --prefix conviction \
   --older-than-days 7 \
   --include-all \
   --dry-run  # Remove --dry-run to actually delete
```

## 🛠️ Operations

### Production Automation System

The conviction-ai pipeline includes automated production infrastructure for continuous model retraining and cleanup:

#### Automated Nightly Retraining

The pipeline automatically retrains models nightly at 21:15 ET (01:15 UTC) using SageMaker Pipelines:

- **Pipeline Name**: `conviction-ai-retrain-pipeline`
- **Schedule**: Daily at 21:15 ET (01:15 UTC)
- **Components**: Glue ETL → LightGBM Training → Model Registry → Conditional Deployment
- **Deployment Threshold**: RMSE < 0.05 for automatic deployment
- **Model Registry**: Automatic registration of qualified models

#### Automated Resource Cleanup

Old endpoints are automatically cleaned up to manage costs:

- **Cleanup Schedule**: Daily at 01:15 UTC (after retraining completes)
- **Retention Policy**: Endpoints older than 14 days are automatically deleted
- **State Machine**: `conviction-ai-cleanup-automation`
- **Monitoring**: CloudWatch logs for cleanup activity

#### Manual Pipeline Triggers

You can manually trigger retraining outside the scheduled time using GitHub Actions:

```bash
# Trigger manual retrain with default settings
gh workflow run dispatch_retrain.yml

# Trigger with custom parameters
gh workflow run dispatch_retrain.yml \
  -f execution_date=2024-01-15 \
  -f max_rmse=0.04 \
  -f s3_bucket=conviction-ai-bucket
```

**Manual Trigger Parameters:**
- `execution_date`: Specific date for training data (YYYY-MM-DD, defaults to today)
- `max_rmse`: RMSE threshold for deployment (defaults to 0.05)
- `s3_bucket`: S3 bucket for artifacts (defaults to conviction-ai-bucket)

#### Setting Up Production Automation

1. **Deploy SageMaker Pipeline:**
   ```bash
   python pipeline_retrain.py --role arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
   ```

2. **Deploy Step Functions State Machine:**
   ```bash
   python create_state_machine.py \
     --lambda-role arn:aws:iam::ACCOUNT:role/LambdaExecutionRole \
     --stepfunctions-role arn:aws:iam::ACCOUNT:role/StepFunctionsExecutionRole \
     --deploy-schedule
   ```

3. **Configure GitHub OIDC (for manual triggers):**
   - Set `AWS_ROLE_ARN` secret in repository settings
   - Ensure role has SageMaker pipeline execution permissions

#### Monitoring Production Automation

- **SageMaker Pipelines Console**: Monitor training pipeline executions
- **Step Functions Console**: Monitor cleanup automation
- **CloudWatch Logs**: Detailed execution logs and reports
- **GitHub Actions**: Manual trigger history and status

### Slack Alerts

Both nightly and weekly workflows automatically post run results to your Slack channel.

**Setup Requirements:**
- Ensure `SLACK_WEBHOOK_URL` is set in repository secrets (Settings → Secrets → Actions)
- Create a Slack App with Incoming Webhooks enabled
- Configure webhook for your desired channel

**Notification Features:**
- **Run Status**: Success/failure status with workflow details
- **Artifacts**: Links to generated artifacts and logs
- **Color Coding**: Green for success, red for failures
- **Workflow Context**: Run number, trigger type, and execution details

**Slack Webhook Setup:**
1. Visit https://api.slack.com/apps
2. Create a new app for your workspace
3. Enable Incoming Webhooks
4. Create webhook for your channel
5. Add webhook URL to repository secrets as `SLACK_WEBHOOK_URL`

### Endpoint Autoscaling

Configure automatic scaling for SageMaker endpoints based on traffic load:

```bash
# Configure autoscaling for a specific endpoint
./configure_endpoint_autoscaling.sh conviction-ai-endpoint-20250721153332

# This will:
# • Set min instances: 1, max instances: 3
# • Target: 50 invocations per instance
# • Scale out/in cooldown: 5 minutes
```

**Autoscaling Features:**
- **Automatic scaling**: Scale between 1-3 instances based on traffic
- **Cost optimization**: Scales down during low traffic periods
- **Performance**: Scales up automatically during high traffic
- **Target tracking**: Maintains 50 invocations per instance target

### CloudWatch Monitoring

Set up comprehensive monitoring and alerting for endpoints:

```bash
# Create CloudWatch alarms for endpoint monitoring
./create_endpoint_alarms.sh conviction-ai-endpoint-20250721153332

# This creates alarms for:
# • Error rate > 1% over 5 minutes
# • P95 latency > 500ms over 10 minutes
# • Low invocation count (< 10 calls/hour for 2 hours)
```

**Monitoring Features:**
- **Error tracking**: Alerts on high 4XX/5XX error rates
- **Latency monitoring**: Detects performance degradation
- **Cost alerts**: Identifies underutilized endpoints
- **CloudWatch integration**: Full AWS monitoring ecosystem

### Operational Commands

```bash
# View autoscaling activities
aws application-autoscaling describe-scaling-activities \
  --service-namespace sagemaker \
  --resource-id endpoint/YOUR_ENDPOINT_NAME/variant/AllTraffic

# Check current endpoint metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=YOUR_ENDPOINT_NAME \
  --start-time 2025-07-21T00:00:00Z \
  --end-time 2025-07-21T23:59:59Z \
  --period 3600 \
  --statistics Sum

# Delete all alarms for an endpoint
aws cloudwatch delete-alarms --alarm-names \
  YOUR_ENDPOINT_NAME-ErrorRate \
  YOUR_ENDPOINT_NAME-HighLatency \
  YOUR_ENDPOINT_NAME-LowInvocations
```

### Model Performance Dashboard

Monitor model performance across training runs with the interactive Streamlit dashboard:

```bash
# Run the monitoring dashboard
streamlit run monitor_dashboard.py --server.port 8501
```

**Dashboard Features:**
- **Time Series Visualization**: Track validation RMSE and R² metrics over time
- **Candidate Comparison**: Compare top performing model candidates for any run date
- **Interactive Filtering**: Filter by date range and top-N results
- **Data Export**: Download filtered metrics data as CSV
- **Real-time Updates**: Refresh data directly from S3 with a button click

**Dashboard Components:**
- **Performance Metrics**: Display best RMSE, R², and run counts
- **Line Charts**: Dual-axis time series for RMSE (red) and R² (blue) trends
- **Bar Charts**: Top 5 model candidates comparison for selected dates
- **Data Tables**: Detailed metrics with sortable columns and formatting
- **Sidebar Controls**: Date range picker, top-N selector, and refresh button

**Setup Requirements:**
- Ensure AWS credentials are configured (environment variables or .env file)
- Metrics files must be stored in S3 at: `s3://convictionai-data/models/blender/metrics/`
- Dashboard automatically discovers and loads all `candidate-metrics.csv` files

**Access Dashboard:**
- Open browser to `http://localhost:8501` after running the command
- Dashboard updates automatically when new metrics files are added to S3

### Model Drift Detection

Automated monitoring system that detects model performance degradation and triggers retraining:

```bash
# Run drift detection manually
python detect_model_drift.py --threshold 0.10 --lookback 5

# Run with custom parameters
python detect_model_drift.py --threshold 0.15 --lookback 7 --verbose

# Test drift detection without triggering actions
python detect_model_drift.py --threshold 0.05 --lookback 3 --verbose
```

**Drift Detection Parameters:**
- `--threshold`: Percentage increase threshold for drift detection (default: 0.10 = 10%)
- `--lookback`: Number of previous runs to use as baseline (default: 5 runs)
- `--verbose`: Enable detailed logging for troubleshooting

**How Drift Detection Works:**
1. **Metrics Collection**: Loads the last `lookback+1` candidate-metrics.csv files from S3
2. **Baseline Calculation**: Computes average ValidationRMSE from previous `lookback` runs
3. **Drift Analysis**: Compares most recent RMSE to baseline using threshold percentage
4. **Alert System**: Sends Slack notification when drift exceeds threshold
5. **Auto-Retrain**: Triggers GitHub Actions `dispatch_retrain.yml` workflow automatically

**Automated Schedule:**
- **Lambda Function**: `lambda_drift_monitor.py` runs daily at 02:00 UTC via EventBridge
- **S3 Integration**: Automatically processes new metrics from `s3://convictionai-data/models/blender/metrics/`
- **GitHub Integration**: Uses `gh` CLI to trigger retrain workflows when drift detected
- **Slack Alerts**: Real-time notifications with drift percentage and threshold details

**Setup Requirements:**
- **Environment Variables**: `SLACK_WEBHOOK_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **GitHub CLI**: Install `gh` CLI and authenticate for workflow dispatch
- **AWS Permissions**: S3 read access to metrics bucket and Lambda execution role
- **S3 Structure**: Metrics stored as `models/blender/metrics/YYYY-MM-DD/candidate-metrics.csv`

**Exit Codes:**
- `0`: No drift detected, model performance is stable
- `1`: Drift detected, alerts sent and retraining triggered

**Example Output:**
```
2025-01-16 14:30:00 - INFO - Starting drift detection (threshold=10.0%, lookback=5)
2025-01-16 14:30:01 - INFO - Found 8 metrics files in S3
2025-01-16 14:30:02 - INFO - Run 2025-01-16: Best ValidationRMSE = 0.125000
2025-01-16 14:30:02 - INFO - Baseline RMSE (avg of 5 runs): 0.105000
2025-01-16 14:30:02 - WARNING - Drift detected: RMSE increased by 19.05%
2025-01-16 14:30:03 - INFO - Slack alert sent successfully
2025-01-16 14:30:04 - INFO - Retrain workflow triggered successfully
```

### Strategy & Trading Operations

```bash
# Generate nightly strategy recommendations
python strategy_selector.py --date 2025-07-22

# Run dry-run without saving to S3
python strategy_selector.py --date 2025-07-22 --dry-run

# Run comprehensive backtest analysis
python backtest_harness.py --start 2023-01-01 --end 2025-06-30

# Save backtest results to file
python backtest_harness.py --start 2024-01-01 --end 2024-12-31 --save-results backtest_2024.json

# Execute trades via Interactive Brokers (paper mode)
python ibkr_execute_trades.py \
  --reco-json s3://convictionai-data/reco/$(date +%F)/strategy_recs.json \
  --paper

# Execute trades via Interactive Brokers (live mode)
python ibkr_execute_trades.py \
  --reco-json s3://convictionai-data/reco/$(date +%F)/strategy_recs.json

# Execute trades from local file
python ibkr_execute_trades.py \
  --reco-json ./strategy_recs.json \
  --paper \
  --host 127.0.0.1 \
  --port 4002 \
  --client-id 1
```

**Required IBKR Environment Variables:**
```bash
export IBKR_HOST=127.0.0.1           # IBKR TWS/Gateway host
export IBKR_PORT=4002                # 4002 for paper, 4001 for live
export IBKR_CLIENT_ID=1              # Unique client ID
```

**IBKR Setup Notes:**
- Install Interactive Brokers TWS or Gateway
- Enable API connections in TWS/Gateway settings
- Use port 4002 for paper trading, 4001 for live trading
- Paper trading accounts typically start with 'D' (e.g., DU123456)
- Live trading accounts typically start with 'U' (e.g., U123456)

## License

[Specify your license here]

## Contact

[Your contact information]
