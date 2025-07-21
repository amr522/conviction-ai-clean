#!/bin/bash
# run_full_pipeline.sh - Executes the entire Conviction-AI pipeline
# This script runs all components of the pipeline in sequence:
# 1. AWS Pipeline (ETL, AutoML, etc.)
# 2. Stacking Pipeline
# 3. Tests

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Display the Conviction AI banner
python ascii_banner.py

# Function to print section headers
print_header() {
    echo ""
    echo "==============================================="
    echo "--- $1 ---"
    echo "==============================================="
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start of script
print_header "Starting Conviction-AI Full Pipeline"

# Check for AWS CLI
if ! command_exists aws; then
    echo "Error: AWS CLI is not installed or not in PATH"
    echo "Please install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check for Python
if ! command_exists python; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check for pytest
if ! python -c "import pytest" &>/dev/null; then
    echo "Warning: pytest is not installed. Will attempt to install it."
    pip install pytest pytest-mock
fi

# Make all scripts executable
print_header "Making scripts executable"
find ./aws_pipeline -name "*.sh" -exec chmod +x {} \;
find ./aws_pipeline -name "*.py" -exec chmod +x {} \;
chmod +x ./tests/run_tests.sh

# Load environment variables from .env file
print_header "Loading environment variables"
if [ -f .env ]; then
    echo "Loading from .env file..."
    set -a
    source .env
    set +a
    echo "Environment variables loaded successfully"
else
    echo "Warning: .env file not found. Make sure all required environment variables are set manually."
fi

# Check for required environment variables
required_vars=("AWS_REGION" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "S3_BUCKET_NAME" "SAGEMAKER_EXECUTION_ROLE")
missing_vars=0

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "Error: Required environment variable $var is not set"
        missing_vars=1
    fi
done

if [ $missing_vars -eq 1 ]; then
    echo "Please set all required environment variables in .env file or environment"
    exit 1
fi

# Display configuration
echo "Using the following configuration:"
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $S3_BUCKET_NAME"
echo "SageMaker Role: $SAGEMAKER_EXECUTION_ROLE"

# Run AWS Pipeline
print_header "Running AWS Pipeline"
./aws_pipeline/run_aws_pipeline.sh \
    --s3-bucket "$S3_BUCKET_NAME" \
    --max-runtime 2 \
    --max-candidates 5 \
    --instance-type ml.m5.large

# Extract the Autopilot job name from endpoint_info.json
if [ -f endpoint_info.json ]; then
    AUTOPILOT_JOB_NAME=$(grep -o '"job_name": "[^"]*' endpoint_info.json | cut -d'"' -f4)
    echo "Autopilot job name: $AUTOPILOT_JOB_NAME"
else
    echo "Error: endpoint_info.json not found. AWS Pipeline may have failed."
    exit 1
fi

# Run Stacking Pipeline with example arguments
print_header "Running Stacking Pipeline"
DEEP_MODEL_OOF_S3_PREFIX="s3://${S3_BUCKET_NAME}/conviction-ai/deep-model-oof/"
STACKING_DATA_S3_PREFIX="s3://${S3_BUCKET_NAME}/conviction-ai/stacking-data/"

./aws_pipeline/stacking_pipeline.sh \
    --autopilot-job-name "$AUTOPILOT_JOB_NAME" \
    --deep-model-oof-prefix "$DEEP_MODEL_OOF_S3_PREFIX" \
    --stacking-data-prefix "$STACKING_DATA_S3_PREFIX" \
    --iam-role-arn "$SAGEMAKER_EXECUTION_ROLE" \
    --s3-bucket "$S3_BUCKET_NAME" \
    --region "$AWS_REGION"

# Run tests
print_header "Running Tests"
python -m pytest tests/test_stacking_pipeline.py -q

# Check for stacked endpoint info
if [ -f stacked_endpoint_info.json ]; then
    STACKED_ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' stacked_endpoint_info.json | cut -d'"' -f4)
    echo "Stacked model endpoint: $STACKED_ENDPOINT_NAME"
else
    echo "Warning: stacked_endpoint_info.json not found. Stacking pipeline may have failed."
fi

print_header "Pipeline Summary"
echo "AWS Pipeline: Completed"
echo "Stacking Pipeline: Completed"
echo "Tests: Passed"
echo ""
echo "✅ Pipeline completed successfully!"
echo ""
echo "Endpoints deployed:"
[ -f endpoint_info.json ] && echo "- Autopilot endpoint: $(grep -o '"endpoint_name": "[^"]*' endpoint_info.json | cut -d'"' -f4)"
[ -f stacked_endpoint_info.json ] && echo "- Stacked model endpoint: $STACKED_ENDPOINT_NAME"
echo ""
echo "To make predictions, use the predict_with_endpoint.py script:"
echo "python aws_pipeline/predict_with_endpoint.py --endpoint-name <endpoint-name> --input-file <input-file>"
