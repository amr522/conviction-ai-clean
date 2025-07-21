#!/bin/bash
# smoke_test_pipeline.sh - Runs the complete mock pipeline for smoke testing

# Exit on error
set -e

# Define script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR" && pwd )"

# Configure colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Conviction-AI Pipeline Smoke Test${NC}"
echo -e "${BLUE}=================================${NC}"
echo

# Check return code and report status
check_status() {
  if [ $1 -eq 0 ]; then
    echo -e "${GREEN}✅ Step $2 completed successfully${NC}"
  else
    echo -e "${RED}❌ Step $2 failed with exit code $1${NC}"
    exit $1
  fi
  echo
}

# Step 1: Run the ETL job (dry run)
echo -e "${YELLOW}Step 1: Running ETL job (dry run)${NC}"
echo "Command: python ${ROOT_DIR}/aws_pipeline/mock_glue_etl_script.py --JOB_NAME dry_run --raw-prefix s3://convictionai-data/conviction-ai/raw/ --output-path s3://convictionai-data/conviction-ai/clean/train_dataset_dry/ --sample-size 100 --null-threshold 0.2 --validate-schema"
echo

python ${ROOT_DIR}/aws_pipeline/mock_glue_etl_script.py \
  --JOB_NAME dry_run \
  --raw-prefix s3://convictionai-data/conviction-ai/raw/ \
  --output-path s3://convictionai-data/conviction-ai/clean/train_dataset_dry/ \
  --sample-size 100 \
  --null-threshold 0.2 \
  --validate-schema

check_status $? "1 (ETL Job)"

# Step 2: Run SageMaker Autopilot
echo -e "${YELLOW}Step 2: Running SageMaker Autopilot${NC}"
echo "Command: python ${ROOT_DIR}/aws_pipeline/mock_run_sagemaker_autopilot.py --target-column return --problem-type Regression --max-candidates 3"
echo

python ${ROOT_DIR}/aws_pipeline/mock_run_sagemaker_autopilot.py \
  --target-column return \
  --problem-type Regression \
  --max-candidates 3

check_status $? "2 (SageMaker Autopilot)"

# Step 3: Invoke the deployed endpoint
echo -e "${YELLOW}Step 3: Invoking SageMaker endpoint${NC}"
# Get the endpoint name from endpoint_info.json
if [ -f "${ROOT_DIR}/endpoint_info.json" ]; then
  ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' ${ROOT_DIR}/endpoint_info.json | cut -d'"' -f4)
  echo "Using endpoint name from endpoint_info.json: $ENDPOINT_NAME"
else
  ENDPOINT_NAME="returned_from_step_2"
  echo "No endpoint_info.json found, using placeholder endpoint name: $ENDPOINT_NAME"
fi

echo "Command: python ${ROOT_DIR}/aws_pipeline/mock_inference_from_sagemaker.py --endpoint-name $ENDPOINT_NAME --sample-size 3"
echo

python ${ROOT_DIR}/aws_pipeline/mock_inference_from_sagemaker.py \
  --endpoint-name "$ENDPOINT_NAME" \
  --sample-size 3

check_status $? "3 (Inference)"

# Step 4a: Cleanup (dry run)
echo -e "${YELLOW}Step 4a: Cleanup SageMaker resources (dry run)${NC}"
echo "Command: python ${ROOT_DIR}/aws_pipeline/mock_cleanup_sagemaker_resources.py --prefix conviction-automl --dry-run"
echo

python ${ROOT_DIR}/aws_pipeline/mock_cleanup_sagemaker_resources.py \
  --prefix conviction-automl \
  --dry-run

check_status $? "4a (Cleanup - Dry Run)"

# Step 4b: Actual cleanup
echo -e "${YELLOW}Step 4b: Cleanup SageMaker resources${NC}"
echo "Command: python ${ROOT_DIR}/aws_pipeline/mock_cleanup_sagemaker_resources.py --prefix conviction-automl"
echo

python ${ROOT_DIR}/aws_pipeline/mock_cleanup_sagemaker_resources.py \
  --prefix conviction-automl

check_status $? "4b (Cleanup)"

# Final summary
echo -e "${GREEN}===========================${NC}"
echo -e "${GREEN}🎉 All steps completed successfully!${NC}"
echo -e "${GREEN}===========================${NC}"
echo
echo "Summary of artifacts created:"
echo "- ETL statistics: ${ROOT_DIR}/etl_stats.json"
echo "- Inference results: ${ROOT_DIR}/inference_results.json"
echo "- Cleanup information: ${ROOT_DIR}/cleanup_info.json"
echo

exit 0
