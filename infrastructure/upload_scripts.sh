#!/bin/bash
# upload_scripts.sh - Script to upload ETL and SageMaker scripts to S3

set -e  # Exit on error

# Default values
BUCKET_NAME=${S3_BUCKET_NAME:-"convictionai-data"}  # Use S3_BUCKET_NAME from env or default
REGION=${AWS_REGION:-"us-east-1"}  # Use AWS_REGION from env or default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket-name)
      BUCKET_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--bucket-name NAME] [--region REGION]"
      echo ""
      echo "Options:"
      echo "  --bucket-name NAME     Set the S3 bucket name (default: from S3_BUCKET_NAME env var or convictionai-data)"
      echo "  --region REGION        Set the AWS region (default: from AWS_REGION env var or us-east-1)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

echo "=== Conviction-AI Script Upload Utility ==="
echo "S3 Bucket Name: $BUCKET_NAME"
echo "AWS Region: $REGION"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if scripts exist
if [ ! -f "infrastructure/glue_etl_script.py" ]; then
    echo "Error: Glue ETL script not found at infrastructure/glue_etl_script.py"
    exit 1
fi

if [ ! -f "infrastructure/run_sagemaker_autopilot.py" ]; then
    echo "Error: SageMaker Autopilot script not found at infrastructure/run_sagemaker_autopilot.py"
    exit 1
fi

if [ ! -f "sagemaker-s3-policy.json" ]; then
    echo "Error: SageMaker S3 policy not found at sagemaker-s3-policy.json"
    exit 1
fi

# Create scripts directory in S3 bucket if it doesn't exist
echo "Creating scripts directory in S3 bucket..."
aws s3 ls s3://$BUCKET_NAME/scripts/ 2>&1 > /dev/null || aws s3api put-object --bucket $BUCKET_NAME --key scripts/ --region $REGION

# Upload Glue ETL script
echo "Uploading Glue ETL script..."
aws s3 cp infrastructure/glue_etl_script.py s3://$BUCKET_NAME/scripts/glue_etl_script.py --region $REGION

# Upload SageMaker Autopilot script
echo "Uploading SageMaker Autopilot script..."
aws s3 cp infrastructure/run_sagemaker_autopilot.py s3://$BUCKET_NAME/scripts/run_sagemaker_autopilot.py --region $REGION

# Upload SageMaker S3 policy
echo "Uploading SageMaker S3 policy..."
aws s3 cp sagemaker-s3-policy.json s3://$BUCKET_NAME/infrastructure/sagemaker-s3-policy.json --region $REGION

echo ""
echo "Script upload completed successfully!"
echo "Glue ETL Script: s3://$BUCKET_NAME/scripts/glue_etl_script.py"
echo "SageMaker Script: s3://$BUCKET_NAME/scripts/run_sagemaker_autopilot.py"
echo "SageMaker S3 Policy: s3://$BUCKET_NAME/infrastructure/sagemaker-s3-policy.json"
