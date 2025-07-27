#!/usr/bin/env bash
set -euo pipefail

# Register all pipeline tables in AWS Glue Data Catalog
S3_BUCKET=${S3_BUCKET_NAME:-"conviction-ai-data"}
S3_PREFIX=${S3_PREFIX:-"processed/"}
GLUE_DATABASE=${GLUE_DATABASE:-"conviction_ai"}
AWS_REGION=${AWS_REGION:-"us-east-1"}

echo "🗂️ Registering Parquet tables in AWS Glue Data Catalog"
echo "S3 Bucket: $S3_BUCKET"
echo "S3 Prefix: $S3_PREFIX"
echo "Glue Database: $GLUE_DATABASE"
echo "AWS Region: $AWS_REGION"

# Check AWS credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' or set environment variables."
    exit 1
fi

# Register tables
python src/utils/glue_catalog.py \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix "$S3_PREFIX" \
    --database "$GLUE_DATABASE" \
    --region "$AWS_REGION"

if [ $? -eq 0 ]; then
    echo "✅ All tables registered successfully!"
    echo ""
    echo "📋 Query tables with Athena:"
    echo "   SELECT * FROM $GLUE_DATABASE.stocks_daily LIMIT 10;"
    echo "   SELECT * FROM $GLUE_DATABASE.options_daily LIMIT 10;"
    echo "   SELECT * FROM $GLUE_DATABASE.intraday_master LIMIT 10;"
    echo ""
    echo "🔗 Athena Console: https://console.aws.amazon.com/athena/"
else
    echo "❌ Table registration failed"
    exit 1
fi