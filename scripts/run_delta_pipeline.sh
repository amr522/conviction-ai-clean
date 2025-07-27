#!/usr/bin/env bash
set -euo pipefail

# Run pipeline with Delta Lake output
DATE=${1:-$(date +%Y-%m-%d)}
DRY_RUN=${2:-false}

echo "🔺 Running pipeline with Delta Lake for $DATE"

# Set Delta Lake environment variables
export S3_PREFIX="delta/"
export SPARK_HOME=${SPARK_HOME:-"/opt/spark"}

# Check if Spark is available
if ! command -v spark-submit &> /dev/null && [ ! -d "$SPARK_HOME" ]; then
    echo "⚠️ Spark not found. Installing PySpark..."
    pip install pyspark>=3.3.0
fi

# Run pipeline with Delta Lake enabled
if [ "$DRY_RUN" = "true" ]; then
    echo "🧪 Running in dry-run mode..."
    python src/run_full_pipeline.py \
        --date "$DATE" \
        --use-delta \
        --dry-run
else
    echo "🚀 Running full pipeline with Delta Lake..."
    python src/run_full_pipeline.py \
        --date "$DATE" \
        --use-delta
fi

if [ $? -eq 0 ]; then
    echo "✅ Delta Lake pipeline completed successfully!"
    echo ""
    echo "📋 Delta tables created:"
    echo "  - s3a://\${S3_BUCKET_NAME}/delta/stocks_daily.delta"
    echo "  - s3a://\${S3_BUCKET_NAME}/delta/options_daily.delta"
    echo "  - s3a://\${S3_BUCKET_NAME}/delta/stocks_30min.delta"
    echo "  - s3a://\${S3_BUCKET_NAME}/delta/options_30min.delta"
    echo "  - s3a://\${S3_BUCKET_NAME}/delta/intraday_master.delta"
    echo ""
    echo "🕰️ Time-travel queries:"
    echo "  python -c \"from src.utils.delta_writer import read_delta_table; df = read_delta_table('s3a://bucket/delta/stocks_daily.delta', version_as_of=0); print(df.head())\""
else
    echo "❌ Delta Lake pipeline failed"
    exit 1
fi