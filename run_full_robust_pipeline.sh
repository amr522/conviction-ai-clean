#!/bin/bash
set -euo pipefail

# Full ML Pipeline with All Features
# This script runs the complete ML pipeline with all the robust features
# Usage: ./run_full_robust_pipeline.sh [--models-file FILE] [email-for-alerts]
#
# --models-file FILE: Optional path to a file containing a list of models to train
#                    Each line should contain a single model/symbol name

# Parse arguments
MODELS_FILE=""
EMAIL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --models-file)
      MODELS_FILE="$2"
      shift 2
      ;;
    *)
      # Assume remaining argument is email
      EMAIL="$1"
      shift
      ;;
  esac
done

echo "🚀 Starting full robust ML pipeline..."

# Check if models file is specified
if [ -n "$MODELS_FILE" ]; then
  if [ -f "$MODELS_FILE" ]; then
    echo "📋 Using models from file: $MODELS_FILE"
  else
    echo "❌ Models file not found: $MODELS_FILE"
    exit 1
  fi
fi

# Create necessary directories
mkdir -p data/processed_features
mkdir -p data/base_model_outputs/11_models
mkdir -p data/sagemaker/56_stocks
mkdir -p logs
mkdir -p data/analysis
mkdir -p data/per_stock
mkdir -p data/per_sector
mkdir -p data/time_series_cv
mkdir -p pipeline_logs

# Build the training command
TRAIN_CMD="./train_models_and_prepare_56_new.sh --use-aws --enhanced --hpo --feature-analysis --time-cv --per-stock --per-sector --deploy"

# Add models file if specified
if [ -n "$MODELS_FILE" ]; then
  TRAIN_CMD="$TRAIN_CMD --models-file $MODELS_FILE"
fi

# 1. Run the main training pipeline with all features enabled
echo "📊 Running main training pipeline..."
if $TRAIN_CMD; then
  echo "✅ Main training pipeline completed successfully"
else
  echo "❌ Main training pipeline failed"
  exit 1
fi

# 2. Register the model in SageMaker Model Registry
echo "📝 Registering model in Model Registry..."
# Extract model URI from SageMaker output
MODEL_URI=$(grep "model_uri:" pipeline_logs/latest.log | tail -n1 | awk '{print $2}')
MODEL_NAME="xgboost-56-stocks"

if [ -z "$MODEL_URI" ]; then
  echo "⚠️ Model URI not found in logs, using default"
  MODEL_URI="s3://${S3_BUCKET:-hpo-bucket-773934887314}/models/xgboost-56-stocks/latest"
fi

if python register_model.py register --model-name "$MODEL_NAME" --model-uri "$MODEL_URI" --description "XGBoost model for 56 stocks"; then
  echo "✅ Model registered successfully"
else
  echo "❌ Model registration failed"
  exit 1
fi

# 3. Create baseline statistics for monitoring
echo "📊 Creating model monitoring baseline..."
ENDPOINT_NAME=$(grep "endpoint_name:" pipeline_logs/latest.log | tail -n1 | awk '{print $2}')

if [ -z "$ENDPOINT_NAME" ]; then
  echo "⚠️ Endpoint name not found in logs, using default"
  ENDPOINT_NAME="xgboost-56-stocks-endpoint"
fi

BASELINE_DIR="s3://${S3_BUCKET:-hpo-bucket-773934887314}/baselines/$ENDPOINT_NAME"

if python create_monitoring_baseline.py --endpoint-name "$ENDPOINT_NAME" --output-s3-uri "$BASELINE_DIR"; then
  echo "✅ Monitoring baseline created successfully"
else
  echo "❌ Monitoring baseline creation failed"
  exit 1
fi

# 4. Set up monitoring and alerts if email is provided
if [ -n "$EMAIL" ]; then
  echo "🔔 Setting up monitoring and alerts..."
  
  if ./setup_model_alerts.sh "$ENDPOINT_NAME" "$EMAIL"; then
    echo "✅ Monitoring and alerts set up successfully"
  else
    echo "❌ Monitoring and alerts setup failed"
    exit 1
  fi
else
  echo "⚠️ No email provided, skipping monitoring and alerts setup"
fi

# 5. Clean up resources
echo "🧹 Cleaning up resources..."
if python cleanup_sagemaker_resources.py; then
  echo "✅ Resources cleaned up successfully"
else
  echo "❌ Resource cleanup failed"
  exit 1
fi

echo "🎉 Full robust ML pipeline completed successfully!"
echo "📈 You can monitor the model performance and receive alerts for any issues"
echo "📊 View model registry: ./list_registered_models.sh"
if [ -n "$EMAIL" ]; then
  echo "📧 Alerts will be sent to: $EMAIL"
fi
