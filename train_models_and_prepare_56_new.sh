#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   USE_AWS=1 ./train_models_and_prepare_56_new.sh [--use-aws] [--enhanced] [--hpo] [--feature-analysis] [--time-cv] [--per-stock] [--per-sector] [--deploy] [--bundle-only] [--hpo-config FILE] [--models-file FILE]
#
# Environment:
#   USE_AWS         if set (or --use-aws passed) will push training to SageMaker
#   S3_BUCKET       defaults to hpo-bucket-773934887314
#   ENHANCED        if set (or --enhanced passed) will use enhanced training pipeline
#   HPO             if set (or --hpo passed) will run hyperparameter optimization
#   FEATURE_ANALYSIS if set (or --feature-analysis passed) will run SHAP feature analysis
#   TIME_CV         if set (or --time-cv passed) will use time-series cross-validation
#   PER_STOCK       if set (or --per-stock passed) will train per-stock models
#   PER_SECTOR      if set (or --per-sector passed) will train per-sector models
#   DEPLOY          if set (or --deploy passed) will deploy the model after training
#   BUNDLE_ONLY     if set (or --bundle-only passed) will only bundle the data without training
#   HPO_CONFIG      if set (or --hpo-config passed) specifies the HPO configuration file
#   MODELS_FILE     if set (or --models-file passed) specifies a file with models/symbols to train

# Check for sufficient disk space (at least 5GB free)
check_disk_space() {
  # Get available disk space in KB for the current directory
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    avail=$(df -k . | tail -1 | awk '{print $4}')
  else
    # Linux
    avail=$(df -k --output=avail . | tail -1)
  fi
  
  # Convert to MB (5GB = 5 * 1024 * 1024 = 5242880 KB)
  needed_kb=5242880
  
  if [ "$avail" -lt "$needed_kb" ]; then
    echo "❌ ERROR: Less than 5GB free disk space available (${avail}KB). Clean up before running."
    exit 1
  fi
  
  echo "✅ Sufficient disk space available: $((avail / 1024))MB"
}

# Run disk space check
check_disk_space

# Parse optional flags
USE_SAGEMAKER=false
USE_ENHANCED=false
USE_HPO=false
USE_FEATURE_ANALYSIS=false
USE_TIME_CV=false
USE_PER_STOCK=false
USE_PER_SECTOR=false
USE_DEPLOY=false
BUNDLE_ONLY=false
HPO_CONFIG=""
MODELS_FILE=""

# Pre-flight verification checks
run_preflight_checks() {
  echo "🚀 Running pre-flight checks..."
  
  # 1. Check AWS credentials
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ ERROR: Invalid AWS credentials"
    exit 1
  fi
  
  # 2. Ensure AWS_DEFAULT_REGION is set, or default to us-west-2
  : "${AWS_DEFAULT_REGION:=us-west-2}"
  echo "📍 Using AWS region: $AWS_DEFAULT_REGION"
  
  # 3. Verify S3 bucket and prefix
  S3_BUCKET="${S3_BUCKET:-hpo-bucket-773934887314}"
  S3_PREFIX="${S3_PREFIX:-56_stocks}"
  echo "🪣 Using S3 bucket: s3://$S3_BUCKET/$S3_PREFIX/"
  
  if ! aws s3 ls "s3://$S3_BUCKET/" >/dev/null 2>&1; then
    echo "❌ ERROR: S3 bucket not found: $S3_BUCKET"
    exit 1
  fi
  
  if ! aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" >/dev/null 2>&1; then
    echo "⚠️ WARNING: S3 prefix $S3_PREFIX/ does not exist yet in bucket $S3_BUCKET"
  fi
  
  # 4. Check models-to-train file if provided
  if [[ -n "${MODELS_FILE:-}" ]]; then
    if [[ ! -f "$MODELS_FILE" ]]; then
      echo "❌ ERROR: Models file $MODELS_FILE not found"
      exit 1
    fi
  fi
  
  # 5. Validate HPO config YAML
  if [[ -n "${HPO_CONFIG:-}" ]]; then
    if [[ ! -f "$HPO_CONFIG" ]]; then
      echo "❌ ERROR: HPO config file $HPO_CONFIG not found"
      exit 1
    fi
    
    # Use a temporary script to validate YAML
    cat > /tmp/validate_yaml.py << EOF
import sys, yaml
try:
    yaml.safe_load(open("$HPO_CONFIG"))
    print("✅ HPO config validation passed")
except Exception as e:
    print(f"❌ ERROR: Invalid HPO config: {e}")
    sys.exit(1)
EOF
    
    if ! python /tmp/validate_yaml.py; then
      exit 1
    fi
    
    rm /tmp/validate_yaml.py
  fi
  
  # 6. Skip IAM role check for local runs
  if $USE_SAGEMAKER; then
    # Allow custom role name via environment variable
    SAGEMAKER_ROLE="${SAGEMAKER_ROLE:-SageMakerExecutionRole}"
    echo "🔑 Using SageMaker IAM role: $SAGEMAKER_ROLE"
    
    if ! aws iam get-role --role-name "$SAGEMAKER_ROLE" >/dev/null 2>&1; then
      echo "❌ ERROR: IAM role $SAGEMAKER_ROLE not found"
      exit 1
    fi
  else
    echo "🏠 Running locally, skipping IAM role check"
  fi
  
  # 7. Ensure local directories exist or create them
  mkdir -p data/base_model_outputs/11_models data/sagemaker_input/11_models models/hpo_best/11_models
  
  echo "✅ Pre-flight checks passed"
}

# Run pre-flight checks
run_preflight_checks

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --use-aws)
      USE_SAGEMAKER=true
      shift
      ;;
    --enhanced)
      USE_ENHANCED=true
      shift
      ;;
    --hpo)
      USE_HPO=true
      shift
      ;;
    --feature-analysis)
      USE_FEATURE_ANALYSIS=true
      shift
      ;;
    --time-cv)
      USE_TIME_CV=true
      shift
      ;;
    --per-stock)
      USE_PER_STOCK=true
      shift
      ;;
    --per-sector)
      USE_PER_SECTOR=true
      shift
      ;;
    --deploy)
      USE_DEPLOY=true
      shift
      ;;
    --bundle-only)
      BUNDLE_ONLY=true
      shift
      ;;
    --hpo-config)
      HPO_CONFIG="$2"
      shift 2
      ;;
    --models-file)
      MODELS_FILE="$2"
      shift 2
      ;;
    --algorithms)
      ALGORITHMS="$2"
      shift 2
      ;;
    --sentiment-source)
      SENTIMENT_SOURCE="$2"
      shift 2
      ;;
  esac
done

# Also check environment variables
[[ "${USE_AWS:-}" == "1" ]] && USE_SAGEMAKER=true
[[ "${ENHANCED:-}" == "1" ]] && USE_ENHANCED=true
[[ "${HPO:-}" == "1" ]] && USE_HPO=true
[[ "${FEATURE_ANALYSIS:-}" == "1" ]] && USE_FEATURE_ANALYSIS=true
[[ "${TIME_CV:-}" == "1" ]] && USE_TIME_CV=true
[[ "${PER_STOCK:-}" == "1" ]] && USE_PER_STOCK=true
[[ "${PER_SECTOR:-}" == "1" ]] && USE_PER_SECTOR=true
[[ "${DEPLOY:-}" == "1" ]] && USE_DEPLOY=true
[[ "${BUNDLE_ONLY:-}" == "1" ]] && BUNDLE_ONLY=true
[[ -n "${HPO_CONFIG_FILE:-}" ]] && HPO_CONFIG="${HPO_CONFIG_FILE}"
[[ -n "${MODELS_FILE_PATH:-}" ]] && MODELS_FILE="${MODELS_FILE_PATH}"

# Print configuration
if $USE_SAGEMAKER; then
  echo "☁️  Will push jobs to SageMaker"
else
  echo "💻  Will run locally"
fi

if $BUNDLE_ONLY; then
  echo "📦 Will only bundle data without training"
fi

# Check if models file is specified
if [ -n "$MODELS_FILE" ]; then
  if [ -f "$MODELS_FILE" ]; then
    echo "📋 Using models from file: $MODELS_FILE"
  else
    echo "❌ ERROR: Models file not found: $MODELS_FILE"
    exit 1
  fi
fi

# Validate HPO config if specified
if $USE_HPO && [ -n "$HPO_CONFIG" ]; then
  if [ ! -f "$HPO_CONFIG" ]; then
    echo "❌ ERROR: HPO config file not found: $HPO_CONFIG"
    exit 1
  fi
  
  # Validate YAML syntax
  if ! python -c "import yaml, sys; yaml.safe_load(open('$HPO_CONFIG'))"; then
    echo "❌ ERROR: Invalid YAML syntax in HPO config file: $HPO_CONFIG"
    exit 1
  fi
fi

if $USE_ENHANCED; then
  echo "🚀 Will use enhanced training pipeline"
  
  echo "  🔧 Configuration:"
  $USE_HPO && echo "  - Hyperparameter optimization: Enabled"
  [ -n "$HPO_CONFIG" ] && echo "  - Using HPO config: $HPO_CONFIG"
  $USE_FEATURE_ANALYSIS && echo "  - Feature analysis with SHAP: Enabled"
  $USE_TIME_CV && echo "  - Time-series cross-validation: Enabled"
  $USE_PER_STOCK && echo "  - Per-stock model training: Enabled"
  $USE_PER_SECTOR && echo "  - Per-sector model training: Enabled"
  $USE_DEPLOY && echo "  - Model deployment: Enabled"
fi

S3_BUCKET="${S3_BUCKET:-hpo-bucket-773934887314}"
SAGEMAKER_ROLE="${SAGEMAKER_ROLE:-SageMakerExecutionRole}"

# Validate AWS credentials and resources if using SageMaker
if $USE_SAGEMAKER; then
  echo "🔍 Validating AWS credentials and resources..."
  
  # Check AWS credentials
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ ERROR: AWS credentials are invalid or not configured"
    exit 1
  fi
  
  # Check IAM role
  if ! aws iam get-role --role-name "$SAGEMAKER_ROLE" >/dev/null 2>&1; then
    echo "❌ ERROR: SageMaker IAM role not found: $SAGEMAKER_ROLE"
    exit 1
  fi
  
  # Check S3 bucket access
  if ! aws s3 ls "s3://$S3_BUCKET" >/dev/null 2>&1; then
    echo "❌ ERROR: S3 bucket not accessible: $S3_BUCKET"
    exit 1
  fi
  
  echo "✅ AWS credentials and resources validated"
fi

# Create necessary directories
mkdir -p data/processed_features_no_filter
mkdir -p data/base_model_outputs/11_models
mkdir -p data/sagemaker/56_stocks
mkdir -p logs
mkdir -p data/analysis
mkdir -p data/per_stock
mkdir -p data/per_sector
mkdir -p data/time_series_cv
mkdir -p pipeline_logs

echo "🔨 Step 1: Training 11 base models on local features"

# Count the symbols before training
if [ -n "$MODELS_FILE" ] && [ -f "$MODELS_FILE" ]; then
  symbol_count=$(wc -l < "$MODELS_FILE")
  echo "🔢 Training with $symbol_count symbols from $MODELS_FILE"
else
  symbol_count=$(wc -l < data/processed_features/processed_symbols.txt)
  echo "🔢 Training with $symbol_count symbols from processed_symbols.txt"
fi

# Add models file if specified
BASE_MODEL_CMD="python run_base_models.py \
     --features-dir data/processed_features \
     --symbols-file data/processed_features/processed_symbols.txt \
     --output-dir data/base_model_outputs/11_models"

# Add models-file flag if specified
if [ -n "$MODELS_FILE" ]; then
  BASE_MODEL_CMD="$BASE_MODEL_CMD --models-file $MODELS_FILE"
fi

# Add AWS flag if needed
if $USE_SAGEMAKER; then
  BASE_MODEL_CMD="$BASE_MODEL_CMD --use-aws"
fi

if $BASE_MODEL_CMD; then
  echo "✅ Base models trained"
else
  echo "❌ Base model training failed" >&2
  exit 1
fi

echo "📦 Step 2: Bundling SageMaker data for 56 stocks"
# prepare_sagemaker_data.py only accepts --input-file, --s3-bucket, --s3-prefix
# our list lives in data/processed_features/processed_symbols.txt
if python prepare_sagemaker_data.py \
  --input-file data/processed_features/all_symbols_features.csv \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix 56_stocks ; then

  # collect the generated s3_uris.txt into its own folder
  mkdir -p data/sagemaker/56_stocks
  mv data/sagemaker/s3_uris.txt data/sagemaker/56_stocks/

  echo "✅ Data preparation for 56 stocks completed"
else
  echo "❌ Data preparation failed" >&2
  exit 1
fi

echo "🔍 Step 2.5: Verifying data alignment"
# Download the training file from S3 if needed
if [ ! -f data/sagemaker/56_stocks/train.csv ]; then
  echo "Downloading training data from S3..."
  train_uri=$(grep "train:" data/sagemaker/56_stocks/s3_uris.txt | cut -d ' ' -f 2)
  aws s3 cp "$train_uri" data/sagemaker/56_stocks/train.csv
fi

# Check if the file exists now
if [ -f data/sagemaker/56_stocks/train.csv ]; then
  echo "=== Header Columns ==="
  head -n1 data/sagemaker/56_stocks/train.csv
  
  echo ""
  echo "=== Sample Rows (Ticker, Date, NewsExcerpt) ==="
  # Adjust this based on your actual data format - the awk command assumes 
  # column 1 is ticker, column 2 is date, and later columns contain news
  head -n5 data/sagemaker/56_stocks/train.csv | awk -F, '{print $1","$2","substr($0, index($0,$5))}'
  
  echo ""
  echo "=== Symbol Count ==="
  symbol_count=$(cut -d, -f1 data/sagemaker/56_stocks/train.csv | tail -n +2 | sort | uniq | wc -l)
  echo "Found $symbol_count unique symbols"
  
  # If models file is specified, filter the data
  if [ -n "$MODELS_FILE" ]; then
    echo "🔍 Filtering to only include models/symbols from $MODELS_FILE"
    model_count=$(wc -l < "$MODELS_FILE")
    echo "Found $model_count models/symbols in file"
    
    # Print the models being used
    echo "Models to be used:"
    cat "$MODELS_FILE"
  elif [ "$symbol_count" -eq 56 ]; then
    echo "✅ All 56 stocks present in training data"
  else
    echo "⚠️ Warning: Expected 56 stocks, but found $symbol_count"
  fi
  
  echo "✅ Data alignment verification complete"
else
  echo "❌ Training data file not found. Skipping verification."
fi

# Exit if bundle-only is specified
if $BUNDLE_ONLY; then
  echo "📦 Data bundling completed. Exiting as requested with --bundle-only flag."
  exit 0
fi

echo "🚀 Step 3: Launching training pipeline"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="56-stocks-xgboost-$TIMESTAMP"

# Add suffix if using subset of models
if [ -n "$MODELS_FILE" ]; then
  model_count=$(wc -l < "$MODELS_FILE")
  JOB_NAME="$model_count-stocks-xgboost-$TIMESTAMP"
fi

# Determine which training script to use
if $USE_ENHANCED; then
  # Build command line arguments for enhanced training
  ENHANCED_ARGS=""
  $USE_HPO && ENHANCED_ARGS="$ENHANCED_ARGS --hpo"
  $USE_FEATURE_ANALYSIS && ENHANCED_ARGS="$ENHANCED_ARGS --feature-analysis"
  $USE_TIME_CV && ENHANCED_ARGS="$ENHANCED_ARGS --time-cv"
  $USE_PER_STOCK && ENHANCED_ARGS="$ENHANCED_ARGS --per-stock"
  $USE_PER_SECTOR && ENHANCED_ARGS="$ENHANCED_ARGS --per-sector"
  $USE_DEPLOY && ENHANCED_ARGS="$ENHANCED_ARGS --deploy"
  [ -n "$MODELS_FILE" ] && ENHANCED_ARGS="$ENHANCED_ARGS --models-file $MODELS_FILE"
  [ -n "$ALGORITHMS" ] && ENHANCED_ARGS="$ENHANCED_ARGS --algorithms $ALGORITHMS"
  [ -n "$SENTIMENT_SOURCE" ] && ENHANCED_ARGS="$ENHANCED_ARGS --sentiment-source $SENTIMENT_SOURCE"
  
  echo "🧠 Using enhanced training pipeline with args: $ENHANCED_ARGS"
  
  if python enhanced_train_sagemaker.py \
       --data-dir data/sagemaker/56_stocks \
       --s3-bucket "$S3_BUCKET" \
       $ENHANCED_ARGS ; then
    echo "✅ Enhanced training pipeline completed successfully"
  else
    echo "❌ Enhanced training failed" >&2
    exit 1
  fi
else
  # Use the standard training script
  echo "📊 Using standard training pipeline"
  
  TRAIN_ARGS=""
  $USE_DEPLOY && TRAIN_ARGS="$TRAIN_ARGS --deploy"
  [ -n "$MODELS_FILE" ] && TRAIN_ARGS="$TRAIN_ARGS --models-file $MODELS_FILE"
  
  if python train_sagemaker_all_stocks.py $TRAIN_ARGS; then
    echo "✅ Combined 56-stock training completed successfully"
  else
    echo "❌ Combined training failed" >&2
    exit 1
  fi
fi

echo "🎉 All steps completed successfully!"
