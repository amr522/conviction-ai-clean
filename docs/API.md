# Conviction-AI API Documentation

## CLI Commands

### Core Pipeline Commands

#### Feature Calculation
```bash
# Calculate features for a single date
python src/calculate_features.py --date 2025-01-15

# Calculate features with GPU acceleration
python src/calculate_features.py --date 2025-01-15 --use-gpu

# Calculate features for date range
python src/calculate_features.py --date 2025-01-01,2025-01-31 --window-days 30
```

#### Model Training
```bash
# Basic training
python src/train_and_evaluate.py \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --model-path models/latest.pkl

# Training with hyperparameter optimization
python src/train_and_evaluate.py \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --tune \
  --n-trials 100 \
  --n-jobs 4

# Dry run (validation only)
python src/train_and_evaluate.py \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --dry-run
```

#### Pipeline Evaluation
```bash
# Evaluate pipeline for specific date
./scripts/evaluate_pipeline.sh 2025-01-15

# Run full validation suite
./scripts/run-all-validations.sh
```

### Validation Commands

#### Signal Validation
```bash
# Validate signal quality
python src/validate_signals.py --threshold 0.9

# Advanced signal validation
python src/validate_advanced_signals.py --threshold 0.8
```

#### Data Drift Detection
```bash
# Check data drift
python src/check_data_drift.py \
  --reference data/baseline.parquet \
  --current data/current.parquet \
  --drift-threshold 0.1
```

#### Schema Validation
```bash
# Validate against schema registry
python src/validate_schema_registry.py \
  --registry ConvictionAIPipelineRegistry \
  --schema-name feature_schema
```

### Production Commands

#### Deployment
```bash
# Complete production rollout
./scripts/complete-production-rollout.sh

# Verify production deployment
./scripts/verify-production.sh

# Production smoke tests
./scripts/production-smoke-test.sh 2025-01-15
```

#### Monitoring
```bash
# Monitor production
./scripts/monitor-production.sh

# Generate incident report
./scripts/generate-incident-report.sh
```

## Python API

### Feature Calculation

```python
from src.calculate_features import calculate_all_features
import polars as pl

# Load master datasets
daily_master = pl.read_parquet("data/daily_master.parquet")
intraday_master = pl.read_parquet("data/intraday_master.parquet")

# Calculate features
features = calculate_all_features(
    daily_master,
    intraday_master,
    window=30,
    use_gpu=True
)
```

### Model Training

```python
from src.train_and_evaluate import run

# Train model programmatically
exit_code = run(
    start_date="2025-01-01",
    end_date="2025-01-31",
    model_path="models/latest.pkl",
    metrics_path="metrics/",
    tune=True,
    n_trials=50,
    n_jobs=4
)
```

### Validation

```python
from src.validate_signals import validate_gamma_coverage, validate_flow_accuracy

# Validate signals
gamma_score = validate_gamma_coverage(df, threshold=0.9)
flow_score = validate_flow_accuracy(df, threshold=0.9)
```

## REST API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

### Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

## Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
export S3_BUCKET_NAME=your-bucket

# MLflow Tracking
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=ConvictionAI-Swing

# Slack Notifications
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# GPU Acceleration
export CUDA_VISIBLE_DEVICES=0
```

### Feature Flags

```bash
# Enable data drift detection
export DRIFT_ENABLED=true
export DRIFT_THRESHOLD=0.1

# Enable GPU acceleration
export USE_GPU=true

# Enable debug logging
export LOG_LEVEL=DEBUG
```

## Error Codes

| Code | Description |
|------|-------------|
| 0    | Success |
| 1    | General error |
| 2    | Data validation failed |
| 3    | Model training failed |
| 4    | Signal validation failed |
| 5    | Schema validation failed |

## Examples

### Complete Pipeline Run
```bash
# 1. Calculate features
python src/calculate_features.py --date 2025-01-15

# 2. Train model
python src/train_and_evaluate.py \
  --start-date 2025-01-01 \
  --end-date 2025-01-15 \
  --tune

# 3. Validate signals
python src/validate_signals.py

# 4. Deploy to production
./scripts/complete-production-rollout.sh
```

### Batch Processing
```bash
# Process multiple dates
for date in 2025-01-{01..31}; do
  python src/calculate_features.py --date $date
done
```

### GPU-Accelerated Training
```bash
# Enable GPU for training
python src/train_and_evaluate.py \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --tune \
  --n-jobs 1  # Use 1 job for GPU
```
