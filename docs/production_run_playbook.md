# Production Run Playbook

This playbook outlines the complete process for executing production runs of the Conviction-AI machine learning pipeline on M2 Ultra systems.

## 📋 Prerequisites & Preconditions

### Environment Variables

**Required:**
```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
export S3_BUCKET="conviction-ai-artifacts"
export S3_PREFIX="production/"

# Slack Notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX"

# MLflow Tracking
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="ConvictionAI-Production"
```

**Optional:**
```bash
# Telegram Alerts (monitoring)
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="-1001234567890"
```

### Data Availability Checks

**Verify data sources:**
```bash
# Check raw data availability
ls -la data/Parquet_data/$(date +%Y-%m-%d)/
aws s3 ls s3://conviction-data/raw/$(date +%Y-%m-%d)/

# Validate data freshness (< 24 hours old)
find data/Parquet_data -name "*.parquet" -mtime -1 | wc -l
```

### Branch & CI Status

**Pre-run validation:**
```bash
# Ensure on main branch with latest changes
git checkout main && git pull origin main

# Verify CI pipeline passed
gh run list --branch main --limit 1 --json status,conclusion

# Check for any pending schema validation issues
./scripts/run_and_inspect.sh --dry-run
```

## 🚀 Step-by-Step Execution

### 1. Pre-Flight Checks

```bash
# Validate M2 Ultra optimizations
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"
system_profiler SPDisplaysDataType | grep "Metal Support"

# Test environment connectivity
curl -s $SLACK_WEBHOOK_URL -X POST -H 'Content-type: application/json' \
  --data '{"text":"🧪 Pre-flight test from M2 Ultra"}' > /dev/null && echo "✅ Slack OK"

aws s3 ls s3://$S3_BUCKET/ > /dev/null && echo "✅ S3 OK"

curl -s $MLFLOW_TRACKING_URI/health > /dev/null && echo "✅ MLflow OK"
```

### 2. Execute Production Run

**Standard production run (yesterday's data):**
```bash
./scripts/run_and_train.sh
```

**Custom date with high-intensity training:**
```bash
./scripts/run_and_train.sh 2025-07-25 200 24
```

**Expected runtime:** 15-45 minutes depending on data size and trial count.

### 3. Monitor Execution

**Real-time log monitoring:**
```bash
# In separate terminal, monitor key metrics
tail -f /tmp/conviction-ai-$(date +%Y%m%d).log | grep -E "(RMSE|GPU|SUCCESS|FAILED)"
```

**M2 Ultra resource monitoring:**
```bash
# Monitor GPU utilization
sudo powermetrics -n 1 -s gpu_power | grep "GPU"

# Monitor CPU utilization
top -l 1 | grep "CPU usage"
```

## ✅ Validation & Smoke Tests

### 1. Schema Validation

```bash
# Validate all Parquet outputs
for ds in stocks_daily options_daily stocks_30min options_30min; do
  python src/validate_schemas.py --parquet-path staged/${ds}_clean.parquet --dataset-type $ds
done
```

### 2. Output File Checks

```bash
DATE=$(date -v-1d +%Y-%m-%d)  # Yesterday's date

# Verify local artifacts
ls -la models/latest.pkl
ls -la metrics/metrics_${DATE}_${DATE}.json
ls -la metrics/calibration.png

# Verify S3 artifacts
aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/models/$DATE/
aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/metrics/$DATE/
```

### 3. Model Performance Validation

```bash
# Extract and validate RMSE from metrics
RMSE=$(python -c "
import json
with open('metrics/metrics_${DATE}_${DATE}.json') as f:
    data = json.load(f)
    print(data['evaluation']['rmse'])
")

# RMSE threshold check (should be < 0.10 for production)
python -c "
rmse = float('$RMSE')
threshold = 0.10
if rmse > threshold:
    print(f'❌ RMSE {rmse:.4f} exceeds threshold {threshold}')
    exit(1)
else:
    print(f'✅ RMSE {rmse:.4f} within acceptable range')
"
```

### 4. MLflow Run Validation

```bash
# Get latest run ID and validate metrics
RUN_ID=$(python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
client = mlflow.tracking.MlflowClient()
runs = client.search_runs('1', order_by=['start_time DESC'], max_results=1)
print(runs[0].info.run_id if runs else 'No runs found')
")

echo "Latest MLflow run: $RUN_ID"
```

## 🔄 Rollback Procedures

### 1. Deactivate Faulty Model

```bash
# Deactivate latest model version in MLflow Registry
python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name='ConvictionAI_Swing_Model',
    version='latest',
    stage='Archived'
)
print('Model deactivated')
"
```

### 2. Revert S3 Artifacts

```bash
CURRENT_DATE=$(date +%Y-%m-%d)
PREVIOUS_DATE=$(date -v-1d +%Y-%m-%d)

# Backup current artifacts
aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/models/$CURRENT_DATE/ \
  s3://$S3_BUCKET/$S3_PREFIX/backup/models/$CURRENT_DATE/ --recursive

# Restore previous day's artifacts
aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/models/$PREVIOUS_DATE/ \
  s3://$S3_BUCKET/$S3_PREFIX/models/$CURRENT_DATE/ --recursive
```

### 3. Re-run with Previous Code

```bash
# Get last successful commit
LAST_GOOD_COMMIT=$(git log --oneline --grep="✅ Production run" -1 --format="%H")

# Checkout previous version
git checkout $LAST_GOOD_COMMIT

# Re-run pipeline
./scripts/run_and_train.sh

# Return to main after successful run
git checkout main
```

## 📊 Monitoring & Alerts

### Prometheus Metrics

**Key job names:**
- `vol_pipeline_last_run_status` - Pipeline success/failure
- `vol_pipeline_runtime_seconds` - Execution time
- `vol_pipeline_data_age_hours` - Data freshness

**Alert thresholds:**
```yaml
# Pipeline failure
- alert: PipelineFailure
  expr: vol_pipeline_last_run_status == 1
  for: 0m

# High latency (> 1 hour)
- alert: PipelineHighLatency
  expr: vol_pipeline_runtime_seconds > 3600
  for: 5m

# Tuning stalls (> 2 hours)
- alert: TuningStall
  expr: vol_pipeline_runtime_seconds > 7200
  for: 10m
```

### Slack Channels & Templates

**Channels:**
- `#conviction-ai-alerts` - Critical failures and alerts
- `#conviction-ai-runs` - Successful run notifications
- `#conviction-ai-monitoring` - Prometheus/Grafana alerts

**Critical incident template:**
```
🚨 CRITICAL: Conviction-AI Pipeline Failure

Date: $(date)
Run ID: $RUN_ID
Error: [ERROR_MESSAGE]
RMSE: [RMSE_VALUE]
Duration: [RUNTIME]

Actions Required:
1. Check logs: tail -f /tmp/conviction-ai-$(date +%Y%m%d).log
2. Validate data: ./scripts/run_and_inspect.sh
3. Consider rollback if needed

Runbook: docs/production_run_playbook.md
```

## 📝 Post-Run Tasks

### 1. Archive Logs

```bash
DATE=$(date +%Y-%m-%d)
LOG_DIR="logs/production/$DATE"
mkdir -p $LOG_DIR

# Archive execution logs
cp /tmp/conviction-ai-*.log $LOG_DIR/
cp metrics/metrics_*.json $LOG_DIR/

# Upload to S3
aws s3 cp $LOG_DIR/ s3://$S3_BUCKET/$S3_PREFIX/logs/$DATE/ --recursive
```

### 2. Tag Successful Run

```bash
# Tag current commit with successful run
DATE=$(date +%Y-%m-%d)
RMSE=$(python -c "
import json
with open('metrics/metrics_${DATE}_${DATE}.json') as f:
    print(json.load(f)['evaluation']['rmse'])
")

git tag -a "prod-$DATE" -m "✅ Production run $DATE - RMSE: $RMSE"
git push origin "prod-$DATE"
```

### 3. Stakeholder Notification

```bash
# Generate run summary
cat > run_summary.md << EOF
# Production Run Summary - $DATE

## Results
- **Status**: ✅ SUCCESS
- **RMSE**: $RMSE
- **Runtime**: [DURATION]
- **GPU Utilization**: [GPU_STATS]
- **CPU Cores Used**: 24

## Artifacts
- Model: s3://$S3_BUCKET/$S3_PREFIX/models/$DATE/
- Metrics: s3://$S3_BUCKET/$S3_PREFIX/metrics/$DATE/
- MLflow Run: $MLFLOW_TRACKING_URI/#/experiments/1/runs/$RUN_ID

## Next Steps
- Model deployed to production registry
- Monitoring dashboards updated
- Drift detection active
EOF

# Send to stakeholders
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"$(cat run_summary.md)\"}" \
  $SLACK_WEBHOOK_URL
```

## 🔧 M2 Ultra Optimizations

### GPU Acceleration
- **Automatic Detection**: Pipeline auto-detects M2 Ultra GPU
- **LightGBM GPU**: Uses Metal Performance Shaders
- **Monitoring**: `sudo powermetrics -s gpu_power`

### CPU Parallelism
- **Default Workers**: All 24 cores (8 performance + 16 efficiency)
- **Optuna Trials**: Parallel hyperparameter optimization
- **Data Loading**: Multi-threaded Parquet processing

### Memory Optimization
- **Unified Memory**: Leverages 64GB+ unified memory architecture
- **Streaming**: Processes data in chunks to avoid memory pressure
- **Garbage Collection**: Explicit cleanup between training phases

## 📚 Reference Documentation

- **Main Documentation**: [README.md](../README.md)
- **Pipeline Architecture**: [Option_parquet.md](../Option_parquet.md)
- **Schema Validation**: [src/validate_schemas.py](../src/validate_schemas.py)
- **Monitoring Setup**: [monitoring/README.md](../monitoring/README.md)
- **CI/CD Pipeline**: [.github/workflows/ci.yml](../.github/workflows/ci.yml)

## 🆘 Emergency Contacts

**On-Call Rotation:**
- Primary: [Your Team Lead]
- Secondary: [ML Engineer]
- Escalation: [Engineering Manager]

**Slack Channels:**
- `#conviction-ai-oncall` - Immediate response team
- `#conviction-ai-alerts` - Automated alerts
- `#conviction-ai-support` - General support

---

**Last Updated**: $(date)
**Version**: 1.0
**Maintained By**: Conviction-AI Team
