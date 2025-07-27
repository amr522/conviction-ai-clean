#!/usr/bin/env bash
set -euo pipefail

# Enhanced training script with data drift monitoring and Slack alerts
# Raw macro data paths
RAW_FRED_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv"
RAW_VIX_JSON="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json"
RAW_DXY_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv"
RAW_NEWS_DIR="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news"

# Environment defaults
export WINDOW_DAYS=${WINDOW_DAYS:-30}
export USE_GPU=${USE_GPU:-false}
export N_JOBS=${N_JOBS:-$(nproc)}

DATE=${1:-$(date +%Y-%m-%d)}
END_DATE=${2:-$DATE}
N_TRIALS=${3:-50}

echo "🚀 Starting training pipeline for $DATE to $END_DATE"

# Create logs directory
mkdir -p logs

# Source Slack notification helper
source "$(dirname "$0")/slack_notify.sh"

# First run full pipeline with macro data and feature calculation
echo "🔄 Running full pipeline with macro data and feature calculation..."
if python src/run_full_pipeline.py --date "$DATE" \
    --raw-fred-csv "$RAW_FRED_CSV" \
    --raw-vix-json "$RAW_VIX_JSON" \
    --raw-dxy-csv "$RAW_DXY_CSV" \
    --raw-news-dir "$RAW_NEWS_DIR" 2>&1 | tee logs/pipeline_${DATE}.log; then
    
    echo "✅ Pipeline completed successfully"
    
    # Features are calculated within run_full_pipeline.py
    FEATURES_PATH="datasets/features_${DATE}.parquet"
    
    if [[ -f "$FEATURES_PATH" ]]; then
        echo "✅ Features available for training"
        
        # Run training with drift monitoring using feature matrix
        echo "📊 Running training and evaluation with engineered features..."
        if python src/train_and_evaluate.py \
            --start-date "$DATE" \
            --end-date "$END_DATE" \
            --feature-path "$FEATURES_PATH" \
            --tune \
            --n-trials "$N_TRIALS" \
            --n-jobs "$N_JOBS" 2>&1 | tee logs/training_${DATE}.log; then
            
            echo "✅ Training completed successfully"
    
            # Check for data drift in logs
            DRIFT=$(grep -c "Drift detected: True" logs/evidently_log.txt 2>/dev/null || echo "0")
            
            if [[ "$DRIFT" -gt 0 ]]; then
                echo "⚠️ Data drift detected!"
                DRIFT_REPORT="metrics/data_drift_report_${DATE}.html"
                
                # Send Slack alert for drift
                if [[ -n "${SECURITY_SLACK_WEBHOOK:-}" ]]; then
                    notify_security "DRIFT DETECTED" "Data drift found for $DATE. Report: $DRIFT_REPORT"
                fi
                
                echo "📄 Drift report available at: $DRIFT_REPORT"
            else
                echo "✅ No data drift detected"
                
                # Send success notification
                if [[ -n "${SECURITY_SLACK_WEBHOOK:-}" ]]; then
                    notify_security "TRAINING SUCCESS" "Training completed for $DATE. No drift detected."
                fi
            fi
            
        else
            echo "❌ Training failed"
            
            # Send failure notification
            if [[ -n "${SECURITY_SLACK_WEBHOOK:-}" ]]; then
                notify_security "TRAINING FAILED" "Training pipeline failed for $DATE. Check logs/training_${DATE}.log"
            fi
            
            exit 1
        fi
    else
        echo "❌ Features file not found: $FEATURES_PATH"
        
        # Send failure notification
        if [[ -n "${SECURITY_SLACK_WEBHOOK:-}" ]]; then
            notify_security "FEATURES MISSING" "Features file not found for $DATE: $FEATURES_PATH"
        fi
        
        exit 1
    fi
else
    echo "❌ Pipeline failed"
    
    # Send failure notification
    if [[ -n "${SECURITY_SLACK_WEBHOOK:-}" ]]; then
        notify_security "PIPELINE FAILED" "Full pipeline failed for $DATE. Check logs/pipeline_${DATE}.log"
    fi
    
    exit 1
fi

echo "🎯 Training pipeline completed for $DATE"