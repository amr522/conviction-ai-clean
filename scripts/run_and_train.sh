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

# Telegram notification helper
send_telegram_alert() {
    local status="$1"
    local payload="$2"
    python -c "from src.telegram_alerts import send_message; send_message('$status','$payload')" 2>/dev/null || echo "⚠️ Telegram alert failed"
}

# First run full pipeline with macro data and feature calculation
echo "🔄 Running full pipeline with macro data and feature calculation..."
if python src/run_full_pipeline.py --date "$DATE" \
    --raw-fred-csv "$RAW_FRED_CSV" \
    --raw-vix-json "$RAW_VIX_JSON" \
    --raw-dxy-csv "$RAW_DXY_CSV" \
    --raw-news-dir "$RAW_NEWS_DIR" 2>&1 | tee logs/pipeline_${DATE}.log; then

    echo "✅ Pipeline completed successfully"

    # Generate feature parquet
    echo "👉 Generating feature Parquet for $DATE…"
    if python src/calculate_features.py --date "$DATE"; then
        echo "✅ Feature parquet generated"
    else
        echo "❌ Feature parquet generation failed"
        exit 1
    fi
    
    FEATURES_PATH="data/Parquet_data/features_${DATE}.parquet"
    LABELS_PATH="data/Parquet_data/labels_${DATE}.parquet"
    TRAIN_PATH="data/Parquet_data/train_dataset_${DATE}.parquet"
    
    # Generate training dataset if labels exist
    if [[ -f "$LABELS_PATH" ]]; then
        echo "🔗 Generating training dataset..."
        if ./scripts/generate-training-dataset.sh "$FEATURES_PATH" "$LABELS_PATH" "$TRAIN_PATH"; then
            echo "✅ Training dataset generated"
            FEATURES_PATH="$TRAIN_PATH"  # Use training dataset for training
        else
            echo "❌ Training dataset generation failed, using features only"
        fi
    else
        echo "⚠️ Labels file not found: $LABELS_PATH, using features only"
    fi

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
            
            # Compute SHAP explanations
            echo "🔍 Computing SHAP explanations..."
            if [[ -f "models/latest.pkl" && -f "$FEATURES_PATH" ]]; then
                python -c "
import sys, os
sys.path.insert(0, 'src')
from inference import explain_predictions, load_model
import polars as pl

try:
    model = load_model('models/latest.pkl')
    feats = pl.read_parquet('$FEATURES_PATH')
    pushgateway_url = os.getenv('PUSHGATEWAY_URL')
    shap_summary = explain_predictions(model, feats, pushgateway_url)
    print(f'✅ SHAP summary computed for {len(shap_summary)} features')
    if shap_summary:
        top_features = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        print('Top 3 features:', [f'{k}: {v:.4f}' for k, v in top_features])
except Exception as e:
    print(f'⚠️ SHAP computation failed: {e}')
"
            else
                echo "⚠️ Model or features not found, skipping SHAP explanations"
            fi

            # Check for data drift in logs
            DRIFT=$(grep -c "Drift detected: True" logs/evidently_log.txt 2>/dev/null || echo "0")

            if [[ "$DRIFT" -gt 0 ]]; then
                echo "⚠️ Data drift detected!"
                DRIFT_REPORT="metrics/data_drift_report_${DATE}.html"

                # Send Telegram alert for drift
                send_telegram_alert "DRIFT DETECTED" "Data drift found for $DATE. Report: $DRIFT_REPORT"

                echo "📄 Drift report available at: $DRIFT_REPORT"
            else
                echo "✅ No data drift detected"

                # Send success notification
                send_telegram_alert "TRAINING SUCCESS" "Training completed for $DATE. No drift detected."
            fi

        else
            echo "❌ Training failed"

            # Send failure notification
            send_telegram_alert "TRAINING FAILED" "Training pipeline failed for $DATE. Check logs/training_${DATE}.log"

            exit 1
        fi
    else
        echo "❌ Features file not found: $FEATURES_PATH"

        # Send failure notification
        send_telegram_alert "FEATURES MISSING" "Features file not found for $DATE: $FEATURES_PATH"

        exit 1
    fi
else
    echo "❌ Pipeline failed"

    # Send failure notification
    send_telegram_alert "PIPELINE FAILED" "Full pipeline failed for $DATE. Check logs/pipeline_${DATE}.log"

    exit 1
fi

echo "🎯 Training pipeline completed for $DATE"
