#!/bin/bash

# Robust HPO with auto-restart
echo "🚀 Starting Robust HPO with Auto-Restart"

# Function to handle interruptions
cleanup() {
    echo "🛑 Received interrupt signal"
    echo "💾 Saving current progress..."
    # Add any cleanup code here
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Models to train (priority order)
MODELS=("extra_trees" "lightgbm" "xgboost" "catboost" "random_forest")

# Continue until completion
while true; do
    echo "📊 Checking progress..."
    
    # Check if HPO is complete
    TOTAL_FILES=$(find models/hpo_20250629 -name "*.json" -o -name "*.pkl" -o -name "*.joblib" | wc -l)
    EXPECTED=1210  # 242 symbols × 5 models
    
    if [ "$TOTAL_FILES" -ge "$EXPECTED" ]; then
        echo "🎉 HPO Complete! ($TOTAL_FILES/$EXPECTED files)"
        break
    fi
    
    echo "📈 Progress: $TOTAL_FILES/$EXPECTED files"
    
    # Run HPO for each model
    for model in "${MODELS[@]}"; do
        echo "🔥 Training $model..."
        
        timeout 1800 python run_hpo.py \
            --symbols all \
            --models $model \
            --n_trials 30 \
            --resume \
            --batch_size 50 || echo "⚠️ $model batch interrupted"
        
        # Brief pause between models
        sleep 10
    done
    
    echo "🔄 Completed one full cycle, checking progress..."
    sleep 30
done

echo "✅ All HPO training completed!"
