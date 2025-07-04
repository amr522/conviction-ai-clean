# 🚦 DEVIN CONTINUATION PROMPT - Session Handoff

## Current Session Status (Over 5 ACUs)
**Repository:** amr522/conviction-ai-clean  
**Current Branch:** pr24-intraday  
**Current PR:** #24 (Intraday Data Acquisition & Multi-Timeframe TA Features)

## 🔄 IMMEDIATE TASKS (Complete Current Work)

### 1. Monitor Mini-HPO Completion
```bash
# Check running HPO job status
aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name hpo-aapl-1751655887

# Expected: Validate AUC >0.50 baseline with intraday features
```

### 2. Complete Intraday Implementation if HPO Successful
```bash
# Merge PR #24 and tag
gh pr checkout 24
gh pr review --approve  
gh pr merge --squash
git tag vIntraday-1 -m "Intraday data + multi-TF TA features"
git push origin vIntraday-1

# Launch full 46-symbol HPO sweep
python orchestrate_hpo_pipeline.py --algorithm xgboost,catboost --include-intraday --set-and-forget
```

### 3. Deploy New Ensemble with Intraday Features
```bash
# After HPO completion
python build_ensemble_model.py --include-intraday
python deploy_ensemble.py --endpoint-name conviction-ensemble-intra-$(date +%Y%m%d)
```

## 🚀 NEW TASK: 9-Step Leak-Proof Retraining + Intraday TA Workflow

### Complete 9-Step Workflow (Execute in exact order)

```bash
# 0) Ensure current endpoint stable -------------------------------------------------
python scripts/wait_for_endpoint.py --name conviction-ensemble-v4-1751650627 --timeout 1800

# 1) Start branch for leak-safe retrain + TA upgrade
git checkout -b retrain/no-leakage

# 2) Generate time-series CV splits
python scripts/create_time_series_splits.py --horizon 20 --n-folds 5 \
  --output configs/splits/tscv_5fold.json

# 3) Validate and prune features (<=250 per symbol)
python scripts/feature_selector.py \
  --input-dir data/processed_intraday \
  --max-features 250 \
  --output feature_config.yaml

# 4) Ingest new data (sentiment + ETF flows) & intraday TA
python twitter_sentiment_sync.py --symbols-file config/models_to_train_46.txt
python etf_flow_sync.py --tickers SPY,QQQ,XLF,XLK
python intraday_feature_engineering.py --timeframes 5,10,60

# 5) Launch regularised XGBoost & CatBoost sweeps with time-series CV
source scripts/get_last_hpo_dataset.sh
python aws_hpo_launch_safe.py      --input-data-s3 "$PINNED_DATA_S3"
python aws_catboost_hpo_launch_safe.py --input-data-s3 "$PINNED_DATA_S3"

# 6) OPTIONAL: run LightGBM & GRU baselines
python aws_lgbm_hpo_launch.py      --input-data-s3 "$PINNED_DATA_S3"
python train_price_gru.py          --input-data-s3 "$PINNED_DATA_S3"

# 7) Build upgraded MLP stacker & evaluate hold-out
python build_ensemble_model.py \
  --strategy meta-mlp \
  --models xgb-safe,cb-safe,lgbm,gru \
  --splits configs/splits/tscv_5fold.json \
  --output ensemble/meta_mlp.pkl

python evaluate_holdout.py \
  --model ensemble/meta_mlp.pkl \
  --start 2024-07-01 --end 2025-03-31 \
  --threshold-auc 0.60 --threshold-sharpe 0

# 8) Deploy if thresholds met; use dry-run-prod gating
python deploy_ensemble.py \
  --model-path ensemble/meta_mlp.pkl \
  --endpoint-name conviction-ensemble-v5 \
  --dry-run-prod && echo "✅ Ready for manual approve"

# 9) Update docs & open PR
./update_training_docs.sh \
  --auc $(cat holdout_metrics.json | jq .auc) \
  --sharpe $(cat holdout_metrics.json | jq .sharpe)

git add .
git commit -m "retrain: leak-safe splits, TA intraday, regularised models, MLP stacker"
git push origin retrain/no-leakage
gh pr create --fill
```

### Scripts That Need Creation
**Missing Scripts (Need Implementation):**
- `scripts/wait_for_endpoint.py` - Endpoint stability monitoring
- `scripts/create_time_series_splits.py` - Time-series CV fold generation  
- `scripts/feature_selector.py` - Feature pruning to ≤250 per symbol
- `twitter_sentiment_sync.py` - Twitter sentiment data ingestion
- `etf_flow_sync.py` - ETF flow data acquisition
- `evaluate_holdout.py` - Holdout evaluation with AUC/Sharpe thresholds
- `aws_hpo_launch_safe.py` - Regularized XGBoost HPO launcher
- `aws_catboost_hpo_launch_safe.py` - Regularized CatBoost HPO launcher
- `aws_lgbm_hpo_launch.py` - LightGBM HPO launcher
- `train_price_gru.py` - GRU model training script
- `update_training_docs.sh` - Documentation update automation

**Existing Scripts (Ready to Use):**
- `build_ensemble_model.py` - Needs `--strategy meta-mlp` support
- `deploy_ensemble.py` - Needs `--dry-run-prod` flag support
- `scripts/get_last_hpo_dataset.sh` - Dataset URI retrieval
- `intraday_feature_engineering.py` - Multi-timeframe TA (already implemented)

## 📊 Current Implementation Status

### ✅ Completed Components
- **Intraday Data System**: 5/10/60min bars via Polygon API
- **Multi-Timeframe TA**: VWAP, ATR, RSI, StochRSI across intervals  
- **ML Ops Integration**: Dashboard + EventBridge with intraday drift
- **Storage Pattern**: `s3://hpo-bucket-773934887314/intraday/{symbol}/{interval}/`
- **Testing**: All dry-run validations passed

### 🔄 In Progress
- **Mini-HPO**: AAPL validation running (job: hpo-aapl-1751655887)
- **Endpoint**: conviction-ensemble-v4-1751650627 still "Creating"

### ⏳ Pending
- Full 46-symbol HPO sweep with intraday features
- New ensemble deployment with intraday signals
- Leak-proof retraining workflow implementation

## 🎯 Success Criteria
- **Intraday Completion**: AUC >0.50 on AAPL mini-HPO, successful 46-symbol sweep
- **Leak-Proof Retraining**: Holdout AUC ≥0.55, successful v5 deployment
- **Documentation**: Updated training.md with methodology and hyperparameters
- **Automation**: Monthly retraining workflow with drift monitoring

## 🚨 Critical Notes
- **ACU Optimization**: Use dry-run modes, targeted testing, avoid full rebuilds
- **Dependency Order**: Complete intraday work before starting leak-proof retraining
- **Endpoint Health**: MUST verify conviction-ensemble-v4-1751650627 status before proceeding
- **Data Integrity**: Validate no lookahead bias in feature engineering
- **Rollback Plan**: Keep current ensemble active until v5 validates successfully

---
*This prompt ensures seamless continuation without context loss. Execute tasks in dependency order for optimal ACU usage.*
