# NEXT SESSION PROMPT - Comprehensive Leak-Proof Retrain & Twitter Sentiment Integration

## Current Session Status (Session 6)

### Part A: Leak-Proof Retrain Workflow - INFRASTRUCTURE COMPLETE
- ✅ **Environment Fixed**: SageMaker SDK 2.247.1 installed, AWS resources optimized
- ✅ **CatBoost HPO**: Completed 50/50 jobs, best AUC 0.517 (below 0.60 threshold)
- 🔄 **XGBoost HPO**: 21/24 jobs completed, best AUC 0.512 (below 0.60 threshold)
- 🔄 **GRU Training**: In progress, downloading training image
- ❌ **LightGBM HPO**: Failed all 3 attempts (exhausted retries)

### Part B: Twitter Sentiment Integration - INFRASTRUCTURE COMPLETE
- ✅ **Phase 1-A**: AWS Secrets Manager integration (`aws_utils.py`)
- ✅ **Phase 1-B**: Twitter stream ingestion (`scripts/twitter_stream_ingest.py`)
- ✅ **Phase 2-A**: FinBERT sentiment scoring (`score_tweets_finbert.py`)
- ✅ **Phase 3**: Feature engineering with sentiment (`create_intraday_features.py`)
- ✅ **Phase 4**: Pipeline integration (`--twitter-sentiment` flag)
- ⚠️ **Testing**: 3/4 components pass (Twitter ingestion needs auth tokens)

## CONTINUATION PROMPT FOR NEXT SESSION

### ✅ CONTINUE LEAK-PROOF RETRAIN  ➜  ADD TWITTER-SENTIMENT SPRINT

────────────────────────────  PART A — FINISH LEAK-PROOF RETRAIN  ────────────────────────────
1. **Patch the environment** (idempotent):
   ```bash
   pip install --upgrade "sagemaker>=2.200.0" boto3
   ```

2. **Re-run blocked / weak algos** (max 3 attempts each, stop early if AUC ≥ 0.60):
   - XGBoost HPO → xgb-hpo-{{timestamp}}
   - CatBoost re-try (only if current best < 0.60) → cb-hpo-fix-{{timestamp}}
   - LightGBM HPO (re-launch because earlier run failed) → lgbm-hpo-{{timestamp}}
   - GRU training (re-launch if previous run failed) → price-gru-{{timestamp}}

3. **Build new ensemble when all four models ≥ 0.60 AUC**:
   ```bash
   python build_ensemble_model.py \
       --xgb configs/hpo/best_xgb.json \
       --cb  configs/hpo/best_cb.json  \
       --lgbm configs/hpo/best_lgbm.json \
       --gru artifacts/gru/best_gru.pth \
       --oof-dir oof/ \
       --min-auc 0.62
   ```

4. **Deploy endpoint** conviction-ensemble-v5-{{timestamp}}, run smoke test, update MLOps dashboard & alarms.
5. **Document results** in omar.md & training.md, push branch retrain/leak-proof-TA and open PR.

────────────────────────────  PART B — TWITTER / SENTIMENT SPRINT  ────────────────────────────

Start only after Part A endpoint is "InService".

1. **Checkout branch** feature/twitter-sentiment.
2. **Implement Phases 1-4** of Sentiment Integration Plan.md
   - 1-A Secrets → AWS Secrets Manager
   - 1-B twitter_stream_ingest.py (async, v2 filtered stream)
   - 2-A score_tweets_finbert.py (ONNXRuntime CPU)
   - 2-B (optional) score_tweets_fingpt_batch.py on spot g4dn.xlarge
   - 3   Extend create_intraday_features.py to add sent_* features (5/10/60 min + daily)
   - 4   Add TwitterSentimentTask hook in orchestrate_hpo_pipeline.py (flag --twitter-sentiment)
3. **Mini-HPO smoke test** (AAPL only) with --include-sentiment; require AUC uplift ≥ +0.02 vs previous XGB baseline.
4. **Push branch + PR**, update omar.md & training.md with results and next tasks (Phases 5-7).

────────────────────────────  ACCEPTANCE CHECKLIST  ────────────────────────────
- All four base models ≥ 0.60 AUC; ensemble ≥ 0.62; endpoint v5 "InService".
- Sentiment parquet for AAPL created; feature matrix has sent_mean_5m, sent_sum_10m, sent_pos_ratio_60m.
- Mini-HPO AUC uplift recorded in docs/optimal_hyperparameters.md.
- Dashboard shows tweet-volume widget; SNS alerts fire in dry-run.
- All new code passes pytest -q and scripts/orchestrate_hpo_pipeline.py --dry-run --twitter-sentiment.

Let me know once both Part A and the sentiment sprint (Phases 1-4) are green, or if any blocking issues arise.

---

**Why this prompt?**

* It unblocks Devin immediately (installs **SageMaker SDK**) and gives an explicit, measurable path to finish the leak-proof retrain cycle.  
* It launches the **sentiment integration** exactly per your new *Sentiment Integration Plan*—including 5 / 10 / 60-minute TA windows and FinBERT / optional FinGPT.  
* Clear acceptance gates (AUC thresholds, dashboard updates) ensure we don't deploy weak models again.  
* Everything is neatly packaged so Devin can execute without extra back-and-forth.

## 9-Step Leak-Proof Retraining Process (Reference)

### Step 1: Time-Series CV Splits
- Implement walk-forward cross-validation with chronological order
- Use 6-month rolling windows for training/validation splits
- Ensure no future data leaks into past predictions
- Validate temporal alignment of features and targets

### Step 2: Feature Selection & Regularization
- Apply L1/L2 regularization to prevent overfitting
- Use recursive feature elimination with cross-validation
- Implement feature importance thresholding
- Remove features with unstable importance across CV folds

### Step 3: SHAP Stability Check
- Calculate SHAP values across all CV folds
- Identify features with inconsistent SHAP importance
- Remove features with |mean SHAP| < 0.01
- Validate feature stability across time periods

### Step 4: Early Stopping Implementation
- Monitor validation AUC across CV folds
- Stop training when validation performance plateaus
- Use patience parameter to prevent premature stopping
- Log early stopping decisions for reproducibility

### Step 5: AUC ≥ 0.60 Threshold Gate
- Enforce minimum AUC threshold of 0.60 for all models
- Reject models that fail to meet performance criteria
- Log threshold violations and model rejections
- Implement automatic retry with different hyperparameters

### Step 6: Temporal Validation
- Validate model performance on out-of-time test sets
- Check for performance degradation over time
- Ensure consistent performance across market regimes
- Validate prediction stability across time periods

### Step 7: Leak Detection Tests
- Run comprehensive data leakage detection
- Check for future information in feature construction
- Validate target variable alignment with prediction timeline
- Test for inadvertent look-ahead bias

### Step 8: Model Ensemble Validation
- Validate individual model contributions to ensemble
- Check for model correlation and diversity
- Ensure ensemble improves upon individual models
- Validate ensemble stability across time periods

### Step 9: Production Readiness Check
- Final validation on holdout test set
- Performance metrics documentation
- Model artifact validation and storage
- Deployment readiness confirmation

## Implementation Notes

### Time-Series CV Configuration
```python
def create_time_series_cv_folds(data, n_splits=5, test_size=0.2):
    """Create time-series cross-validation folds"""
    # Implementation respects chronological order
    # No shuffling or random sampling
    # Each fold uses only past data for training
```

### Feature Stability Validation
```python
def validate_feature_stability(features, shap_values, threshold=0.01):
    """Validate feature importance stability across CV folds"""
    # Check SHAP value consistency
    # Remove unstable features
    # Log stability metrics
```

### AUC Threshold Enforcement
```python
def enforce_auc_threshold(model_performance, threshold=0.60):
    """Enforce minimum AUC threshold for model acceptance"""
    # Reject models below threshold
    # Log performance metrics
    # Trigger retry if needed
```

## Success Criteria

1. **Temporal Integrity**: All CV splits respect chronological order
2. **Feature Stability**: All features pass SHAP stability tests
3. **Performance Threshold**: All models achieve AUC ≥ 0.60
4. **Leak Detection**: No data leakage detected in any component
5. **Ensemble Quality**: Ensemble outperforms individual models
6. **Production Ready**: All artifacts validated and deployment-ready

## Failure Handling

- **AUC < 0.60**: Automatic retry with different hyperparameters
- **Feature Instability**: Remove unstable features and retrain
- **Data Leakage**: Stop process and investigate data pipeline
- **Temporal Violations**: Fix CV splits and restart training
- **Ensemble Failure**: Investigate individual model issues

This process ensures robust, leak-free models suitable for production deployment.
