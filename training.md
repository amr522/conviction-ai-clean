# 🚀 High-Accuracy Trading Models Roadmap

This README outlines an 8‑step plan to maximize model accuracy through advanced Technical Analysis (TA), institutional-grade data, and AI‑assisted rule discovery. It also identifies data gaps and how to acquire missing datasets.

---

## 1️⃣ TA-Heavy Feature Engineering
**Objective:** Enrich data with multi-timeframe TA indicators and volume/tick features.

- **Required Data:**
  - Intraday price bars (5‑, 15‑, 60‑min)
  - Volume & tick data for VWAP, OBV, MFI
  - Timestamped news headlines

- **Acquisition Sources:**
  - **Polygon.io** (minute bars & volume): `/v2/aggs/ticker/{symbol}/range/1/min/{from}/{to}`
  - **IEX Cloud** or **Tiingo** (tick/NBBO data)
  - **XAI News API** or **NewsAPI.org** for high-res news

- **Next:** Build `scripts/polygon_ingest.py` & `scripts/iex_ingest.py`.

---

## 2️⃣ Baseline & Deep TA Models
**Objective:** Leverage engineered TA features to train base models (XGBoost, CatBoost).

- **Data:** Uses features from Step 1 + existing `train.csv`.
- **Next:** Run nested HPO on TA hyperparameters & model params.

---

## 3️⃣ Ensemble & Regime Meta-Learner
**Objective:** Stack TA model outputs and regime signals to boost robustness.

- **Data:** Model predictions from Step 2 + regime flags (e.g. volatility regime).
- **Next:** Implement `scripts/build_ensemble_model.py`.

---

## 4️⃣ TA-Window Hyperparameter Ranges
**Objective:** Tune lookback/window lengths for TA indicators.

- **Data Span:** ≥720 trading days (3 years) of minute bars.
- **Next:** Expand HPO ranges to include window length parameters.

---

## 5️⃣ Drift Detection & Retraining Triggers
**Objective:** Monitor live feature distributions and trigger retraining on drift.

- **Required Data:** Hourly streams of TA metrics (ATR, RSI, etc.)
- **Acquisition:** Publish TA metrics via Kinesis → Lambda → CloudWatch.
- **Next:** Write `scripts/setup_drift_detection.sh` & CloudFormation template.

---

## 6️⃣ Profit-Driven Backtest Data
**Objective:** Simulate realistic trade P&L with slippage & fees.

- **Required Data:**
  - Transaction cost estimates
  - Historical bid/ask spreads

- **Acquisition:**
  - **Polygon NBBO** endpoints
  - **IEX Cloud** bid/ask data

- **Next:** Integrate into `scripts/backtest_with_costs.py`.

---

## 7️⃣ AI-Assisted Rule Discovery
**Objective:** Use LLMs to propose new TA-based rules and feature interactions.

- **Required Data:** Combined TA & price datasets in LLM‑friendly format.
- **Acquisition:** Use data from Steps 1 and 4.
- **Next:** Prepare prompt templates and connect to OpenAI GPT-4 or local LLaMA.

---

## 8️⃣ Orchestration & Deployment
**Objective:** Automate end‑to‑end pipeline from ingestion to live endpoint.

- **Scripts:** `scripts/orchestrate_hpo_pipeline.py` supports `--set-and-forget`.
- **Next:** Schedule via GitHub Actions or Cron; ensure idempotency & dry-run modes.

---

## 📊 Data Gaps & Acquisition
| Data                          | Purpose                                | Source / Script                        |
|-------------------------------|----------------------------------------|----------------------------------------|
| Intraday 5‑60 min bars       | Multi-TF TA indicators                 | Polygon.io (`scripts/polygon_ingest.py`)
| Tick / NBBO quotes            | Backtest slippage & spreads           | IEX Cloud (`scripts/iex_ingest.py`)
| High-res news headlines       | Event‑driven features                  | XAI News API (`scripts/news_ingest.py`)
| Twitter sentiment             | Social‑media features                  | Twitter API (`scripts/twitter_ingest.py`)
| FRED macro data               | Macro overlays (rates, CPI)            | FRED API (`scripts/fred_ingest.py`)
| Options flow imbalances       | Liquidity signals                       | Polygon options API (`scripts/options_ingest.py`)
| Economic calendar events      | Event lag/leads (FED, earnings)       | EconCalendar CSV (`scripts/events_ingest.py`)

---

## 🚀 Current Status & Implementation Progress

### ✅ Completed Components
- **HPO Automation System**: Full "set-and-forget" capabilities with XGBoost + CatBoost ensemble
- **ML Ops Dashboard**: Real-time monitoring with CloudWatch integration (PR #23 merged)
- **Automated Retraining**: EventBridge triggers for AUC < 0.50 and 7-day data cycles
- **AWS Infrastructure**: Account 773934887314, S3 bucket hpo-bucket-773934887314

### 🔄 Current Implementation Phase
**Phase 1: Intraday Data Acquisition & Multi-Timeframe TA**
- Target: 5min, 10min, 60min bars for all 46 symbols, 3 years historical
- Storage: `s3://hpo-bucket-773934887314/intraday/{symbol}/{interval}/YYYY-MM-DD.csv`
- Integration: Extend existing ML Ops dashboard for intraday drift monitoring

### 📋 Implementation Status

#### ✅ Phase 1: Intraday Data Acquisition (COMPLETED)
- **Scripts Created:**
  - `scripts/fetch_intraday_polygon.py` - Multi-interval data fetching from Polygon API
  - `scripts/feature_engineering_intraday.py` - Multi-timeframe TA feature engineering
- **Features Supported:** VWAP, ATR, RSI, StochRSI, Bollinger Bands, MACD, Volume indicators
- **Storage Pattern:** `s3://hpo-bucket-773934887314/intraday/{symbol}/{interval}/YYYY-MM-DD.csv`
- **Testing:** Dry-run mode implemented and tested

#### ✅ Phase 2: ML Ops Integration (COMPLETED)
- **Enhanced Dashboard:** Extended `mlops_dashboard.py` with intraday drift monitoring
- **EventBridge Rules:** Added intraday drift triggers in `setup_eventbridge.py`
- **CloudWatch Alarms:** Extended `setup_monitoring.py` with 5min/10min/60min drift alarms
- **CLI Interface:** Updated `mlops_cli.py` with enhanced dashboard support

#### 🔄 Phase 3: HPO & Algorithm Expansion (IN PROGRESS)
- **Algorithms Ready:** XGBoost, CatBoost, LightGBM, RandomForest, ExtraTreesClassifier
- **Current Status:** Mini-HPO running for AAPL validation (job: hpo-aapl-1751655887)
- **Next:** Complete AAPL validation → Full 46-symbol HPO sweep with intraday features
- **Integration:** Use existing `scripts/orchestrate_hpo_pipeline.py` framework

#### 📋 Current Session Status (Over 5 ACUs - Continuation Required)
1. ✅ Intraday data pipeline implemented and tested end-to-end
2. ✅ Multi-timeframe TA feature engineering completed
3. ✅ ML Ops dashboard extended with intraday drift monitoring
4. ✅ EventBridge automation updated with intraday triggers
5. 🔄 Mini-HPO validation running (AAPL with intraday features)
6. ⏳ Awaiting HPO completion to validate AUC >0.50 baseline
7. ⏳ Full 46-symbol HPO sweep pending validation success
8. ⏳ New ensemble deployment with intraday features

#### 🚦 Next Session Priority Tasks

**IMMEDIATE (Complete Current Intraday Work):**
- Monitor mini-HPO completion: `hpo-aapl-1751655887`
- Merge PR #24 if validation successful (AUC >0.50)
- Deploy new ensemble endpoint with intraday features

**NEW 9-STEP LEAK-PROOF RETRAINING WORKFLOW:**

```bash
# 0) Ensure current endpoint stable
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

### 🧪 Testing Commands
```bash
# Test intraday data acquisition
python scripts/fetch_intraday_polygon.py --intervals 5 10 60 --dry-run

# Test feature engineering
python scripts/feature_engineering_intraday.py --test --dry-run

# Test enhanced dashboard
python mlops_cli.py dashboard --endpoint-name conviction-ensemble-v4-1751650627 --enhanced --dry-run

# Test EventBridge setup
python setup_eventbridge.py setup --dry-run

# Test ML Ops monitoring
python setup_monitoring.py mlops --endpoint-name test-endpoint --topic-arn arn:aws:sns:us-east-1:123456789012:test --dry-run
```

### 🚀 Key Achievements
- **Multi-Timeframe TA:** 5min, 10min, 60min technical analysis features
- **Drift Detection:** Real-time monitoring of intraday feature distributions
- **Automated Triggers:** EventBridge rules for intraday drift-based retraining
- **Enhanced Dashboard:** Visual monitoring of all timeframe drift metrics
- **Scalable Architecture:** Built on existing ML Ops infrastructure

> With these components, we've created a comprehensive intraday trading system that monitors drift across multiple timeframes and automatically triggers retraining when needed! 🚀
