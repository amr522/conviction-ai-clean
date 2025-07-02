README.md

# Conviction AI Trading System 🚀

**2,662 trained models** across 242 stocks with **0.516 mean AUC** (ExtraTrees best).

## 🎯 Status: COMPLETE (2025-06-29)

| Model           | Count   | AUC   | Status |
|-----------------|---------|-------|--------|
| ExtraTrees      | 242/242 | 0.516 | ✅     |
| GradientBoost   | 242/242 | 0.512 | ✅     |
| Random Forest   | 242/242 | 0.511 | ✅     |
| XGBoost         | 242/242 | 0.511 | ✅     |
| CatBoost        | 242/242 | 0.511 | ✅     |
| Neural Networks | 242/242 | 0.507 | ✅     |
| FT-Transformer  | 242/242 | 0.499 | ✅     |

---

## 🚀 Quick Start

```bash
# 1. Check pipeline progress
python check_progress.py

# 2. Hyperparameter optimization (Optuna)
python run_hpo.py \
  --symbols all \
  --models lightgbm,xgboost,extra_trees \
  --n_trials 30

# 3. Stacking ensemble
python generate_oof.py \
  --symbols all \
  --base_models top3 \
  --cv 5

python train_stacking_meta.py \
  --oof_dir oof/ \
  --meta_model logistic

# 4. Backtesting
python backtest_pipeline.py \
  --pred_dir predictions/stacked \
  --cost_per_share 0.0005

# 5. Portfolio optimization
python portfolio_optimizer.py

# 6. Live API
python live_prediction_api.py


⸻

📁 Repository Structure

conviction-ai/
├── data/processed_with_news_20250628/    # 242 feature CSVs
├── models/full_inventory_20250629/       # 2,662 trained model files
├── outputs/                              # Reports & backtest results
└── predictions/stacked/                  # Ensemble prediction CSVs


⸻

🛠️ Installation

pip install pandas numpy scikit-learn lightgbm xgboost catboost torch optuna
pip install yfinance fastapi uvicorn cvxpy scipy


⸻

📊 Data Schema

Column	Description
timestamp	Trading day (NYSE)
open, high, low, close, volume	OHLCV data
return, volatility_5d, ma5, ma10, ma20	Price-based features
rsi_14, bb_upper, bb_lower	Technical indicators
vix_close, dxy_close	Macro features
news_cnt, news_sent	News sentiment
direction	Target (1 if next return > 0)


⸻

🎯 Key Features
	•	Complete model zoo: 11 algorithms × 242 stocks
	•	News sentiment integration
	•	Stacking ensembles with meta-learner
	•	Realistic backtesting with transaction costs
	•	FastAPI-based live prediction service
	•	Portfolio optimization module
	•	Drift monitoring and alerts

⸻

📈 Performance
	•	Best model: ExtraTrees (0.516 AUC)
	•	AUC range: 0.375 – 0.652
	•	Total training time: ~8 hours
	•	Data span: 4 years of OHLCV + news + macro

⸻

Below are details on production training, troubleshooting, and deployment.
For our future roadmap, see README2.md.

---

**`README2.md`**  
```markdown
# Roadmap & Future Work

Use this checklist when coordinating with Devine or planning the next phases:

1. **Finish Current HPO Sweep**  
   ```bash
   aws sagemaker describe-hyper-parameter-tuning-job \
     --hyper-parameter-tuning-job-name 46-models-final-1751428406 \
     --query 'TrainingJobStatusCounters'

Wait until Completed = 138, InProgress = 0.
	2.	Retrieve & Sanity-Check Best Models

aws s3 cp --recursive \
  s3://hpo-bucket-773934887314/56_stocks/46_models_hpo/best/ \
  models/hpo_best/46_models/

Spot-check one model locally:

MODEL_FILE=$(ls models/hpo_best/46_models/*.pkl | head -n 1)
python - << 'PYCODE'
import joblib
m = joblib.load("$MODEL_FILE")
print(m)
PYCODE


	3.	Train Regression / Stacking Ensemble

python train_regression_ensemble.py \
  --features-dir models/hpo_best/46_models \
  --out-dir models/regression_ensemble

Run quick CV and inspect residuals.

	4.	Generate Final Report

python generate_report.py \
  --input-dir models/hpo_best/46_models \
  --output-file DEVIN_46_models_report_final.md

Review per-symbol hyperparameters, scores, and ACU usage.

	5.	Deepen Hyperparameter Search (Optional)
	•	Edit config/hpo_config.yaml:

ResourceLimits:
  MaxNumberOfTrainingJobs: 230   # ≈5 trials per symbol
  MaxParallelTrainingJobs: 4


	•	Switch to Bayesian or Hyperband strategy.
	•	Relaunch tuning job when ready.

	6.	Optimize Parallelism & Cost
	•	Increase MaxParallelTrainingJobs to 8 or 16 if budget allows.
	•	Decrease concurrency for cost savings.
	7.	Automate & Monitor
	•	Add a polling script or GitHub Action to watch tuning status.
	•	Integrate Slack/email alerts via SNS/CloudWatch Events.
	8.	Productionize & Schedule
	•	Merge final hyperparameters into main or production branch.
	•	Schedule monthly retraining via GitHub Actions or EventBridge.
	•	Document deployment, rollback, and monitoring procedures.

```markdown
*Place `README.md` at your repo root for app usage, and `README2.md` alongside it for future plans.*```