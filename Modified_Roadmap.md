# ML Training Roadmap — Conviction AI Swing Suite
*Last updated: 2025-07-27*

---

## 0 | Why a **Partitioned Parquet Dataset** (not one monolithic file)?

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Single Parquet file** | Simple path; easy to copy | • 10-100 GB single blob → slow writes<br>• Hard to append/incremental<br>• Risk of corruption kills entire set | ❌ *Avoid* |
| **Partitioned dataset** (`root/train_dataset/…`) | • Append-friendly (per-date)<br>• Spark/pyarrow can load lazily<br>• Checkpoints = “latest date written”<br>• Easy S3 sync & versioning | Slightly more paths to manage | ✅ **Chosen** |

We’ll write **one dataset root**, partitioned by `date=` directories (e.g. `date=2025-07-27/part-000.parquet`).
All script checkpoints already map cleanly to the *last successful partition*.

---

## 1 | Feature-Coverage Checklist (37 required columns)

| Bucket | Must-have columns | Script(s) | Status |
|--------|------------------|-----------|--------|
| **Macro / VIX / DXY** | `vix_value`, `vix_ma_divergence`, `dxy_value`, `iv_rank_30d` | `clean_macro_data.py`, `calculate_features.py` | ✅ READY |
| **News** | `news_count_lag1`, `avg_sentiment_lag1` | `clean_news.py` | ✅ READY |
| **Dealer Flow** | `gex_spx_lag1` | `feature_builder.dealer_flow()` | ❌ BLOCKED — need SPX GEX feed |
| **Options daily** | `optd_iv30`, `optd_hv30`, `optd_iv_skew_slope`, `optd_vol_surprise`, `optd_put_call_ratio`, `optd_volume` (+ all `_lag1`) | `clean_options_daily.py` | ✅ READY |
| **Options 30-min** | `opt30_mid_price_return`, `opt30_bid_ask_spread`, `opt30_implied_volatility`, `opt30_delta`, `opt30_theta`, `opt30_volume_return`, `opt30_rolling_vol_5`, `opt30_flow_divergence`, `opt30_gamma_squeeze` | `clean_options_30min.py` | ✅ READY |
| **Stocks daily** | `stockd_close`, `stockd_volume`, `stockd_return_1d`, `stockd_vol_7d`, `stockd_volume_pct_change`, `stockd_beta_spy`, `stockd_days_to_earnings`, `stockd_earnings_flag` (+ `_lag1` where noted) | `clean_stocks_daily.py` | ✅ READY |
| **Stocks 30-min** | `stock30_close_return`, `stock30_rolling_vol_5`, `stock30_is_last_30min` | `clean_stocks_30min.py` | ✅ READY |
| **Sector features** | `sector`, `is_tech`, `is_financial`, `is_energy`, `market_beta_20d` | `calculate_features.py` | ✅ READY |
| **Market direction** | `spy_return_1d`, `qqq_return_1d`, `spy_volume`, `qqq_volume` | `calculate_features.py` | ✅ READY |
| **Commodity refs** | `gold_return_1d`, `oil_return_1d`, `gold_volatility_10d` | `calculate_features.py` | ✅ READY |

*All columns (plus their lagged variants) are validated in `validate_feature_lagging.py`.*

---

## 2 | Execution Timeline (local iMac M2 Ultra)

| Day | Task | Script / Flow | Output | Status |
|-----|------|---------------|--------|--------|
| -1 | **Missing-feature close-out** (VIX MA, IV rank, etc.) | `dealer_flow.py` + feature fixes | New swing features | ✔ |
| 0 | **Smoke-test single date** | commands below | `train_dataset_<DATE>.parquet` | ✔ (completed) |
| 1 | **Historical back-fill 2021 → 2025** | `run_historical_backfill.py` | Partitioned dataset under `training/` | ⏳ |
| 2 | **Cross-val guardrail** | `train_and_evaluate.py --tune` | AUC < 0.75 | ☐ |
| 3 | **Train all L0 models** | `train_and_evaluate.py` | `models/*.pkl` | ☐ |
| 4 | **CI / unit tests** | `pytest` | CI badge green | ☐ |
| 5 | **Strategy back-test** | back-test scripts | `backtest_results.json` | ☐ |
| 6 | **(Optional) deploy endpoints** | SageMaker helper scripts | Live endpoint | ☐ |
| 7 | **Monitoring hooks** | SHAP / Prom scripts | Live alerts | ☐ |

---

## 3 | Smoke-Test (one trading day)

```bash
DATE=2025-07-23   # or any recent full session

python src/clean_stocks_30min.py  --date $DATE
python src/clean_stocks_daily.py  --date $DATE
python src/clean_options_30min.py --date $DATE
python src/clean_options_daily.py --date $DATE
python src/clean_macro_data.py    --date $DATE
python src/clean_news.py          --date $DATE

python src/calculate_features.py  --date $DATE --use-gpu
python src/generate_labels.py     --date $DATE

./scripts/generate-training-dataset.sh \
  data/Parquet_data/features_$DATE.parquet \
  data/Parquet_data/labels_$DATE.parquet \
  data/Parquet_data/train_dataset_$DATE.parquet

python validate_option_features.py \
  --input-path data/Parquet_data/train_dataset_$DATE.parquet
python validate_feature_lagging.py \
  --input-path data/Parquet_data/train_dataset_$DATE.parquet
