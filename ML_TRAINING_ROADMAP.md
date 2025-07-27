# ML Training Roadmap — Conviction AI Swing Suite
*Last updated: 2025-01-15*

---

## 0 | Why a **Partitioned Parquet Dataset** (not one monolithic file)?

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Single Parquet file** | Simple path; easy to copy | • 10-100 GB single blob → slow writes<br>• Hard to append/incremental<br>• Risk of corruption kills entire set | ❌ *Avoid* |
| **Partitioned dataset** (`root/train_dataset/…`) | • Append-friendly (per-date)<br>• Spark/pyarrow can load lazily<br>• Checkpoints = simply "latest date written"<br>• Easy S3 sync & versioning | Slightly more paths to manage | ✅ **Chosen** |

👉 We'll write **one dataset root**, partitioned by `date=` directories (e.g. `date=2025-07-23/part-000.parquet`).
Check-points already in your scripts map cleanly to "last successful partition".

---

## 1 | Feature Coverage Checklist  (37 Required Columns)

| Bucket | Must-have columns | Script(s) | Status |
|--------|------------------|-----------|--------|
| Macro/VIX/DXY | `vix_value`, `vix_ma_divergence`, `dxy_value`, `iv_rank_30d` | `clean_macro_data.py`, `calculate_features.py` | ✅ **READY** |
| News | `news_count_lag1`, `avg_sentiment_lag1` | `clean_news.py`, `news_sentiment_features.py` | ❌ **BLOCKED** - no news data |
| Dealer Flow | `gex_spx_lag1` | `feature_builder.dealer_flow()` | ❌ **BLOCKED** - no SPX GEX feed |
| Options **daily** | `optd_iv30`, `optd_hv30`, `optd_iv_skew_slope`, `optd_vol_surprise`, `optd_put_call_ratio`, `optd_volume` (+ all `_lag1`) | `clean_options_daily.py` | ✅ **READY** |
| Options **30-min** | `opt30_mid_price_return`, `opt30_bid_ask_spread`, `opt30_implied_volatility`, `opt30_delta`, `opt30_theta`, `opt30_volume_return`, `opt30_rolling_vol_5`, `opt30_flow_divergence`, `opt30_gamma_squeeze` | `clean_options_30min.py` | ⚠️ **PARTIAL** - missing input data |
| Stocks **daily** | `stockd_close`, `stockd_volume`, `stockd_return_1d`, `stockd_vol_7d`, `stockd_volume_pct_change`, `stockd_beta_spy`, `stockd_days_to_earnings`, `stockd_earnings_flag` (+ `_lag1` where noted) | `clean_stocks_daily.py` | ⚠️ **PARTIAL** - missing input data |
| Stocks **30-min** | `stock30_close_return`, `stock30_rolling_vol_5`, `stock30_is_last_30min` | `clean_stocks_30min.py` | ⚠️ **PARTIAL** - missing input data |

*All listed columns **+ back-shifted variants** are asserted in `validate_feature_lagging.py`.*

---

## 2 | Execution Timeline   *(local iMac M2 Ultra)*

| Day | Task | Script / Flow | Output | Done |
|-----|------|---------------|--------|------|
| -1 | **Missing-Feature Close-out v1.0** | dealer_flow.py, enhanced options, VIX MA divergence, IV rank | New swing features | [x] |
| 0 | **Smoke-test single date** (e.g. 2025-07-23) | see § 3 | `train_dataset_2025-07-23.parquet` | [ ] |
| 1 | **Full back-fill 2018→present** | Prefect `BuildTrainingDatasetFlow` | Partitioned dataset under `training/` | [ ] |
| 2 | **Cross-val guardrail** | `run_cv_guardrail.py` | AUC < 0.75 OK | [ ] |
| 3 | **L0-A Direction** fine-tune | `train_l0_direction.py` | `models/l0_direction.pkl` | [ ] |
| 4 | **L0-C IV-HV spread** | `train_ivhv_spread.py` | `models/ivhv_spread.pkl` | [ ] |
| 5 | **L0-B EGARCH vol** | `train_egarch_volatility.py` | `models/egarch_vol.pkl` | [ ] |
| 6 | **L0-D Regime (BOCPD)** | `train_regime_detector.py` | `models/regime.pkl` | [ ] |
| 7-8 | **L0-E PatchTST tail risk** | `train_patchtst_tail.py` | `models/patchtst_tail.pt` | [ ] |
| 9 | All unit tests + CI push | `pytest` | CI ✅ | [ ] |
| 10 | **L1 Blender** training | `train_l1_blender.py` | `models/l1_blender.pkl` | [ ] |
| 11 | Strategy backtest | `backtest_covered_call_strategy.py` | `backtest_results.json` | [ ] |
| 12 | Deploy endpoints (opt.) | `deploy_blender_endpoint.py` | SageMaker endpoint | [ ] |
| 13 | Monitoring hooks | `setup_model_monitoring.py` | Prom/SHAP alerts | [ ] |

---

## 3 | Smoke-Test Commands (copy-paste)

```bash
DATE=2025-07-23
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

python validate_option_features.py   --input-path data/Parquet_data/train_dataset_$DATE.parquet
python validate_feature_lagging.py   --input-path data/Parquet_data/train_dataset_$DATE.parquet
```

Expect ≥ 200 columns, 0 % NaN in labels.

---

## 4 | Guardrail CV Script

```bash
python utils/run_cv_guardrail.py \
  --data-path data/Parquet_data/training \
  --method tscv --gap 390 --n-splits 5
```

AUC < 0.75 ⇒ no leakage.

---

## 5 | Key Recommendations

1. **Memory flags**: set `PYARROW_MEMORY_POOL=jemalloc` to avoid fragmentation on M2.
2. **GPU LightGBM**: install `lightgbm --install-option="--gpu"` (works on Apple Metal via OpenCL).
3. **Dataset versioning**: `aws s3 sync training/ s3://convictionai-data/training/ --delete` nightly; rely on S3 versioning for rollbacks.
4. **Rapid iteration**: during L0 dev, train on 2023-2024 slice (`date>=2023-01-01`) to cut runtime by 70%.
5. **CI gates**: block merge if `run_cv_guardrail` AUC > 0.75 or if `validate_feature_lagging` fails.
6. **Future proof**: add `opt30_iv_surface_skew` + `stockd_div_yield` in next feature revision.

---

## 6 | Backlog - Data Acquisition

### **BLOCKED Features - Require External Data**

| Feature | Data Source Needed | Priority | Estimated Effort |
|---------|-------------------|----------|------------------|
| `gex_spx_lag1` | SPX Gamma Exposure feed (CBOE/Bloomberg) | High | 2-3 days |
| `news_count_lag1`, `avg_sentiment_lag1` | News sentiment data feed | Medium | 1-2 days |
| VIX futures term structure | VIX futures historical data | Low | 3-5 days |

### **Next Data Acquisition Sprint**
- **Priority 1**: SPX GEX data for dealer flow signals
- **Priority 2**: News sentiment feed integration
- **Priority 3**: VIX futures for term structure analysis

---

End of Roadmap
