# ML Training Roadmap — Conviction AI Swing Suite
*Last updated: 2025-01-16*

---

## 0 | Why a **Partitioned Parquet Dataset** (not one monolithic file)?

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Single Parquet file** | Simple path; easy to copy | • 10-100 GB single blob → slow writes<br>• Hard to append/incremental<br>• Risk of corruption kills entire set | ❌ *Avoid* |
| **Partitioned dataset** (`root/train_dataset/…`) | • Append-friendly (per-date)<br>• Spark/pyarrow can load lazily<br>• Checkpoints = simply "latest date written"<br>• Easy S3 sync & versioning | Slightly more paths to manage | ✅ **Chosen** |

👉 We'll write **one dataset root**, partitioned by `date=` directories (e.g. `date=2025-01-16/part-000.parquet`).
Check-points already in your scripts map cleanly to "last successful partition".

---

## 1 | Feature Coverage Checklist  (37 Required Columns)

| Bucket | Must-have columns | Script(s) | Status |
|--------|------------------|-----------|--------|
| Macro/VIX/DXY | `vix_value`, `vix_ma_divergence`, `dxy_value`, `iv_rank_30d` | `clean_macro_data.py`, `calculate_features.py` | ✅ **READY** |
| News | `news_count_lag1`, `avg_sentiment_lag1` | `clean_news.py` | ✅ **READY** |
| Dealer Flow | `gex_spx_lag1` | `feature_builder.dealer_flow()` | ❌ **BLOCKED** - no SPX GEX feed |
| Options **daily** | `optd_iv30`, `optd_hv30`, `optd_iv_skew_slope`, `optd_vol_surprise`, `optd_put_call_ratio`, `optd_volume` (+ all `_lag1`) | `clean_options_daily.py` | ✅ **READY** |
| Options **30-min** | `opt30_mid_price_return`, `opt30_bid_ask_spread`, `opt30_implied_volatility`, `opt30_delta`, `opt30_theta`, `opt30_volume_return`, `opt30_rolling_vol_5`, `opt30_flow_divergence`, `opt30_gamma_squeeze` | `clean_options_30min.py` | ✅ **READY** |
| Stocks **daily** | `stockd_close`, `stockd_volume`, `stockd_return_1d`, `stockd_vol_7d`, `stockd_volume_pct_change`, `stockd_beta_spy`, `stockd_days_to_earnings`, `stockd_earnings_flag` (+ `_lag1` where noted) | `clean_stocks_daily.py` | ✅ **READY** |
| Stocks **30-min** | `stock30_close_return`, `stock30_rolling_vol_5`, `stock30_is_last_30min` | `clean_stocks_30min.py` | ✅ **READY** |
| **Sector Features** | `sector`, `is_tech`, `is_financial`, `is_energy`, `market_beta_20d` | `calculate_features.py` | ✅ **READY** |
| **Market Direction** | `spy_return_1d`, `qqq_return_1d`, `spy_volume`, `qqq_volume` | `calculate_features.py` | ✅ **READY** |
| **Commodity Refs** | `gold_return_1d`, `oil_return_1d`, `gold_volatility_10d` | `calculate_features.py` | ✅ **READY** |

*All listed columns **+ back-shifted variants** are asserted in `validate_feature_lagging.py`.*

---

## 2 | Execution Timeline   *(local iMac M2 Ultra)*

| Day | Task | Script / Flow | Output | Done |
|-----|------|---------------|--------|------|
| -1 | **Missing-Feature Close-out v1.0** | dealer_flow.py, enhanced options, VIX MA divergence, IV rank | New swing features | [x] |
| 0 | **Smoke-test single date** (e.g. 2025-01-16) | see § 3 | `train_dataset_2025-01-16.parquet` | [x] ← **COMPLETED** |
| 0.1 | **Create and test pipeline scripts** | `single_day_pipeline.sh`, `single_day_pipeline_standalone.sh`, `single_day_pipeline_manual.sh` | Automated pipeline execution | [x] ← **COMPLETED** |
| 1 | **Full back-fill 2021→2025** | `run_historical_backfill.py` | Partitioned dataset under `training/` | [ ] ← **READY TO START** |
| 2 | **Cross-val guardrail** | `train_and_evaluate.py --tune` | AUC < 0.75 OK | [ ] |
| 3 | **L0 Models** training | `train_and_evaluate.py` (existing) | `models/latest.pkl` | [ ] |
| 4 | All unit tests + CI push | `pytest` | CI ✅ | [ ] |
| 5 | Strategy backtest | Use existing backtest tools | `backtest_results.json` | [ ] |
| 6 | Deploy endpoints (opt.) | Use existing SageMaker tools | SageMaker endpoint | [ ] |
| 7 | Monitoring hooks | Use existing monitoring | Prom/SHAP alerts | [ ] |

---

## 3 | Smoke-Test Commands (copy-paste)

### Option 1: Automated Pipeline Scripts (M2 Ultra Optimized)

```bash
# Full pipeline (recommended) - runs complete validation → master datasets → features → labels → training dataset
# Automatically uses all 24 cores and 64GB RAM optimizations
./scripts/single_day_pipeline.sh

# Standalone mode - requires existing master datasets
# Optimized for M2 Ultra with GPU acceleration
./scripts/single_day_pipeline_standalone.sh

# Manual mode - specify date explicitly
# Full hardware optimization enabled
./scripts/single_day_pipeline_manual.sh 2025-07-27
```

**M2 Ultra Optimizations Enabled:**
- ✅ **24 CPU Cores**: `POLARS_MAX_THREADS=24`
- ✅ **64GB RAM**: Optimized memory allocation with jemalloc
- ✅ **Apple Metal GPU**: MPS backend for GPU acceleration
- ✅ **Parallel Processing**: `N_JOBS=24` for concurrent operations
- ✅ **Large Chunks**: 50,000 row chunks optimized for 64GB RAM

### Option 2: Manual Step-by-Step

```bash
DATE=2025-01-16  # Use current date
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
# Use existing train_and_evaluate.py with time-series CV
python src/train_and_evaluate.py \
  --start-date 2023-01-01 --end-date 2024-12-31 \
  --tune --n-trials 10 \
  --dry-run
```

AUC < 0.75 ⇒ no leakage.

---

## 5 | Key Recommendations

1. **M2 Ultra Optimization**: Optimized for 24-core Apple Silicon with 64GB RAM
   - `POLARS_MAX_THREADS=24` for maximum parallel processing
   - `PYARROW_MEMORY_POOL=jemalloc` for efficient memory allocation
   - Streaming chunk size optimized for 64GB RAM capacity
2. **GPU LightGBM**: install `lightgbm --install-option="--gpu"` (works on Apple Metal via OpenCL).
3. **Dataset versioning**: `aws s3 sync training/ s3://convictionai-data/training/ --delete` nightly; rely on S3 versioning for rollbacks.
4. **Rapid iteration**: during L0 dev, train on 2023-2024 slice to cut runtime by 70% (2 years vs 4 years).
5. **CI gates**: block merge if `train_and_evaluate.py` AUC > 0.75 or if `validate_feature_lagging` fails.
6. **Training Universe**: 36 stocks with complete intraday + daily + options data.
7. **Apple Metal GPU**: Leverage M2 Ultra GPU for NVDA/AMD model training with MPS backend.
8. **High IV Priority**: Weight TSLA, NVDA, PLTR higher in training (high volatility = more signal).
9. **Multi-Target Models**: Single model for stock + options predictions.
10. **Sector-Specific Models**: Separate TECH vs FINANCIAL vs HIGH_IV models.
11. **Feature Store**: Cache sector/market features for all 36 stocks.
12. **Parallel Processing**: Daily updates with 24-core parallel processing for speed.

---

## 6 | Backlog - Data Acquisition

### **BLOCKED Features - Require External Data**

| Feature | Data Source Needed | Priority | Estimated Effort |
|---------|-------------------|----------|------------------|
| `gex_spx_lag1` | SPX Gamma Exposure feed (CBOE/Bloomberg) | High | 2-3 days |
| ~~`news_count_lag1`, `avg_sentiment_lag1`~~ | ~~News sentiment data feed~~ | ~~Medium~~ | ~~COMPLETED~~ |
| VIX futures term structure | VIX futures historical data | Low | 3-5 days |

### **Next Data Acquisition Sprint**
- **Priority 1**: SPX GEX data for dealer flow signals
- **Priority 2**: ~~News sentiment feed integration~~ ✅ **COMPLETED**
- **Priority 3**: VIX futures for term structure analysis

---

## 7 | News Processing Implementation

### **Available Scripts**

1. **`src/clean_news.py`** - Single-day news processing
   ```bash
   python src/clean_news.py --date 2025-05-05
   ```

2. **`src/build_news_features.py`** - Multi-day news processing
   ```bash
   python src/build_news_features.py --start-date 2025-01-01 --end-date 2025-01-31
   ```

### **Generated Features**
- **`news_count_lag1`**: Count of news articles per ticker (lagged by 1 day)
- **`avg_sentiment_lag1`**: Average sentiment score per ticker (lagged by 1 day)
  - Positive sentiment: +1.0
  - Neutral sentiment: 0.0
  - Negative sentiment: -1.0

### **Data Source**
- **Location**: `data/Parquet_data/Raw/news/YYYY/MM/DD/news_data.json`
- **Coverage**: January 2025 (current month)
- **Articles per day**: ~200 articles
- **Tickers covered**: ~300-500 unique tickers per day

### **Anti-Leakage Design**
- All features are lagged by 1 day to prevent forward-looking bias
- First day of any sequence will have null lag features (expected behavior)
- Sentiment scores extracted from article insights, not headlines

---

---

## 8 | Enhanced Feature Strategy

### **Training Universe: 36 Stocks with Complete Data Coverage**
- **Tech**: AAPL, GOOGL, META, MSFT, NFLX, SMCI
- **Tech (High IV)**: AMD, NVDA, PLTR
- **Financial**: BAC, GS, JPM, MA, MS, V
- **Healthcare**: ABBV, JNJ, MRK, PFE, UNH
- **Consumer**: DIS, NKE, SBUX, WMT, AMZN
- **Energy**: CVX, XOM
- **Industrial**: BA, CAT, GE
- **Crypto/Fintech (High IV)**: COIN, HOOD, MSTR
- **Auto (High IV)**: TSLA
- **🎯 ETF Options Targets**: QQQ, SPY (for your heavy ETF trading)

### **Data Timeframes Used**
- **Daily Options**: End-of-day options data for all 36 stocks
- **Daily Stocks**: End-of-day stock data for all 36 stocks
- **30-Minute Options**: Intraday 30-min bar options data for all 36 stocks
- **30-Minute Stocks**: Intraday 30-min bar stock data for all 36 stocks

### **ETFs as Reference Features**
- **Market Direction**: SPY, QQQ → `spy_return_1d`, `qqq_return_1d` + **Options Signals**
- **Sector Rotation**: XLK, XLF, XLV, XLE → sector relative strength

---

## 9 | Files Requiring Updates (Based on Pipeline Analysis)

### **Immediate Priority - Update Existing Files for 36-Stock Universe**
1. **`src/calculate_features.py`** - Add PLTR to ticker list and high IV classification
2. **`validate_feature_lagging.py`** - Update expected ticker universe
3. **`validate_option_features.py`** - Update validation for 36 stocks

### **Use Existing Training Infrastructure**
- **`src/train_and_evaluate.py`** - Already has comprehensive training pipeline with:
  - Hyperparameter optimization (Optuna)
  - GPU support
  - MLflow tracking
  - Feature importance monitoring
  - Cross-validation capabilities

### **Testing Updates**
4. **`tests/test_calculate_features.py`** - Update for 36-stock universe
5. **`tests/test_performance_utils.py`** - Add PLTR to high IV tests
6. **`feature_repo/entities.py`** - Update ticker entities

**Summary: 6 files to update, 0 new files to create**

The existing pipeline already has all necessary infrastructure. Next step: Update `src/calculate_features.py` to add PLTR to the 36-stock universe, then proceed with historical backfill using existing `run_full_pipeline.py`. strength
- **Commodity Proxy**: GLD → `gold_return_1d`, CVX+XOM → `oil_return_1d`
- **Risk Regime**: AGG, IWM, EEM → risk-on/risk-off signals

### **Key Feature Categories**
1. **Sector Classification**: Each stock tagged with sector + binary flags
2. **Market Beta**: 20-day rolling correlation with SPY
3. **Sector Relative Performance**: Stock vs sector ETF performance
4. **QQQ/SPY Divergence**: Tech concentration vs broad market
5. **Commodity Exposure**: Gold/oil sensitivity via proxy assets
6. **Sector Rotation Signals**: Cross-sector momentum detection
7. **QQQ/SPY Options Signals**: Direct options predictions for ETF trading
8. **High IV Stock Focus**: TSLA, NVDA, AMD, COIN, MSTR for volatility plays

### **Data Coverage Validation**
✅ **Historical Data Range**: July 2, 2021 → July 2, 2025 (4 years complete history)
✅ **Complete Coverage (36 stocks)**: All have daily stocks + daily options + 30-min stocks + 30-min options
✅ **ETF Options Targets**: QQQ, SPY (complete coverage for heavy ETF trading)
✅ **High IV Stocks**: TSLA, NVDA, PLTR, COIN, HOOD, MSTR (full 4-year history)
✅ **All Sectors Covered**: Tech, Financial, Healthcare, Consumer, Energy, Industrial

---

## 9 | Performance & Architecture Recommendations

### **🚀 Performance Optimization**
1. **GPU Training Priority**: Focus on NVDA/AMD for GPU-accelerated model training
2. **High IV Weighting**: Weight TSLA, NVDA, COIN, MSTR higher in training (more volatile = more signal)
3. **Intraday Options Focus**: Use 30-minute options data for enhanced signal quality
4. **Memory Management**: 36 stocks × 200+ features = ~7.2K columns, use chunked processing
5. **Apple Metal GPU**: Leverage M2 Ultra GPU for LightGBM and neural network training

### **📊 Model Architecture Strategy**
1. **Multi-Target Models**: Train single model for stock + options predictions simultaneously
2. **Sector-Specific Models**: Separate models for TECH vs FINANCIAL vs HIGH_IV clusters
3. **Time-Series Models**: PatchTST for intraday options signals (QQQ/SPY focus)
4. **Ensemble Strategy**: Blend sector models + market-wide model for final predictions
5. **Model Hierarchy**: L0 (sector models) → L1 (blender) → L2 (risk overlay)

### **⚡ Trading Strategy Integration**
1. **QQQ/SPY Options Priority**: Primary focus for your heavy ETF trading (29K + 39K records)
2. **High IV Volatility Plays**: TSLA/NVDA/PLTR for volatility expansion trades (137K + 151K + 62K records)
3. **Sector Rotation Timing**: Use XLK/XLF relative strength for sector entry/exit
4. **Risk Management**: VIX + sector rotation signals for dynamic position sizing
5. **Options Chain Analysis**: Use full options data for gamma/delta hedging strategies

### **🔄 Data Pipeline Architecture**
1. **Feature Store**: Cache sector/market features (computed once, used by all 36 stocks)
2. **Incremental Training**: Daily model updates vs full retraining for faster iteration
3. **A/B Testing Framework**: Compare sector-aware vs market-wide models in production
4. **Data Lineage**: Track feature dependencies for debugging and validation
5. **Chunked Processing**: Process 36 stocks in batches to optimize memory usage

### **📈 Success Metrics & KPIs**
1. **Options Signal Accuracy**: QQQ/SPY options hit rate target >60%
2. **High IV Performance**: TSLA/NVDA/PLTR volatility prediction accuracy >55%
3. **Sector Rotation Timing**: XLK vs SPY relative strength signals
4. **Risk-Adjusted Returns**: Target Sharpe ratio >1.5 on paper trading
5. **Model Stability**: Feature importance consistency across retraining cycles

### **🎯 Implementation Priority**
1. **Phase 1**: Complete 36-stock training dataset with sector features
2. **Phase 2**: Deploy QQQ/SPY options models for immediate trading
3. **Phase 3**: High IV models for TSLA/NVDA volatility plays
4. **Phase 4**: Sector rotation models for regime detection
5. **Phase 5**: Full ensemble with risk management overlay

---

### **📊 30-Minute Bar Training Strategy**

**Training Data Structure (As Agreed):**
- **Daily Models**: Use daily bars for swing/position signals (1-5 day holds)
- **30-Minute Models**: Use 30-min bars for intraday entry/exit signals
- **Combined Features**: Daily + 30-minute features for comprehensive signals

**Implementation Strategy:**
1. **Phase 1**: Train models using daily + 30-minute data (current pipeline)
2. **Phase 2**: Focus on high IV stocks (TSLA, NVDA, PLTR) with enhanced 30-min features
3. **Phase 3**: Deploy for QQQ/SPY ETF trading with 30-minute precision

**Key Feature Categories**
1. **Sector Classification**: Each stock tagged with sector + binary flags
2. **Market Beta**: 20-day rolling correlation with SPY
3. **Sector Relative Performance**: Stock vs sector ETF performance
4. **QQQ/SPY Divergence**: Tech concentration vs broad market
5. **Commodity Exposure**: Gold/oil sensitivity via proxy assets
6. **Sector Rotation Signals**: Cross-sector momentum detection
7. **QQQ/SPY Options Signals**: Direct options predictions for ETF trading
8. **High IV Stock Focus**: TSLA, NVDA, AMD, COIN, MSTR for volatility plays



---

## 9 | Additional Recommendations

### **🚀 Performance Optimization**
1. **GPU Training**: Focus on NVDA/AMD for GPU-accelerated model training
2. **High IV Priority**: Weight TSLA, NVDA, PLTR, COIN, MSTR higher in training (more volatile = more signal)
3. **Intraday Focus**: Use 30-minute options data for same-day signals
4. **Memory Management**: 36 stocks × 200+ features = ~7.2K columns, use chunked processing

### **📊 Model Architecture**
1. **Multi-Target Models**: Train single model for stock + options predictions
2. **Sector-Specific Models**: Separate models for TECH vs FINANCIAL vs HIGH_IV
3. **Time-Series Models**: PatchTST for intraday options signals (QQQ/SPY focus)
4. **Ensemble Strategy**: Blend sector models + market-wide model

### **⚡ Trading Strategy Integration**
1. **QQQ/SPY Options**: Primary focus for your heavy ETF trading
2. **High IV Plays**: TSLA/NVDA/PLTR for volatility expansion trades
3. **Sector Rotation**: Use XLK/XLF relative strength for sector timing
4. **Risk Management**: Use VIX + sector rotation for position sizing

### **🔄 Data Pipeline Optimization**
1. **Real-Time**: Prioritize 30-minute data processing for same-day signals
2. **Feature Store**: Cache sector/market features (computed once, used by all stocks)
3. **Incremental Training**: Daily model updates vs full retraining
4. **A/B Testing**: Compare sector-aware vs market-wide models

### **📈 Success Metrics**
1. **Options Accuracy**: QQQ/SPY options signal hit rate >60%
2. **High IV Performance**: TSLA/NVDA/PLTR volatility prediction accuracy
3. **Sector Rotation**: XLK vs SPY timing signals
4. **Risk-Adjusted Returns**: Sharpe ratio >1.5 on paper tradingence**: Tech concentration vs broad market
5. **Commodity Exposure**: Gold/oil sensitivity via proxy assets
6. **Sector Rotation Signals**: Cross-sector momentum detection
7. **QQQ/SPY Options Signals**: Direct options predictions for ETF trading
8. **High IV Stock Focus**: TSLA, COIN, MSTR, NVDA for volatility plays

---

End of Roadmap
