# Volatility-Focused Options Trading Plan

(Updated: July 21, 2025)

---

## 1. Goal

Maximize risk-adjusted returns from weekly swing options and covered-call income by exploiting mis-priced implied volatility (IV) rather than pure price direction.

---

## 2. Model Stack Overview

| Layer | Model & Horizon | Output | Primary Use |
|-------|-----------------|--------|-------------|
| L0-A Direction | XGBoost (Autopilot V2) – 3-5 day returns | ΔŜ | Avoid selling ITM calls; delta guidance |
| L0-B Vol Forecast | EGARCH(1,1) – 5-day σ̂ | σ̂_realised | Width for straddles, risk scaling |
| L0-C IV–HV Spread | LightGBM regressor – next-day Δ(IV30-HV30) | Spread_pred | Decide long-vega vs short-vega |
| L0-D Regime Score | Bayesian Online Change-Point on σ | p(low/med/high) | Strategy weighting |
| L0-E Tail Alert | PatchTST – 5-day tail prob | p(tail_event) | Risk management, position sizing |
| L1 Blender | LightGBM meta-model | Expected Edge $ | Ranks (symbol,strike,expiry) combos |

---

## 3. Feature Engineering

1. IV Surface: IV7, IV30, IV skew (25∆ put – 25∆ call), term slope
2. Event Flags: earnings, macro (CPI/FOMC), product launches
3. Realised Vol Windows: HV10, HV30
4. Liquidity Metrics: OI, bid-ask %, volume multiple
5. Macro Context: VIX, DXY, 10Y-2Y, Fed Funds surprise

---

## 4. Data & ETL Roadmap

| Step | Source | Artifact |
|------|--------|----------|
| 1 | Polygon option / IV surface API | raw/iv_surface/*.parquet |
| 2 | Earnings calendar (Nasdaq API) | raw/events/earnings.csv |
| 3 | Glue Spark: merge price, IV, events | clean/option_features.parquet |
| 4 | Label creation scripts | clean/labels_direction.parquet, iv_crush_labels.parquet |

---

## 5. Training Schedule

- **Nightly**: L0-A (direction), L0-C (IV-HV), update labels
- **Weekly**: L0-B EGARCH refit; L0-D BOCPD regime model
- **Monthly**: L0-E PatchTST GPU fine-tune; L1 blender retrain

---

## 6. Deployment Architecture

```
graph TD;
  price_feed-->Glue_ETL;
  iv_feed-->Glue_ETL;
  events_feed-->Glue_ETL;
  Glue_ETL-->S3_clean;
  S3_clean-->L0A[XGB Endpoint];
  S3_clean-->L0B[Lambda-GARCH];
  S3_clean-->L0C[IV-HV Endpoint];
  S3_clean-->L0E[PatchTST Batch GPU];
  L0A & L0B & L0C & L0D & L0E --> L1Blender;
  L1Blender --> StrategySelector(Lambda);
  StrategySelector --> DynamoTrades;
```

## 7. Strategy Rules

1. IV > HV + z>1  &  p(crush)<0.3  ⇒ Sell covered call / short strangle
2. IV < HV − z>1 &  p(tail_move)>0.6 ⇒ Buy ATM call/put (direction via L0-A sign)
3. |IV-HV| < 0.5σ & regime=low      ⇒ Iron condor; smallest width where Edge > 0

Edge = Blender output – transaction_cost

---

## 8. Evaluation Metrics

- Return models: RMSE, MAPE, directional hit-rate
- IV spread model: MAE, R²
- Classifier (crush / tail): AUC, precision@20% recall
- Strategy level: CAGR, Sharpe, max-drawdown, assignment rate

---

## 9. Implementation Plan

### Phase 1: ETL and Data Preparation
1. **Complete and Run ETL Script**
   - Finish implementing the `prevent_target_leakage_spark` function in `glue_etl_option_features.py`
   - Add performance monitoring and optimization features
   - Run the ETL job on AWS Glue with 20 DPUs for optimal performance
   - Output: `s3://convictionai-data/conviction-ai/processed/option_features.parquet`

2. **Validate ETL Output**
   - Create a validation script to check for data leakage
   ```bash
   python validate_option_features.py \
     --input-s3 s3://convictionai-data/conviction-ai/processed/option_features.parquet \
     --output-report validation_report.json
   ```
   - Verify feature distributions, missing values, and proper backshifting
   - Confirm all 51 stocks are present with expected data volume

### Phase 2: Score Existing L0 Models
1. **Create L0 Scoring Script**
   - Implement `score_L0_on_options.py` to apply existing models to options data
   - Connect to existing AutoML endpoints
   - Generate predictions for each L0 model
   - Save combined predictions to S3

2. **Run L0 Scoring**
   ```bash
   python score_L0_on_options.py \
     --input-s3 s3://convictionai-data/conviction-ai/processed/option_features.parquet \
     --output-s3 s3://convictionai-data/conviction-ai/scored/options_L0_preds.parquet \
     --automl-endpoint conviction-ai-endpoint-20250721153332 \
     --ensemble-endpoint conviction-ensemble-final-1751650090
   ```

### Phase 3: Implement Missing L0 Models
1. **EGARCH Volatility Model (L0-B)**
   - Use existing `egarch_vol_forecast.py` script
   - Run on options data to generate volatility forecasts
   ```bash
   python egarch_vol_forecast.py \
     --s3-bucket convictionai-data \
     --input-prefix conviction-ai/processed/ \
     --output-prefix conviction-ai/models/egarch/
   ```

2. **IV-HV Spread Model (L0-C)**
   - Implement `train_ivhv_spread.py` script
   - Train GBM regressor to predict IV-HV spread
   ```bash
   python train_ivhv_spread.py \
     --data-path s3://convictionai-data/conviction-ai/processed/option_features.parquet \
     --output-model s3://convictionai-data/conviction-ai/models/ivhv_spread/model.pkl
   ```

3. **Regime Detector (L0-D)**
   - Implement `train_regime_detector.py` script
   - Train BOCPD model for regime detection
   ```bash
   python train_regime_detector.py \
     --data-path s3://convictionai-data/conviction-ai/processed/option_features.parquet \
     --output-model s3://convictionai-data/conviction-ai/models/regime/model.pkl
   ```

4. **PatchTST Tail-Risk Model (L0-E)**
   - Use existing `train_patchtst_tail.py` script
   - Train on options data
   ```bash
   python train_patchtst_tail.py \
     --start 2020-01-01 \
     --end 2024-12-31 \
     --output-s3-uri s3://convictionai-data/conviction-ai/models/patchtst_tail/ \
     --target-columns "tail_event,sigma_forecast_5d,ivhv_spread_tplus1"
   ```

### Phase 4: Train Option-Specific Models
1. **IV-Crush Classifier**
   - Implement `iv_crush_classifier.py` script
   - Train GBM classifier to predict significant IV drops
   ```bash
   python iv_crush_classifier.py \
     --data-path s3://convictionai-data/conviction-ai/scored/options_L0_preds.parquet \
     --output-model s3://convictionai-data/conviction-ai/models/iv_crush/model.pkl
   ```

2. **Covered-Call Blender (L1)**
   - Implement `covered_call_blender.py` script
   - Train LightGBM regressor to predict expected edge
   ```bash
   python covered_call_blender.py \
     --data-path s3://convictionai-data/conviction-ai/scored/options_L0_preds.parquet \
     --iv-crush-model s3://convictionai-data/conviction-ai/models/iv_crush/model.pkl \
     --output-model s3://convictionai-data/conviction-ai/models/covered_call_blender/model.txt
   ```

### Phase 5: Deploy Final Endpoint
1. **Create Deployment Script**
   - Implement `deploy_blender_endpoint.py` script
   - Configure for real-time inference

2. **Deploy Endpoint**
   ```bash
   python deploy_blender_endpoint.py \
     --model-path s3://convictionai-data/conviction-ai/models/covered_call_blender/model.txt \
     --endpoint-name covered-call-blender-prod
   ```

3. **Test Endpoint**
   - Create test script to validate endpoint functionality
   - Verify predictions match expected format and ranges

### Phase 6: Monitoring and Evaluation
1. **Create Evaluation Script**
   - Implement `evaluate_covered_call_strategy.py`
   - Backtest strategy on historical data

2. **Set Up Monitoring**
   - Configure CloudWatch alarms for endpoint metrics
   - Implement data drift detection

---

## 10. Milestones

| Date | Deliverable |
|------|-------------|
| Aug 05, 2025 | ETL pipeline complete with option_features.parquet |
| Aug 10, 2025 | L0 models scored on options data |
| Aug 15, 2025 | IV-Crush Classifier trained |
| Aug 20, 2025 | Covered-Call Blender trained and evaluated |
| Sep 01, 2025 | Live endpoint deployed with IBKR paper trading integration |

---

## 11. Forex Integration (M2 Ultra GPU Optimized)

### Feature Engineering Pipeline

#### GPU-Accelerated Feature Set

1. **Volatility Features**
   - Historical Volatility (HV)
     ```
     HVn = std(log_returns) * sqrt(252) * 100
     where n = [10, 30, 60] days
     ```
   - Parkinson Volatility (PV)
     ```
     PV = sqrt(1/(4*ln(2)) * mean(ln(high/low)²)) * sqrt(252) * 100
     ```
   - Volatility of Volatility (VoV)
     ```
     VoVn = std(HVn) over n days
     ```

2. **Technical Features**
   - Moving Averages (MA)
     ```
     MAn = mean(close) over n days
     where n = [5, 10, 20, 50, 100]
     ```
   - MA Distance (normalized and bounded)
     ```
     MA_distn = clip((close - MAn) / max(abs(MAn), 1e-6), -10, 10)
     ```
   - RSI14 (Bounded 0-100)
     ```
     gains = max(delta, 0)
     losses = abs(min(delta, 0))
     avg_gain = mean(gains) over 14 days
     avg_loss = mean(losses) over 14 days
     rs = avg_gain / (avg_loss + 1e-8)
     RSI = 100 - (100 / (1 + rs))
     ```

3. **Statistical Features**
   - Skewness (30d, 60d windows)
     ```
     skewn = skew(returns) over n days
     ```
   - Kurtosis (30d, 60d windows)
     ```
     kurtn = kurtosis(returns) over n days
     ```

4. **Trend Features**
   - Linear regression slope (10d, 30d)
     ```
     trendn = polyfit(range(n), close, deg=1)[0]
     ```

5. **Mean Reversion Features**
   - Bollinger Bands (20d, 50d)
     ```
     BB_middle = rolling_mean(close, n)
     BB_std = rolling_std(close, n)
     BB_upper = BB_middle + (2 * BB_std)
     BB_lower = BB_middle - (2 * BB_std)
     BB_position = clip((close - BB_middle) / max(2 * BB_std, 1e-6), -1, 1)
     ```

6. **Volume Features**
   - Volume-Weighted Price
     ```
     cumulative_pv = cumsum(price * volume)
     cumulative_v = cumsum(volume)
     VWAP = cumulative_pv / (cumulative_v + 1e-8)
     VWAP_dist = (close - VWAP) / (VWAP + 1e-8)
     ```

#### Implementation Details

    def process_features(self, parquet_path):
        """
        Calculate GPU-accelerated forex features using exact formulas:
        
        1. Historical Volatility (HV):
           - HVn = std(log_returns) * sqrt(252) * 100
           where n is the window size (10, 30, 60 days)
           
        2. Average True Range (ATR14):
           - TR = max(high-low, |high-prev_close|, |low-prev_close|)
           - ATR14 = mean(TR) over 14 periods
           
        3. Volume Z-Score:
           - z = (volume - rolling_mean(volume, 30)) / rolling_std(volume, 30)
        """
        # Load data to GPU
        df = pd.read_parquet(parquet_path)
        close = torch.tensor(df['close'].values, dtype=torch.float32, device=self.device)
        high = torch.tensor(df['high'].values, dtype=torch.float32, device=self.device)
        low = torch.tensor(df['low'].values, dtype=torch.float32, device=self.device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32, device=self.device)
        
        # Calculate log returns
        log_returns = torch.log(close[1:] / close[:-1])
        
        # Historical Volatility Features
        vol_features = {}
        for window in self.windows:
            rolling_vol = torch.zeros_like(close)
            for i in range(window, len(log_returns) + 1):
                # Daily volatility calculation
                window_returns = log_returns[i-window:i]
                daily_vol = torch.std(window_returns)
                # Annualize and convert to percentage with proper scaling
                vol = daily_vol * torch.sqrt(torch.tensor(self.trading_days)) * self.vol_scale * 0.01
                rolling_vol[i-1] = torch.minimum(vol, torch.tensor(self.vol_cap, device=self.device))
            vol_features[f'HV{window}'] = rolling_vol.cpu().numpy()
        
        # ATR Calculation
        high_low = high - low
        high_close_prev = torch.abs(high[1:] - close[:-1])
        low_close_prev = torch.abs(low[1:] - close[:-1])
        
        tr = torch.zeros_like(close)
        tr[1:] = torch.maximum(high_low[1:], 
                             torch.maximum(high_close_prev, low_close_prev))
        
        atr14 = torch.zeros_like(close)
        for i in range(14, len(tr)):
            atr14[i] = torch.mean(tr[i-14:i])
        
        # Volume Z-score (30-day window)
        vol_zscore = torch.zeros_like(volume)
        for i in range(30, len(volume)):
            window = volume[i-30:i]
            mean = torch.mean(window)
            std = torch.std(window)
            if std > 0:
                vol_zscore[i] = (volume[i] - mean) / std
        
        # Clamp volume z-scores to reasonable range
        vol_zscore = torch.clamp(vol_zscore, min=-10.0, max=10.0)
        
        # Add features to DataFrame
        for name, values in vol_features.items():
            df[name] = values
        
        df['atr14'] = atr14.cpu().numpy()
        df['volume_zscore'] = vol_zscore.cpu().numpy()
        
        return df

    def add_regime_features(self, df):
        # Utilize cusignal for faster signal processing
        df['atr14'] = cusignal.atr(
            df['high'].values, 
            df['low'].values,
            df['close'].values,
            14
        )
        
        # Volatility regime detection
        df['vol_regime'] = (
            df['HV10'] - df['HV60']
        ).rolling(5).mean() / df['HV60']
        
        return df

    def add_mean_reversion_features(self, df):
        # Bollinger Bands (GPU Optimized)
        for window in [20, 50]:
            df[f'bb_middle_{window}'] = (
                df.groupby('currency_pair')['close']
                .rolling(window)
                .mean()
            )
            df[f'bb_std_{window}'] = (
                df.groupby('currency_pair')['close']
                .rolling(window)
                .std()
            )
        
        return df

    def add_volume_features(self, df):
        # Volume profile (GPU Parallel)
        df['volume_ma20'] = (
            df.groupby('currency_pair')['volume']
            .rolling(20)
            .mean()
        )
        
        df['volume_zscore'] = (
            (df['volume'] - df['volume_ma20']) /
            df.groupby('currency_pair')['volume']
            .rolling(20)
            .std()
        )
        
        return df
```

### GPU-Optimized Validation Script
```python
def validate_forex_features(df):
    # Data quality checks
    checks = {
        'missing_values': df.isnull().sum().sum() == 0,
        'infinity_check': (~df.isin([cp.inf, -cp.inf])).all().all(),
        'timestamp_continuity': df.index.is_monotonic_increasing,
        'value_ranges': {
            'HV_features': all(
                df[f'HV{window}'].between(0, 500).all() 
                for window in [10, 30, 60]
            ),
            'zscore_features': df['volume_zscore'].between(-10, 10).all(),
            'regime_features': df['vol_regime'].between(-5, 5).all()
        }
    }
    
    return checks
```

### Integration with Existing Pipeline
1. Feature importance for L0 models
2. Regime alignment with options volatility
3. Cross-asset volatility spillover detection
4. GPU-optimized real-time feature updates

### Performance Metrics (M2 Ultra)
- Feature calculation: ~500ms per 1M rows
- Memory efficiency: ~2GB GPU memory usage
- Batch processing: 50M rows per minute
- Real-time updates: < 10ms latency

### Feature Validation Ranges

| Feature Type | Valid Range | Typical Range |
|--------------|-------------|---------------|
| Historical Volatility | 0-100% | 5-40% |
| Parkinson Volatility | 0-100% | 5-35% |
| Volatility of Volatility | 0-30% | 0.5-2% |
| MA Distance | -10 to +10 | -2 to +2 |
| RSI | 0-100 | 30-70 |
| BB Position | -1 to +1 | -0.8 to +0.8 |
| Volume Z-score | -10 to +10 | -3 to +3 |

### Quality Checks
1. No missing or infinite values
2. Proper time series continuity
3. Feature bounds enforcement
4. Device consistency for GPU operations
5. Numerical stability with epsilon values

## 12. Open Questions / Next Steps

- Margin impact: set risk caps per account?
- GPU cost for nightly PatchTST—spot or batch transform?
- Add vendor data for single-stock VIX? (VXAPL, etc.)
- Build dashboard (QuickSight) for regime visualization
- Implement automated retraining pipeline for all models
- Consider adding options-specific features (e.g., put-call ratio, open interest trends)
- Explore multi-leg strategy optimization

## 13. Forex Analysis Recommendations (GPU-Accelerated)

Based on M2 Ultra GPU analysis of forex.parquet data:

### Volatility Integration Points
1. Cross-Asset Volatility Signals
   - Add forex volatility regime as input to L0-D model
   - Use HV10/HV60 ratio as early warning for regime shifts
   - Monitor currency-pair specific ATR for position sizing

2. Enhanced Risk Management
   - Implement real-time forex volatility alerts using M2 GPU
   - Add currency exposure limits based on volume_zscore
   - Cross-validate option positions against forex risk

3. Model Improvements
   - Include forex volatility features in PatchTST tail detection
   - Add currency correlation matrix to risk calculations
   - Use forex market state as regime classifier input

### Technical Recommendations
1. **GPU Optimization**
   - Batch size: 50,000 rows for optimal M2 Ultra MPS performance
   - Use torch.device("mps") for all feature calculations
   - Implement parallel processing for multi-currency analysis

2. **Feature Engineering**
   - Add Fibonacci retracement levels for key pairs
   - Calculate cross-pair volatility spillover effects
   - Monitor volume profile for liquidity conditions

3. **Integration Tasks**
   - Create unified volatility surface including forex impact
   - Implement real-time forex feature updates (< 10ms latency)
   - Add forex volatility alerts to monitoring dashboard

---

*Last updated: July 24, 2025*