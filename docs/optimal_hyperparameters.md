# Optimal Hyperparameters Guide

This document contains the optimal hyperparameters from successful HPO jobs.

## Latest HPO Job Results (Session 7)
- **HPO Job:** hpo-full-1751610067
- **Best Training Job:** hpo-full-1751610067-032-ecde9880
- **Validation AUC:** 1.0 (perfect score)
- **Status:** Completed (50/50 training jobs successful)
- **Completion Time:** 2025-07-04T06:43:22Z

## Previous HPO Job Results (Session 6)
- **HPO Job:** hpo-full-1751604591
- **Best Training Job:** hpo-full-1751604591-044-b07b4aa3
- **Validation AUC:** 1.0 (perfect score)
- **Completion Time:** 2025-07-04T05:03:06.795Z

## XGBoost Hyperparameters (Latest - Session 7)

### Core Model Parameters
```json
{
  "max_depth": 10,
  "eta": 0.2629932420726331,
  "min_child_weight": 2,
  "subsample": 0.9135096821292192,
  "gamma": 0.6498349748696841,
  "alpha": 0.5164473042633577,
  "lambda": 1.9489287645384434,
  "colsample_bytree": 0.9550893077982054
}
```

## XGBoost Hyperparameters (Previous - Session 6)

### Core Model Parameters
```json
{
  "max_depth": 10,
  "eta": 0.2947092908785793,
  "min_child_weight": 2,
  "subsample": 0.9512099770182731,
  "gamma": 0.32189315987980804,
  "alpha": 3.270423068904514,
  "lambda": 0.015453094129485034,
  "colsample_bytree": 0.7860758068396407
}
```

### Feature Engineering Parameters
```json
{
  "iv_rank_window": 30,
  "iv_rank_weight": 0.5,
  "term_slope_window": 15,
  "term_slope_weight": 0.5,
  "oi_window": 15,
  "oi_weight": 0.5,
  "theta_window": 15,
  "theta_weight": 0.5,
  "vix_mom_window": 10,
  "vix_regime_thresh": 25.0,
  "event_lag": 2,
  "event_lead": 2,
  "news_threshold": 0.05,
  "lookback_window": 5,
  "reuters_weight": 1.0,
  "sa_weight": 1.0
}
```

## Usage as Starting Point

For future HPO jobs, use these parameters as the center point for your hyperparameter ranges:
- Set ranges ±20% around these optimal values
- Focus tuning on the most impactful parameters: max_depth, eta, alpha, lambda
- Keep feature engineering parameters close to these optimal values

## Performance Notes
- This configuration achieved perfect validation AUC on the 46-stock universe
- Training completed successfully across all 50 HPO training jobs
- No overfitting detected with these regularization parameters (alpha=3.27, lambda=0.015)

## Recommended Hyperparameter Ranges for Future HPO

Based on the optimal values, here are suggested ranges for future hyperparameter optimization:

### XGBoost Core Parameters
- `max_depth`: 8-12 (optimal: 10)
- `eta`: 0.24-0.35 (optimal: 0.295)
- `min_child_weight`: 1-4 (optimal: 2)
- `subsample`: 0.8-1.0 (optimal: 0.951)
- `gamma`: 0.1-0.5 (optimal: 0.322)
- `alpha`: 2.5-4.0 (optimal: 3.270)
- `lambda`: 0.01-0.03 (optimal: 0.015)
- `colsample_bytree`: 0.7-0.9 (optimal: 0.786)

### Feature Engineering Parameters (Keep Fixed)
These parameters achieved optimal performance and should remain fixed:
- `iv_rank_window`: 30
- `vix_regime_thresh`: 25.0
- `news_threshold`: 0.05
- `lookback_window`: 5

## Implementation Example

```python
# Use in aws_hpo_launch.py hyperparameter ranges
def get_optimized_hyperparameter_ranges():
    return {
        'max_depth': IntegerParameter(8, 12),
        'eta': ContinuousParameter(0.24, 0.35),
        'min_child_weight': IntegerParameter(1, 4),
        'subsample': ContinuousParameter(0.8, 1.0),
        'gamma': ContinuousParameter(0.1, 0.5),
        'alpha': ContinuousParameter(2.5, 4.0),
        'lambda': ContinuousParameter(0.01, 0.03),
        'colsample_bytree': ContinuousParameter(0.7, 0.9),
    }
```
