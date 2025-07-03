# Feature Group Evaluation Report

**Generated**: 2025-07-03 05:05:09 UTC

**Symbols**: AAPL, MSFT, AMZN, GOOGL, TSLA

**Model**: LightGBM with 3-fold stratified cross-validation

## Cross-Validation Results by Feature Group

| Symbol | Lagged Returns Only | Cross-Asset Signals Only | Combined Features |
|--------|-------------------|--------------------------|------------------|
| AAPL | 0.5009  | 0.4956  | 0.5393  |
| MSFT | 0.4613  | 0.4717  | 0.4636  |
| AMZN | 0.4712  | 0.5239  | 0.4773  |
| GOOGL | 0.5064  | 0.4793  | 0.5084  |
| TSLA | 0.5105  | 0.5396  | 0.5544 ✅ |

## Analysis

### Lagged Returns Only
- **Successful Training**: 5/5 symbols
- **Symbols Meeting AUC ≥ 0.55**: 0

### Cross-Asset Signals Only
- **Successful Training**: 5/5 symbols
- **Symbols Meeting AUC ≥ 0.55**: 0

### Combined Features
- **Successful Training**: 5/5 symbols
- **Symbols Meeting AUC ≥ 0.55**: 1
- **High-Performing Symbols**: TSLA (0.5544)

## Integration Decision

✅ **RECOMMEND INTEGRATION**: 1 feature group(s) achieved AUC ≥ 0.55

**Combined Features**:
- TSLA: 0.5544

### Next Steps for Integration:
1. Merge successful feature groups into main feature engineering pipeline
2. Update feature metadata and training scripts
3. Relaunch HPO job for all 46 symbols with enhanced features
4. Monitor validation performance across full symbol universe

