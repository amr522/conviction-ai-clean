# Troubleshooting Guide

## Feature Correlation Failures

### Understanding Correlation Test Failures

The pipeline includes automated tests to detect highly correlated features that may indicate redundancy or data leakage.

**Error Message:**
```
AssertionError: Features feature1 and feature2 too highly correlated: 0.987
```

**Interpretation:**
- Features with correlation > 0.95 (absolute value) are flagged
- High correlation may indicate:
  - Duplicate features with different names
  - Features derived from the same underlying data
  - Potential data leakage
  - Multicollinearity issues for model training

### Resolution Steps

1. **Investigate Feature Definitions**
   ```bash
   # Check feature calculation logic
   grep -r "feature1\|feature2" src/calculate_features.py
   ```

2. **Analyze Feature Distributions**
   ```python
   import polars as pl
   df = pl.read_parquet("data/Parquet_data/features_test.parquet")
   print(df.select(["feature1", "feature2"]).describe())
   ```

3. **Remove Redundant Features**
   - Update `docs/features_list.md` to remove one of the correlated features
   - Modify feature calculation logic in `src/calculate_features.py`
   - Re-run validation tests

4. **Adjust Correlation Threshold**
   - If correlation is expected (e.g., related but distinct features)
   - Modify threshold in `tests/test_feature_correlations.py`
   - Document the decision in feature documentation

## Cleanup Behavior

### Automatic Cleanup on Failures

All validation scripts include automatic cleanup on failure using bash traps:

```bash
trap 'echo "💥 Failure detected, cleaning up..."; rm -f temp_files; exit 1' ERR
```

**Cleaned Files:**
- `data/Parquet_data/features_test.parquet`
- `data/Parquet_data/features_validation.parquet`
- `data/Parquet_data/daily_master.parquet`
- `data/Parquet_data/intraday_master.parquet`
- `staged/` directory contents

### Manual Cleanup

If cleanup fails or you need to clean manually:

```bash
# Remove test artifacts
rm -f data/Parquet_data/features_*.parquet
rm -f data/Parquet_data/*_master.parquet
rm -rf staged/

# Reset to clean state
git clean -fd data/Parquet_data/
```

## Common Issues

### Low Variance Features

**Error:**
```
AssertionError: Features with low variance: ['feature_name (var=1.23e-07)']
```

**Resolution:**
- Check if feature is constant across all samples
- Verify feature calculation logic
- Consider removing constant features from feature list

### Missing Features in Correlation Test

**Error:**
```
SKIPPED: Features feature1 or feature2 not found in dataframe
```

**Resolution:**
- Ensure feature calculation generates all expected features
- Check `docs/features_list.md` for typos
- Verify feature names match between calculation and validation

### Null Correlation Values

**Error:**
```
SKIPPED: Cannot compute correlation between feature1 and feature2 (constant features)
```

**Resolution:**
- One or both features have zero variance
- Check for constant values or missing data
- Review feature engineering logic

## Performance Issues

### Slow Correlation Tests

For large feature sets, correlation tests may be slow due to pairwise combinations:

**Optimization Options:**
1. **Reduce Feature Set**: Remove unnecessary features from `docs/features_list.md`
2. **Parallel Testing**: Use pytest-xdist for parallel execution
3. **Sampling**: Test correlations on data subset for CI

### Memory Issues

**Symptoms:**
- Out of memory errors during correlation calculation
- Slow test execution

**Solutions:**
```bash
# Use smaller test dataset
export TEST_SAMPLE_SIZE=1000

# Increase available memory
export POLARS_MAX_THREADS=4
```

## Debugging Tips

### Enable Verbose Output

```bash
# Run correlation tests with verbose output
pytest tests/test_feature_correlations.py -v -s

# Check specific feature pair
pytest tests/test_feature_correlations.py -k "feature1 and feature2" -v
```

### Inspect Feature Data

```python
import polars as pl

# Load and inspect features
df = pl.read_parquet("data/Parquet_data/features_test.parquet")

# Check feature statistics
print(df.describe())

# Check for null values
print(df.null_count())

# Compute correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
```

### Test Individual Features

```python
# Test specific feature pair
from tests.test_feature_correlations import test_low_correlation
import polars as pl

df = pl.read_parquet("data/Parquet_data/features_test.parquet")
test_low_correlation(df, "feature1", "feature2")
```