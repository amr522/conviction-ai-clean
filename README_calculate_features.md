# Feature Calculation Module

The `src/calculate_features.py` module generates the final feature matrix from daily and intraday master datasets using high-performance Polars operations.

## Usage

### Basic Usage

```bash
# Calculate features for a single date
python src/calculate_features.py \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path datasets/features.parquet \
  --date 2025-01-16

# Calculate features for a date range
python src/calculate_features.py \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path datasets/features.parquet \
  --date 2025-01-01,2025-01-31 \
  --window-days 30
```

### Advanced Options

```bash
# With GPU acceleration and parallel processing
python src/calculate_features.py \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path datasets/features.parquet \
  --date 2025-01-16 \
  --window-days 20 \
  --use-gpu \
  --n-jobs 8
```

## CLI Arguments

- `--daily-master-path`: Path to daily master parquet file (required in full mode)
- `--intraday-master-path`: Path to intraday master parquet file (required in full mode)
- `--output-path`: Output path for final feature matrix (required in full mode)
- `--date`: Single date (YYYY-MM-DD) or date range (YYYY-MM-DD,YYYY-MM-DD) (required)
- `--window-days`: Rolling window size in days (default: 30)
- `--use-gpu`: Enable GPU acceleration for Polars operations
- `--n-jobs`: Number of parallel jobs for ticker processing (default: 1)

## Execution Modes

### Standalone Mode (Recommended for existing pipelines)

```bash
# Simple call with auto-detected master file paths
python src/calculate_features.py --date 2025-01-16 --use-gpu

# With custom window size
python src/calculate_features.py --date 2025-01-16 --window-days 20 --use-gpu
```

**Requirements**: Master datasets must exist at:
- `staged/daily_master.parquet`
- `datasets/intraday_master.parquet`

### Full Mode (For custom paths)

```bash
# Explicit paths for all arguments
python src/calculate_features.py \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path datasets/features.parquet \
  --date 2025-01-16 \
  --use-gpu
```

## Generated Features

### Rolling Macro Features
- `fred_rate_mean`: Rolling mean of federal funds rate
- `vix_std`: Rolling standard deviation of VIX index
- `news_count_rolling`: Rolling sum of news article counts

### Rolling Options Features
- `optd_iv30_mean`: Rolling mean of 30-day implied volatility
- `optd_volume_std`: Rolling standard deviation of options volume

### Rolling Stock Features
- `stockd_vol_rolling`: Rolling volatility of stock returns
- `stockd_volume_mean`: Rolling mean of stock volume

### Intraday Features
- `ret_1h`: 1-hour returns calculated from 30-minute intervals

### Cross-Sectional Features
- `vol_zscore`: Volume z-score across tickers per date
- `iv_rank`: IV rank across tickers per date
- `ret_relative`: Return relative to market average

## Pipeline Scripts

### Full Pipeline (Recommended)

```bash
# Runs complete pipeline: schema validation → master datasets → features → labels → training dataset
./single_day_pipeline.sh
```

### Standalone Pipeline

```bash
# Uses existing master datasets to generate features, labels, and training dataset
./single_day_pipeline_standalone.sh
```

### Manual Pipeline

```bash
# Uses explicit paths for all components
./single_day_pipeline_manual.sh
```

## Integration with Pipeline

The feature calculation module is integrated into the main training pipeline via `scripts/run_and_train.sh`:

```bash
# Run full pipeline with feature calculation
./scripts/run_and_train.sh 2025-01-16

# With custom parameters
DATE=2025-01-16 N_JOBS=8 ./scripts/run_and_train.sh
```

The pipeline automatically:
1. Runs the full data pipeline to generate master datasets
2. Calculates features using the new Polars-based module
3. Trains models using the engineered feature matrix
4. Monitors for data drift and sends Slack notifications

## Performance

- **Polars Backend**: High-performance columnar operations
- **Parallel Processing**: Multi-threaded ticker processing with `--n-jobs`
- **GPU Acceleration**: Optional GPU support for compatible hardware
- **Memory Efficient**: Streaming operations for large datasets

## Testing

Run the test suite to validate feature calculations:

```bash
# Run all feature calculation tests
pytest tests/test_calculate_features.py -v

# Run specific test
pytest tests/test_calculate_features.py::test_calculate_rolling_features -v
```

The tests validate:
- Date range parsing
- Rolling feature calculations
- Intraday feature generation
- Cross-sectional feature calculations
- Full integration pipeline
