# Bronze ETL Framework

A clean, modular framework for building bronze-layer ETL processes based on the specifications in `parquet_issues.md`.

## Overview

This framework provides a structured approach to processing raw financial data into bronze-layer Parquet files with:

- **Unified Universe**: 30 underlyings (28 equities + SPY & QQQ)
- **Consistent Date Range**: 2023-07-07 to 2024-07-07
- **GPU Acceleration**: Apple MPS or CUDA/cuDF support
- **Market Hours Filtering**: 09:30–16:00 ET, exclude weekends/holidays
- **Schema Validation**: Great Expectations integration
- **Options Parsing**: Extract ticker, strike, expiry, type, bid/ask, IV
- **30-Minute Aggregation**: Group 1-min OHLCV into 30-min bars
- **End-to-End Validation**: Quality checks at each stage

## Architecture

```
bronze_etl/
├── config.py          # Configuration constants and helpers
├── validate_schema.py # Schema validation and data quality checks
├── parse_options.py   # Options ticker parsing and validation
├── aggregate_30min.py # 30-minute OHLCV aggregation
├── loader.py          # Parquet I/O with GPU acceleration
├── utils.py           # Common utilities and filters
├── etl_main.py        # Main orchestration pipeline
├── test_framework.py  # Framework testing
└── __init__.py        # Package exports
```

## Quick Start

### 1. Basic Usage

```python
from src.bronze_etl import run_etl_pipeline

# Process stocks daily data
results = run_etl_pipeline(
    data_type="stocks_daily",
    step="all"
)

# Process options 30-minute data with sample
results = run_etl_pipeline(
    data_type="options_30min",
    step="all",
    sample_size=10000
)
```

### 2. CLI Usage

```bash
# Process stocks daily data
python -m src.bronze_etl.etl_main --data-type stocks_daily

# Process options 30-minute data with sample
python -m src.bronze_etl.etl_main --data-type options_30min --sample-size 10000

# Run only validation step
python -m src.bronze_etl.etl_main --data-type stocks_30min --step validate

# Validate configuration
python -m src.bronze_etl.etl_main --validate-config
```

### 3. Individual Module Usage

```python
from src.bronze_etl import (
    setup_gpu_acceleration,
    load_parquet_with_timing,
    validate_schema_complete,
    parse_options_complete,
    aggregate_30min_complete,
    apply_common_filters
)

# Setup GPU acceleration
setup_gpu_acceleration()

# Load and validate data
df, timing = load_parquet_with_timing("data/raw.parquet")
validation = validate_schema_complete(df, "stocks_30min")

# Apply common filters
df = apply_common_filters(df, "stocks_30min")

# Parse options (if applicable)
if "options" in data_type:
    df, parsing_validation = parse_options_complete(df)

# Aggregate to 30-minute bars (if applicable)
if "30min" in data_type:
    df, aggregation_validation = aggregate_30min_complete(df, data_type)
```

## Configuration

### Universe Definition

The framework uses a unified universe of 30 underlyings:

```python
from src.bronze_etl import UNIVERSE, ETF_TICKERS

print(f"Universe: {len(UNIVERSE)} tickers")
print(f"ETFs: {ETF_TICKERS}")
```

### Date Range

Consistent training period across all datasets:

```python
from src.bronze_etl import DATE_START, DATE_END

print(f"Date range: {DATE_START} to {DATE_END}")
```

### GPU Configuration

Optimized for Apple M2 Ultra and CUDA:

```python
from src.bronze_etl import GPU_CONFIG, BATCH_CONFIG

print(f"GPU config: {GPU_CONFIG}")
print(f"Batch config: {BATCH_CONFIG}")
```

## Data Types

The framework supports four main data types:

### 1. stocks_daily

- **Source**: `data/Parquet_data/Raw/Stocks_daily`
- **Output**: `staged/bronze_stocks_daily.parquet`
- **Processing**: Filter to universe, apply market calendar

### 2. stocks_30min

- **Source**: `data/Parquet_data/Raw/stocks_minute` (1-min data)
- **Output**: `staged/bronze_stocks_30min.parquet`
- **Processing**: Filter to universe, aggregate to 30-min bars

### 3. options_daily

- **Source**: `data/Parquet_data/Raw/options_daily`
- **Output**: `staged/bronze_options_daily.parquet`
- **Processing**: Parse options, filter by underlying universe

### 4. options_30min

- **Source**: `data/Parquet_data/Raw/option_minute` (1-min data)
- **Output**: `staged/bronze_options_30min.parquet`
- **Processing**: Parse options, aggregate to 30-min bars

## Processing Steps

### 1. Schema Validation

- Load Great Expectations schemas
- Validate required columns and field types
- Check timestamp sanity and market hours alignment

### 2. Universe & Date Filtering

- Filter to 30-ticker universe
- Apply date range (2023-07-07 to 2024-07-07)
- Filter to market hours (09:30–16:00 ET)
- Exclude weekends and holidays

### 3. Options Parsing

- Extract underlying ticker from option symbol
- Parse expiration date, strike price, option type
- Validate parsing success rates
- Filter to universe underlyings

### 4. 30-Minute Aggregation

- Group 1-minute OHLCV into 30-minute windows
- Aggregate: Open=first, High=max, Low=min, Close=last, Volume=sum
- Filter to :00 and :30 timestamps only
- Validate expected bar counts (13 per trading day)

### 5. Quality Validation

- Check for missing values and data quality
- Validate bar alignment and completeness
- Generate summary statistics
- Log performance metrics

## Validation

### Schema Validation

```python
from src.bronze_etl import validate_schema_complete

validation = validate_schema_complete(df, "stocks_30min")
print(f"Schema valid: {validation['schema_valid']}")
print(f"Quality score: {validation['quality_metrics']['quality_score']}")
```

### Options Parsing Validation

```python
from src.bronze_etl import parse_options_complete

df, validation = parse_options_complete(df)
print(f"Parse success rate: {validation['parsing']['parse_success_rate']}")
print(f"Universe coverage: {validation['parsing']['universe_coverage']}")
```

### Bar Alignment Validation

```python
from src.bronze_etl import aggregate_30min_complete

df, validation = aggregate_30min_complete(df, "stocks_30min")
print(f"Alignment score: {validation['alignment']['alignment_score']}")
print(f"Expected vs actual: {validation['expected_bars']} vs {validation['final_records']}")
```

## Performance

### GPU Acceleration

The framework automatically configures GPU acceleration:

- **Apple Silicon**: MPS (Metal Performance Shaders)
- **NVIDIA**: CUDA/cuDF with RAPIDS
- **Memory Management**: Configurable memory fractions
- **Batch Processing**: Optimized chunk sizes

### Benchmarking

```python
from src.bronze_etl import benchmark_loading_performance

benchmark = benchmark_loading_performance("data/raw/", [1000, 10000, 100000])
for size, metrics in benchmark.items():
    print(f"Sample {size}: {metrics['records_per_second']:,.0f} records/s")
```

## Testing

### Run Framework Tests

```bash
python -m src.bronze_etl.test_framework
```

### Test Individual Modules

```python
from src.bronze_etl.test_framework import (
    test_config,
    test_schema_validation,
    test_options_parsing,
    test_30min_aggregation,
    test_utils,
    test_loader
)

# Test specific modules
test_config()
test_options_parsing()
test_30min_aggregation()
```

## Error Handling

The framework provides comprehensive error handling:

```python
from src.bronze_etl import BronzeETLError, SchemaValidationError, OptionsParsingError

try:
    results = run_etl_pipeline("stocks_daily")
except BronzeETLError as e:
    print(f"ETL failed: {e}")
except SchemaValidationError as e:
    print(f"Schema validation failed: {e}")
except OptionsParsingError as e:
    print(f"Options parsing failed: {e}")
```

## Logging

Configure logging levels and outputs:

```python
from src.bronze_etl import setup_logging

# Setup logging
setup_logging(
    log_level="INFO",
    log_file="bronze_etl.log"
)
```

## Migration from Old Scripts

The framework is designed to replace the existing bronze ETL scripts:

### Old Scripts

- `build_bronze_stocks_daily.py`
- `build_bronze_stocks_30min.py`
- `build_bronze_options_daily.py`
- `build_bronze_options_30min.py`

### New Framework

```python
# Replace individual scripts with unified framework
for data_type in ["stocks_daily", "stocks_30min", "options_daily", "options_30min"]:
    results = run_etl_pipeline(data_type)
```

## Requirements

- Python 3.8+
- Polars (with GPU support)
- PyArrow
- Pandas
- PyTZ
- FastJSONSchema (for schema validation)

## Performance Targets

- **GPU Speedup**: 2× faster than CPU-only processing
- **Memory Efficiency**: Handle large datasets without OOM
- **Validation**: <1% deviation from expected bar counts
- **Quality**: >95% data quality score

## Contributing

1. Follow the modular architecture
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Maintain backward compatibility
5. Follow the existing code style and patterns

## License

Internal use only - Conviction AI Team
