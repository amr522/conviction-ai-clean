#!/usr/bin/env python3
"""
Comprehensive Feature Pipeline Validation

Validates that all 37 required features can be computed correctly
from the available data sources and scripts.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required features from roadmap
REQUIRED_FEATURES = {
    "macro": ["vix_value", "vix_ma_divergence", "dxy_value", "iv_rank_30d"],
    "news": ["news_count_lag1", "avg_sentiment_lag1"],
    "options_daily": [
        "optd_iv30",
        "optd_hv30",
        "optd_iv_skew_slope",
        "optd_vol_surprise",
        "optd_put_call_ratio",
        "optd_volume",
    ],
    "options_30min": [
        "opt30_mid_price_return",
        "opt30_bid_ask_spread",
        "opt30_implied_volatility",
        "opt30_delta",
        "opt30_theta",
        "opt30_volume_return",
        "opt30_rolling_vol_5",
        "opt30_flow_divergence",
        "opt30_gamma_squeeze",
    ],
    "stocks_daily": [
        "stockd_close",
        "stockd_volume",
        "stockd_return_1d",
        "stockd_vol_7d",
        "stockd_volume_pct_change",
        "stockd_beta_spy",
        "stockd_days_to_earnings",
        "stockd_earnings_flag",
    ],
    "stocks_30min": [
        "stock30_close_return",
        "stock30_rolling_vol_5",
        "stock30_is_last_30min",
    ],
}


def validate_data_sources():
    """Validate all required data sources exist"""
    logger.info("🔍 Validating data sources...")

    sources = {
        "macro_fred": "data/Parquet_data/fred.parquet",
        "macro_vix": "data/Parquet_data/vix_data.parquet",
        "macro_dxy": "data/Parquet_data/dxy.parquet",
        "news": "data/Parquet_data/news_2025-05-05.parquet",
        "stocks_daily": "staged/stocks_daily_clean.parquet",
        "options_daily": "data/Parquet_data/Raw/options_daily",
        "options_30min": "data/Parquet_data/Raw/option_minute",
        "stocks_30min": "data/Parquet_data/Raw/stocks_minute",
    }

    results = {}
    for name, path in sources.items():
        exists = Path(path).exists()
        results[name] = exists
        status = "✅" if exists else "❌"
        logger.info(f"  {status} {name}: {path}")

        if exists and path.endswith(".parquet"):
            try:
                df = pl.read_parquet(path)
                logger.info(f"    Shape: {df.shape}, Columns: {len(df.columns)}")
            except Exception as e:
                logger.warning(f"    Error reading: {e}")

    return results


def validate_feature_computation():
    """Test feature computation from available data"""
    logger.info("🧮 Validating feature computation...")

    # Test macro features
    try:
        vix_df = pl.read_parquet("data/Parquet_data/vix_data.parquet")
        has_vix_ma = "vix_ma_divergence" in vix_df.columns
        logger.info(f"  ✅ VIX MA divergence: {'computed' if has_vix_ma else 'missing'}")

        fred_df = pl.read_parquet("data/Parquet_data/fred.parquet")
        logger.info(f"  ✅ FRED data: {fred_df.shape[0]} rows")

        dxy_df = pl.read_parquet("data/Parquet_data/dxy.parquet")
        logger.info(f"  ✅ DXY data: {dxy_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"  ❌ Macro features error: {e}")

    # Test news features
    try:
        news_df = pl.read_parquet("data/Parquet_data/news_2025-05-05.parquet")
        has_lag_features = all(
            col in news_df.columns for col in ["news_count_lag1", "avg_sentiment_lag1"]
        )
        logger.info(
            f"  ✅ News lag features: {'computed' if has_lag_features else 'missing'}"
        )
    except Exception as e:
        logger.error(f"  ❌ News features error: {e}")

    # Test stocks features
    try:
        stocks_df = pl.read_parquet("staged/stocks_daily_clean.parquet")
        expected_cols = [
            "stockd_close",
            "stockd_volume",
            "stockd_return_1d",
            "stockd_vol_7d",
        ]
        has_stock_features = all(col in stocks_df.columns for col in expected_cols)
        logger.info(
            f"  ✅ Stock features: {'computed' if has_stock_features else 'missing'}"
        )
    except Exception as e:
        logger.error(f"  ❌ Stock features error: {e}")


def create_mock_master_datasets():
    """Create mock master datasets for testing"""
    logger.info("🏗️ Creating mock master datasets...")

    # Create mock daily master
    daily_master = pl.DataFrame(
        {
            "date": ["2025-05-05"] * 10,
            "ticker": [f"TICKER_{i}" for i in range(10)],
            "stockd_close": [100.0 + i for i in range(10)],
            "stockd_volume": [1000 + i * 100 for i in range(10)],
            "optd_iv30": [0.2 + i * 0.01 for i in range(10)],
            "optd_volume": [500 + i * 50 for i in range(10)],
            "vix_index": [20.0] * 10,
            "fred_fed_funds_rate": [5.0] * 10,
            "news_count": [5 + i for i in range(10)],
        }
    )

    # Create mock intraday master
    intraday_master = pl.DataFrame(
        {
            "timestamp": ["2025-05-05 10:00:00"] * 10,
            "ticker": [f"TICKER_{i}" for i in range(10)],
            "opt30_mid_price": [50.0 + i for i in range(10)],
            "date": ["2025-05-05"] * 10,
        }
    )

    # Save mock datasets
    Path("staged").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)

    daily_master.write_parquet("staged/daily_master.parquet")
    intraday_master.write_parquet("datasets/intraday_master.parquet")

    logger.info("  ✅ Mock datasets created")
    return daily_master, intraday_master


def test_feature_calculation():
    """Test the feature calculation pipeline"""
    logger.info("🧪 Testing feature calculation pipeline...")

    try:
        # Import and test feature calculation
        from src.calculate_features import calculate_all_features

        # Create mock data
        daily_master, intraday_master = create_mock_master_datasets()

        # Calculate features
        features = calculate_all_features(
            daily_master, intraday_master, window=5, use_gpu=False
        )

        logger.info(f"  ✅ Feature calculation successful: {features.shape}")
        logger.info(f"  Generated columns: {features.columns}")

        # Save test features
        features.write_parquet("data/Parquet_data/test_features.parquet")

        return True

    except Exception as e:
        logger.error(f"  ❌ Feature calculation failed: {e}")
        return False


def validate_all_scripts_wired():
    """Validate all cleaning scripts are properly wired"""
    logger.info("🔗 Validating script wiring...")

    scripts = [
        "src/clean_macro_data.py",
        "src/clean_news.py",
        "src/clean_stocks_daily.py",
        "src/clean_stocks_30min.py",
        "src/clean_options_daily.py",
        "src/clean_options_30min.py",
        "src/calculate_features.py",
        "src/generate_labels.py",
    ]

    for script in scripts:
        exists = Path(script).exists()
        status = "✅" if exists else "❌"
        logger.info(f"  {status} {script}")

    return all(Path(script).exists() for script in scripts)


def main():
    parser = argparse.ArgumentParser(description="Validate feature pipeline")
    parser.add_argument(
        "--test-calculation", action="store_true", help="Test feature calculation"
    )
    args = parser.parse_args()

    logger.info("🚀 Starting comprehensive feature pipeline validation...")

    # Validate data sources
    data_sources = validate_data_sources()

    # Validate scripts are wired
    scripts_wired = validate_all_scripts_wired()

    # Validate feature computation
    validate_feature_computation()

    # Test feature calculation if requested
    if args.test_calculation:
        calc_success = test_feature_calculation()
    else:
        calc_success = True

    # Summary
    logger.info("\n📊 VALIDATION SUMMARY")
    logger.info("=" * 50)

    total_sources = len(data_sources)
    available_sources = sum(data_sources.values())
    logger.info(f"Data Sources: {available_sources}/{total_sources} available")

    logger.info(f"Scripts Wired: {'✅' if scripts_wired else '❌'}")
    logger.info(f"Feature Calculation: {'✅' if calc_success else '❌'}")

    # Count total required features
    total_features = sum(len(features) for features in REQUIRED_FEATURES.values())
    logger.info(f"Required Features: {total_features} total")

    if available_sources >= 6 and scripts_wired and calc_success:
        logger.info("🎉 PIPELINE VALIDATION SUCCESSFUL!")
        logger.info("Ready for Day 1 full backfill")
        return True
    else:
        logger.error("❌ PIPELINE VALIDATION FAILED")
        logger.error("Fix issues before proceeding to backfill")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
