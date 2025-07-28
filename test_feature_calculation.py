#!/usr/bin/env python3
"""
Simple test of feature calculation without GPU dependencies
"""

import os
import sys
from pathlib import Path

import polars as pl

# Add src to path
sys.path.append("src")


def create_test_data():
    """Create minimal test data for feature calculation"""

    # Create test daily data
    daily_data = pl.DataFrame(
        {
            "date": ["2025-05-05"] * 5,
            "ticker": [f"TEST{i}" for i in range(5)],
            "stockd_close": [100.0 + i for i in range(5)],
            "stockd_volume": [1000 + i * 100 for i in range(5)],
            "stockd_return_1d": [0.01 * i for i in range(5)],
            "stockd_vol_7d": [0.1 + i * 0.01 for i in range(5)],
        }
    )

    # Create test news data
    news_data = pl.DataFrame(
        {
            "date": ["2025-05-05"] * 5,
            "ticker": [f"TEST{i}" for i in range(5)],
            "news_count_lag1": [5 + i for i in range(5)],
            "avg_sentiment_lag1": [0.1 * i for i in range(5)],
        }
    )

    # Create test macro data
    macro_data = pl.DataFrame(
        {
            "date": ["2025-05-05"],
            "vix_value": [20.0],
            "vix_ma_divergence": [0.05],
            "dxy_value": [105.0],
            "fred_fed_funds_rate": [5.0],
        }
    )

    return daily_data, news_data, macro_data


def test_basic_feature_joins():
    """Test basic feature joining without complex calculations"""
    print("🧪 Testing basic feature joins...")

    daily_data, news_data, macro_data = create_test_data()

    # Join features
    features = daily_data.join(news_data, on=["date", "ticker"], how="left")
    features = features.join(macro_data, on="date", how="left")

    print(f"✅ Feature join successful: {features.shape}")
    print(f"Columns: {features.columns}")

    # Add some computed features
    features = features.with_columns(
        [
            # IV rank simulation
            pl.lit(0.5).alias("iv_rank_30d"),
            # Volume change
            (pl.col("stockd_volume").pct_change()).alias("stockd_volume_pct_change"),
            # Simple beta (mock)
            pl.lit(1.0).alias("stockd_beta_spy"),
            # Days to earnings (mock)
            pl.lit(30).alias("stockd_days_to_earnings"),
            # Earnings flag (mock)
            pl.lit(False).alias("stockd_earnings_flag"),
        ]
    )

    print(f"✅ Computed features added: {features.shape}")

    # Save test features
    Path("data/Parquet_data").mkdir(parents=True, exist_ok=True)
    features.write_parquet("data/Parquet_data/test_features_basic.parquet")

    return features


def validate_required_feature_coverage():
    """Check which required features we can generate"""
    print("📊 Validating required feature coverage...")

    # Features we can generate from current data
    available_features = {
        "macro": [
            "vix_value",
            "vix_ma_divergence",
            "dxy_value",
        ],  # iv_rank_30d needs options data
        "news": ["news_count_lag1", "avg_sentiment_lag1"],
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
        "options_daily": [],  # Need to process options data
        "options_30min": [],  # Need to process options data
        "stocks_30min": [],  # Need to process 30min data
    }

    total_available = sum(len(features) for features in available_features.values())
    total_required = 32  # From roadmap

    print(f"Available features: {total_available}/{total_required}")

    for category, features in available_features.items():
        if features:
            print(f"  ✅ {category}: {len(features)} features")
        else:
            print(f"  ⚠️  {category}: needs data processing")

    return total_available >= 15  # At least half the features


def main():
    print("🚀 Testing feature calculation pipeline...")

    # Test basic joins
    features = test_basic_feature_joins()

    # Validate coverage
    coverage_ok = validate_required_feature_coverage()

    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Basic feature joins: ✅")
    print(f"Feature coverage: {'✅' if coverage_ok else '⚠️'}")

    if coverage_ok:
        print("🎉 FEATURE PIPELINE READY")
        print("Core features can be computed from available data")
        print("Options and 30min features need data processing")
        return True
    else:
        print("❌ FEATURE PIPELINE NEEDS WORK")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
