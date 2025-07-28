#!/usr/bin/env python3
"""
Performance benchmarks for feature calculation pipeline
"""
import os
import sys
from datetime import date

import polars as pl
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calculate_features import (calculate_cross_sectional_features,
                                calculate_intraday_features,
                                calculate_rolling_features)


def calculate_macro_rollings(df, window=30):
    """Wrapper for macro rolling features benchmark"""
    return calculate_rolling_features(df, window)


def calculate_intraday_returns(df):
    """Wrapper for intraday returns benchmark"""
    return calculate_intraday_features(df)


def calculate_vol_zscore(df):
    """Wrapper for volume z-score benchmark"""
    return calculate_cross_sectional_features(df)


class TestBenchmarks:
    """Performance benchmarks for feature calculation functions"""

    @pytest.mark.benchmark(group="macro_rolling", min_rounds=5)
    def test_bench_macro_rollings(self, benchmark):
        """Benchmark macro rolling features with 6 months of daily data"""
        # Generate 181 days of data (6 months)
        dates = pl.date_range(date(2025, 1, 1), date(2025, 6, 30), "1d", eager=True)
        df = pl.DataFrame(
            {
                "date": dates,
                "fred_fed_funds_rate": [3.0 + i * 0.01 for i in range(181)],
                "vix_index": [20.0 + i * 0.1 for i in range(181)],
                "news_count": [5 + i % 10 for i in range(181)],
                "optd_iv30": [0.25 + i * 0.001 for i in range(181)],
                "optd_volume": [1000 + i * 10 for i in range(181)],
                "stockd_return_1d": [0.01 + (i % 20 - 10) * 0.001 for i in range(181)],
                "stockd_volume": [100000 + i * 1000 for i in range(181)],
            }
        )

        result = benchmark(lambda: calculate_macro_rollings(df, window=30))

        # Verify benchmark produces valid output
        assert result is not None
        assert result.shape[0] == 181

    @pytest.mark.benchmark(group="intraday_returns", min_rounds=5)
    def test_bench_intraday_returns(self, benchmark):
        """Benchmark intraday returns with 10,000 rows of hourly data"""
        # Generate 10,000 hours of data (~1.1 years)
        timestamps = pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0),
            end=pl.datetime(2026, 2, 15, 15, 0),
            interval="1h",
            eager=True,
        )
        # Take exactly 10000 timestamps
        timestamps = timestamps[:10000]
        df = pl.DataFrame(
            {
                "ticker": ["AAPL"] * len(timestamps),
                "timestamp": timestamps,
                "opt30_mid_price": [
                    100.0 + i * 0.01 + (i % 100 - 50) * 0.1
                    for i in range(len(timestamps))
                ],
            }
        )

        result = benchmark(lambda: calculate_intraday_returns(df))

        # Verify benchmark produces valid output
        assert result is not None
        assert result.shape[0] == len(timestamps)  # Should match input size

    @pytest.mark.benchmark(group="vol_zscore", min_rounds=5)
    def test_bench_vol_zscore(self, benchmark):
        """Benchmark volume z-score with 100 tickers over 1,440 timestamps"""
        # Generate 100 tickers × 1,440 10-minute intervals (10 days)
        n_tickers = 100
        n_timestamps = 1440
        total_rows = n_tickers * n_timestamps

        # Use datetime_range for 10-minute intervals
        timestamps = pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0),
            end=pl.datetime(2025, 1, 10, 23, 50),
            interval="10m",
            eager=True,
        )[:n_timestamps]
        repeated_timestamps = []
        for _ in range(n_tickers):
            repeated_timestamps.extend(timestamps.to_list())

        df = pl.DataFrame(
            {
                "date": repeated_timestamps,
                "ticker": [f"TICKER_{i%n_tickers:03d}" for i in range(total_rows)],
                "optd_volume": [
                    1000 + i * 10 + (i % 1000 - 500) * 5 for i in range(total_rows)
                ],
                "optd_iv30": [0.25 + (i % 100) * 0.001 for i in range(total_rows)],
                "stockd_return_1d": [
                    0.01 + (i % 200 - 100) * 0.0001 for i in range(total_rows)
                ],
            }
        )

        result = benchmark(lambda: calculate_vol_zscore(df))

        # Verify benchmark produces valid output
        assert result is not None
        assert result.shape[0] == total_rows

    @pytest.mark.benchmark(group="feature_pipeline", min_rounds=3)
    def test_bench_full_pipeline(self, benchmark):
        """Benchmark full feature calculation pipeline"""
        # Create realistic dataset sizes
        n_days = 90  # 3 months
        n_tickers = 50

        # Daily master data
        dates = pl.date_range(date(2025, 1, 1), date(2025, 3, 31), "1d", eager=True)[
            :n_days
        ]
        repeated_dates = []
        for _ in range(n_tickers):
            repeated_dates.extend(dates.to_list())

        daily_df = pl.DataFrame(
            {
                "date": repeated_dates,
                "ticker": [
                    f"STOCK_{i%n_tickers:02d}" for i in range(n_days * n_tickers)
                ],
                "fred_fed_funds_rate": [
                    3.0 + (i % 100) * 0.01 for i in range(n_days * n_tickers)
                ],
                "vix_index": [20.0 + (i % 50) * 0.5 for i in range(n_days * n_tickers)],
                "news_count": [5 + i % 15 for i in range(n_days * n_tickers)],
                "optd_iv30": [
                    0.25 + (i % 200) * 0.001 for i in range(n_days * n_tickers)
                ],
                "optd_volume": [1000 + i * 50 for i in range(n_days * n_tickers)],
                "stockd_return_1d": [
                    0.01 + (i % 400 - 200) * 0.0001 for i in range(n_days * n_tickers)
                ],
                "stockd_volume": [100000 + i * 2000 for i in range(n_days * n_tickers)],
            }
        )

        def full_pipeline():
            # Step 1: Rolling features
            rolling_result = daily_df.group_by("ticker").map_groups(
                lambda df: calculate_rolling_features(df.sort("date"), 30)
            )

            # Step 2: Cross-sectional features
            cross_sectional_result = calculate_cross_sectional_features(rolling_result)

            return cross_sectional_result

        result = benchmark(full_pipeline)

        # Verify pipeline produces valid output
        assert result is not None
        assert result.shape[0] == n_days * n_tickers
        assert "vol_zscore" in result.columns
        assert "fred_rate_mean" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
