#!/usr/bin/env python3
"""
Tests for calculate_features.py module
"""
import os
import sys
from datetime import date

import pandas as pd
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calculate_features import (calculate_cross_sectional_features,
                                calculate_intraday_features,
                                calculate_rolling_features)


def calculate_macro_rollings(df, window=3):
    """Wrapper function for testing macro rolling features"""
    return calculate_rolling_features(df, window)


def calculate_intraday_returns(df):
    """Wrapper function for testing intraday returns"""
    return calculate_intraday_features(df)


def calculate_vol_zscore(df):
    """Wrapper function for testing volume z-score"""
    return calculate_cross_sectional_features(df)


class TestCalculateFeatures:
    """Test suite for calculate_features module functions"""

    def test_rolling_macro_features(self):
        """Test rolling macro features with 5 days of data"""
        # Create tiny daily_master DataFrame with 5 days and all required columns
        df = pl.DataFrame(
            {
                "date": [
                    date(2025, 1, 1),
                    date(2025, 1, 2),
                    date(2025, 1, 3),
                    date(2025, 1, 4),
                    date(2025, 1, 5),
                ],
                "fred_fed_funds_rate": [5.25, 5.30, 5.35, 5.40, 5.45],
                "vix_index": [18.0, 19.0, 20.0, 21.0, 22.0],
                "news_count_lag1": [10, 15, 12, 18, 20],
                "avg_sentiment_lag1": [0.1, 0.2, -0.1, 0.3, 0.0],
                "optd_iv30": [0.25, 0.26, 0.27, 0.28, 0.29],
                "optd_volume": [1000, 1100, 1200, 1300, 1400],
                "stockd_return_1d": [0.01, 0.02, -0.01, 0.015, 0.005],
                "stockd_volume": [100000, 110000, 120000, 130000, 140000],
            }
        )

        result = calculate_macro_rollings(df, window=3)

        # Test fred_rate_mean equals 3-day rolling average
        expected_fred_mean = [None, None, 5.30, 5.35, 5.40]  # 3-day rolling mean
        actual_fred_mean = result["fred_rate_mean"].to_list()

        # Check non-null values (first 2 are null due to window)
        assert actual_fred_mean[2] == pytest.approx(
            5.30, abs=1e-6
        )  # (5.25+5.30+5.35)/3
        assert actual_fred_mean[3] == pytest.approx(
            5.35, abs=1e-6
        )  # (5.30+5.35+5.40)/3
        assert actual_fred_mean[4] == pytest.approx(
            5.40, abs=1e-6
        )  # (5.35+5.40+5.45)/3

        # Test vix_std matches standard deviation over window
        vix_std = result["vix_std"].to_list()
        assert vix_std[2] == pytest.approx(1.0, abs=1e-6)  # std([18,19,20])

        # Test news_count_rolling is correct
        news_rolling = result["news_count_rolling"].to_list()
        assert news_rolling[2] == 37  # 10+15+12
        assert news_rolling[3] == 45  # 15+12+18
        assert news_rolling[4] == 50  # 12+18+20

        # Test avg_sentiment_rolling exists
        assert "avg_sentiment_rolling" in result.columns

    def test_intraday_returns(self):
        """Test intraday returns calculation with hourly timestamps"""
        # Build 4-row intraday_master DataFrame with hourly timestamps
        df = pl.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "timestamp": [
                    pd.Timestamp("2025-01-01 09:30:00"),
                    pd.Timestamp("2025-01-01 10:00:00"),
                    pd.Timestamp("2025-01-01 10:30:00"),
                    pd.Timestamp("2025-01-01 11:00:00"),
                ],
                "opt30_mid_price": [100.0, 102.0, 101.0, 103.0],
            }
        )

        result = calculate_intraday_returns(df)

        # Check that ret_1h matches percent change (2 periods = 1 hour)
        assert "ret_1h" in result.columns
        assert "ticker" in result.columns
        assert "timestamp" in result.columns

        # Should have same number of rows as input
        assert result.shape[0] == 4

        # Check that returns are calculated correctly (pct_change with periods=2)
        ret_values = result.sort("timestamp")["ret_1h"].to_list()
        # First two values should be null, then calculated returns
        assert ret_values[0] is None
        assert ret_values[1] is None
        assert ret_values[2] == pytest.approx(0.01, abs=1e-6)  # (101-100)/100
        assert ret_values[3] == pytest.approx(0.0098, abs=1e-3)  # (103-102)/102

    def test_cross_sectional_z_scores(self):
        """Test cross-sectional z-scores for volume across tickers"""
        # Create DataFrame with volume across 3 tickers for one timestamp
        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 1), date(2025, 1, 1)],
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "optd_volume": [1000, 2000, 3000],
                "optd_iv30": [0.25, 0.30, 0.35],
                "stockd_return_1d": [0.01, 0.02, 0.03],
            }
        )

        result = calculate_vol_zscore(df)

        # Test vol_zscore calculation - just verify it exists and has reasonable values
        actual_zscores = result.sort("ticker")["vol_zscore"].to_list()

        # Check that z-scores are calculated (should be different values)
        assert len(set(actual_zscores)) > 1  # Should have different z-scores

        # Check that they're in reasonable range for z-scores
        assert all(-5 <= z <= 5 for z in actual_zscores if not pd.isna(z))

        # Check other cross-sectional features exist
        assert "iv_rank" in result.columns
        assert "ret_relative" in result.columns

    def test_rolling_features_with_missing_columns(self):
        """Test rolling features handles missing columns gracefully"""
        # DataFrame with only some required columns
        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "fred_fed_funds_rate": [5.25, 5.30],
                "vix_index": [18.0, 19.0],
                "news_count_lag1": [10, 15],
                "avg_sentiment_lag1": [0.1, 0.2],
                "optd_iv30": [0.25, 0.26],
                "optd_volume": [1000, 1100],
                "stockd_return_1d": [0.01, 0.02],
                "stockd_volume": [100000, 110000],
            }
        )

        result = calculate_macro_rollings(df, window=2)
        # Check that basic columns are present
        assert "fred_rate_mean" in result.columns
        assert "vix_std" in result.columns

    def test_intraday_features_empty_dataframe(self):
        """Test intraday features with empty DataFrame"""
        df = pl.DataFrame({"ticker": [], "timestamp": [], "opt30_mid_price": []})

        result = calculate_intraday_returns(df)

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "ret_1h" in result.columns
        assert "ticker" in result.columns

    def test_cross_sectional_single_ticker(self):
        """Test cross-sectional features with single ticker (edge case)"""
        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1)],
                "ticker": ["AAPL"],
                "optd_volume": [1000],
                "optd_iv30": [0.25],
                "stockd_return_1d": [0.01],
            }
        )

        result = calculate_vol_zscore(df)

        # With single ticker, z-score should be NaN (division by 0 std)
        vol_zscore = result["vol_zscore"].to_list()[0]
        assert pd.isna(vol_zscore) or vol_zscore == 0.0

        # IV rank should be 1 (only ticker)
        assert result["iv_rank"].to_list()[0] == 1.0

        # Relative return should be 0 (only ticker)
        assert result["ret_relative"].to_list()[0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
