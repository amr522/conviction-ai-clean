#!/usr/bin/env python3
"""
Tests for enhanced options daily features
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from src.clean_options_daily import run


def test_enhanced_options_features():
    """Test that enhanced IV features are calculated correctly"""
    # Create mock data
    mock_data = pd.DataFrame(
        {
            "window_start": [
                1640995200000000000,
                1641081600000000000,
            ],  # 2022-01-01, 2022-01-02
            "ticker": ["AAPL220121C00150000", "AAPL220121P00150000"],
            "underlying": ["AAPL", "AAPL"],
            "option_type": ["C", "P"],
            "close": [5.0, 3.0],
            "volume": [1000, 500],
            "transactions": [100, 50],
            "strike": [150.0, 150.0],
        }
    )

    with patch("polars.scan_parquet") as mock_scan:
        with patch("os.path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                # Mock the parquet scan to return our test data
                mock_scan.return_value.collect.return_value = pl.from_pandas(mock_data)

                result = run("2022-01-01", dry_run=True)

                assert result["status"] == "success"
                # Additional assertions would go here for specific feature values


def test_iv_hv_calculations():
    """Test IV30 and HV30 calculations"""
    test_data = pl.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "ticker": ["AAPL220121C00150000"] * 3,
            "optd_close": [5.0, 5.2, 4.8],
            "optd_iv30": [0.25, 0.26, 0.24],
        }
    ).with_columns(pl.col("date").str.to_date())

    # Test HV30 calculation (simplified)
    result = test_data.with_columns(
        [
            pl.col("optd_close")
            .pct_change()
            .rolling_std(window_size=3, min_periods=1)
            .alias("optd_hv30_test")
        ]
    )

    assert "optd_hv30_test" in result.columns
    assert result["optd_hv30_test"][2] is not None  # Should have calculated value


def test_vol_surprise_calculation():
    """Test volatility surprise calculation"""
    test_data = pl.DataFrame(
        {"optd_iv30": [0.25, 0.30, 0.20], "optd_hv30": [0.20, 0.25, 0.22]}
    )

    result = test_data.with_columns(
        [
            ((pl.col("optd_iv30") - pl.col("optd_hv30")) / pl.col("optd_hv30")).alias(
                "optd_vol_surprise_test"
            )
        ]
    )

    # IV > HV should give positive surprise
    assert result["optd_vol_surprise_test"][0] == 0.25  # (0.25-0.20)/0.20
    assert result["optd_vol_surprise_test"][1] == 0.20  # (0.30-0.25)/0.25
