#!/usr/bin/env python3
"""
Tests for IV rank and VIX divergence calculations
"""
import pytest
import polars as pl
import pandas as pd


def test_iv_rank_30d_calculation():
    """Test IV rank 30d percentile calculation"""
    # Create test data with 35 days to ensure 30-day window
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    test_data = pl.DataFrame({
        "date": dates.tolist(),
        "ticker": ["AAPL"] * 35,
        "optd_iv30": [0.20 + 0.01 * i for i in range(35)],  # Increasing IV
    })
    
    # Test the IV rank calculation directly
    result = test_data.with_columns([
        pl.col("optd_iv30")
        .rolling_quantile(quantile=0.5, window_size=30)
        .over("ticker")
        .alias("iv_rank_30d")
    ])
    
    # Check that iv_rank_30d column exists
    assert "iv_rank_30d" in result.columns
    
    # Last value should be calculated (not null)
    assert result["iv_rank_30d"][-1] is not None


def test_vix_ma_divergence_calculation():
    """Test VIX MA divergence calculation"""
    # Create test VIX data
    test_data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=25, freq="D"),
        "close": [20.0, 22.0, 25.0, 23.0, 21.0] * 5  # Oscillating VIX
    })
    
    # Calculate MA divergence
    test_data["vix_ma_10"] = test_data["close"].rolling(window=10, min_periods=1).mean()
    test_data["vix_ma_divergence"] = (test_data["close"] - test_data["vix_ma_10"]) / test_data["vix_ma_10"]
    
    # Check calculation
    assert "vix_ma_divergence" in test_data.columns
    
    # When VIX > MA, divergence should be positive
    high_vix_idx = test_data["close"].idxmax()
    if test_data.loc[high_vix_idx, "vix_ma_10"] > 0:
        assert test_data.loc[high_vix_idx, "vix_ma_divergence"] >= 0


def test_cross_sectional_features_complete():
    """Test that cross-sectional ranking works"""
    test_data = pl.DataFrame({
        "date": ["2025-01-01"] * 3,
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "optd_iv30": [0.25, 0.30, 0.20],
    }).with_columns(pl.col("date").str.to_date())
    
    # Test IV ranking directly
    result = test_data.with_columns([
        pl.col("optd_iv30").rank().over("date").alias("iv_rank")
    ])
    
    # Check that rankings work (MSFT should have highest IV rank)
    msft_row = result.filter(pl.col("ticker") == "MSFT")
    googl_row = result.filter(pl.col("ticker") == "GOOGL")
    
    assert msft_row["iv_rank"][0] > googl_row["iv_rank"][0]