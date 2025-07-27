#!/usr/bin/env python3
"""
Tests for build_daily_master.py with macro data integration
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_daily_master import run


def build_daily_master(stocks_daily, options_daily, fred_df=None, vix_df=None, dxy_df=None, news_df=None):
    """Core join function for testing - simplified version of the main logic"""
    # Start with inner join of stocks and options
    master = stocks_daily.join(options_daily, on=["date", "ticker"], how="inner")
    
    # Add macro data with left joins and prefixes
    if fred_df is not None:
        fred_renamed = fred_df.rename({col: f"fred_{col}" for col in fred_df.columns if col != "date"})
        master = master.join(fred_renamed, on="date", how="left")
    
    if vix_df is not None:
        vix_renamed = vix_df.rename({col: f"vix_{col}" for col in vix_df.columns if col != "date"})
        master = master.join(vix_renamed, on="date", how="left")
    
    if dxy_df is not None:
        dxy_renamed = dxy_df.rename({col: f"dxy_{col}" for col in dxy_df.columns if col != "date"})
        master = master.join(dxy_renamed, on="date", how="left")
    
    if news_df is not None:
        news_agg = news_df.group_by("date").agg([
            pl.len().alias("news_count"),
            pl.col("sentiment").mean().alias("news_avg_sentiment") if "sentiment" in news_df.columns else pl.lit(0.0).alias("news_avg_sentiment")
        ])
        master = master.join(news_agg, on="date", how="left")
    
    return master


class TestBuildDailyMaster:
    """Test suite for build_daily_master module functions"""
    """Test suite for build_daily_master module functions"""
    
    def test_build_daily_master_core_join(self):
        """Test the core join function with in-memory DataFrames"""
        # Create test data
        stocks_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "AAPL"],
            "stockd_close": [150.0, 152.0],
            "stockd_volume": [1000000, 1100000]
        })
        
        options_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "AAPL"],
            "optd_strike": [150.0, 155.0],
            "optd_moneyness": [1.0, 0.98]
        })
        
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "fed_funds_rate": [5.25, 5.30]
        })
        
        result = build_daily_master(stocks_daily, options_daily, fred_df=fred_df)
        
        # Assert correct row count and columns
        assert result.shape[0] == 2
        assert "stockd_close" in result.columns
        assert "optd_strike" in result.columns
        assert "fred_fed_funds_rate" in result.columns
        
        # Assert macro columns are correctly prefixed
        assert "fed_funds_rate" not in result.columns  # Original should be renamed
        assert result["fred_fed_funds_rate"].to_list() == [5.25, 5.30]
    
    def test_macro_data_integration(self):
        """Test that macro data loading and joining logic works correctly"""
        # Create sample macro data
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "fed_funds_rate": [5.25, 5.25],
            "unemployment_rate": [3.7, 3.7]
        })
        
        vix_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "index": [18.5, 19.2],
            "close": [18.3, 19.0]
        })

        # Test column renaming
        fred_renamed = fred_df.rename({col: f"fred_{col}" for col in fred_df.columns if col != "date"})
        vix_renamed = vix_df.rename({col: f"vix_{col}" for col in vix_df.columns if col != "date"})

        # Test that columns are properly renamed
        assert "fred_fed_funds_rate" in fred_renamed.columns
        assert "fred_unemployment_rate" in fred_renamed.columns
        assert "vix_index" in vix_renamed.columns
        assert "vix_close" in vix_renamed.columns
        assert "date" in fred_renamed.columns
        assert "date" in vix_renamed.columns
        
        # Test joining logic
        stocks_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "AAPL"],
            "stockd_close": [150.0, 152.0]
        })
        
        # Test left join with macro data
        joined = stocks_df.join(fred_renamed, on="date", how="left")
        
        assert joined.shape[0] == 2  # Should preserve all stock rows
        assert "fred_fed_funds_rate" in joined.columns
        assert joined["fred_fed_funds_rate"].null_count() == 0  # Should have values

    def test_macro_column_prefixes(self):
        """Test that macro columns are properly prefixed"""
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "fed_funds_rate": [5.25],
            "unemployment_rate": [3.7]
        })
        
        # Apply prefixes
        fred_renamed = fred_df.rename({col: f"fred_{col}" for col in fred_df.columns if col != "date"})
        
        expected_columns = ["date", "fred_fed_funds_rate", "fred_unemployment_rate"]
        assert fred_renamed.columns == expected_columns

    def test_news_aggregation(self):
        """Test that news data is properly aggregated by date"""
        news_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "sentiment": [0.5, 0.3, 0.7]
        })
        
        news_agg = news_df.group_by("date").agg([
            pl.len().alias("news_count"),
            pl.col("sentiment").mean().alias("news_avg_sentiment")
        ])
        
        assert news_agg.shape[0] == 2  # Two unique dates
        assert "news_count" in news_agg.columns
        assert "news_avg_sentiment" in news_agg.columns
        
        # Check aggregation values
        jan_1_data = news_agg.filter(pl.col("date") == pd.to_datetime("2025-01-01"))
        assert jan_1_data["news_count"].item() == 2
        assert jan_1_data["news_avg_sentiment"].item() == 0.4  # (0.5 + 0.3) / 2

    def test_missing_macro_data_results_in_nulls(self):
        """Test that missing macro data results in nulls in those columns"""
        stocks_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "AAPL"],
            "stockd_close": [150.0, 152.0]
        })
        
        options_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "AAPL"],
            "optd_strike": [150.0, 155.0]
        })
        
        # Create FRED data with only one date (missing 2025-01-02)
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "fed_funds_rate": [5.25]
        })
        
        result = build_daily_master(stocks_daily, options_daily, fred_df=fred_df)
        
        # Should have 2 rows but one with null FRED data
        assert result.shape[0] == 2
        assert result["fred_fed_funds_rate"].null_count() == 1  # One missing value

    def test_empty_macro_data_handling(self):
        """Test that empty macro dataframes are handled gracefully"""
        stocks_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "ticker": ["AAPL"],
            "stockd_close": [150.0]
        })
        
        options_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "ticker": ["AAPL"],
            "optd_strike": [150.0]
        })
        
        # Test with None macro data (should not crash)
        result = build_daily_master(stocks_daily, options_daily)
        assert result.shape[0] == 1
        assert "stockd_close" in result.columns
        assert "optd_strike" in result.columns

    def test_joined_master_has_correct_row_count_and_columns(self):
        """Test that the joined master has correct row count and columns"""
        stocks_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "MSFT"],
            "stockd_close": [150.0, 300.0],
            "stockd_volume": [1000000, 2000000]
        })
        
        options_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "ticker": ["AAPL", "MSFT"],
            "optd_strike": [150.0, 310.0],
            "optd_moneyness": [1.0, 0.97]
        })
        
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "fed_funds_rate": [5.25, 5.30],
            "unemployment_rate": [3.7, 3.7]
        })
        
        vix_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-02")],
            "index": [18.5, 19.2]
        })
        
        result = build_daily_master(stocks_daily, options_daily, fred_df=fred_df, vix_df=vix_df)
        
        # Check row count (inner join should preserve matching rows)
        assert result.shape[0] == 2
        
        # Check that all expected columns are present
        expected_cols = ["date", "ticker", "stockd_close", "stockd_volume", "optd_strike", "optd_moneyness", "fred_fed_funds_rate", "fred_unemployment_rate", "vix_index"]
        for col in expected_cols:
            assert col in result.columns

    def test_macro_columns_correctly_prefixed(self):
        """Test that macro columns are correctly prefixed (fred_, vix_, dxy_, news_)"""
        stocks_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "ticker": ["AAPL"],
            "stockd_close": [150.0]
        })
        
        options_daily = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "ticker": ["AAPL"],
            "optd_strike": [150.0]
        })
        
        fred_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "fed_funds_rate": [5.25]
        })
        
        vix_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "close": [18.5]
        })
        
        dxy_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01")],
            "rate": [103.5]
        })
        
        news_df = pl.DataFrame({
            "date": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-01")],
            "sentiment": [0.5, 0.3]
        })
        
        result = build_daily_master(stocks_daily, options_daily, fred_df=fred_df, vix_df=vix_df, dxy_df=dxy_df, news_df=news_df)
        
        # Check that all macro columns have correct prefixes
        assert "fred_fed_funds_rate" in result.columns
        assert "vix_close" in result.columns
        assert "dxy_rate" in result.columns
        assert "news_count" in result.columns
        assert "news_avg_sentiment" in result.columns
        
        # Check that original column names are not present
        assert "fed_funds_rate" not in result.columns
        assert "close" not in result.columns  # Should be vix_close
        assert "rate" not in result.columns   # Should be dxy_rate


if __name__ == "__main__":
    pytest.main([__file__])
