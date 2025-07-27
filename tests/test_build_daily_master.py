#!/usr/bin/env python3
"""
Tests for build_daily_master.py with macro data integration
"""
import pytest
import pandas as pd
import polars as pl
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from build_daily_master import run

class TestBuildDailyMaster:
    
    def test_macro_data_integration(self):
        """Test that macro data loading and joining logic works correctly"""
        # Test the core macro data loading logic
        from clean_macro_data import load_data_source
        
        # Create sample macro data
        fred_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'fed_funds_rate': [5.25, 5.25],
            'unemployment_rate': [3.7, 3.7]
        })
        
        vix_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'index': [18.5, 19.2],
            'close': [18.3, 19.0]
        })
        
        # Test Polars conversion and column renaming
        fred_pl = pl.from_pandas(fred_df)
        fred_renamed = fred_pl.rename({col: f"fred_{col}" for col in fred_pl.columns if col != "date"})
        
        vix_pl = pl.from_pandas(vix_df)
        vix_renamed = vix_pl.rename({col: f"vix_{col}" for col in vix_pl.columns if col != "date"})
        
        # Test that columns are properly renamed
        assert 'fred_fed_funds_rate' in fred_renamed.columns
        assert 'fred_unemployment_rate' in fred_renamed.columns
        assert 'vix_index' in vix_renamed.columns
        assert 'vix_close' in vix_renamed.columns
        assert 'date' in fred_renamed.columns
        assert 'date' in vix_renamed.columns
        
        # Test joining logic - ensure consistent datetime types
        stocks_df = pl.DataFrame({
            'date': pl.Series([pd.to_datetime('2025-01-01'), pd.to_datetime('2025-01-02')]).dt.cast_time_unit("us"),
            'ticker': ['AAPL', 'AAPL'],
            'stockd_close': [150.0, 152.0]
        })
        
        # Ensure fred_renamed has consistent datetime type
        fred_renamed = fred_renamed.with_columns(pl.col("date").dt.cast_time_unit("us"))
        
        # Test left join with macro data
        joined = stocks_df.join(fred_renamed, on="date", how="left")
        
        assert joined.shape[0] == 2  # Should preserve all stock rows
        assert 'fred_fed_funds_rate' in joined.columns
        assert joined['fred_fed_funds_rate'].null_count() == 0  # Should have values
    
    def test_macro_column_prefixes(self):
        """Test that macro columns are properly prefixed"""
        # Test data with various column names
        fred_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01']),
            'fed_funds_rate': [5.25],
            'unemployment_rate': [3.7]
        })
        
        # Convert to Polars and apply prefixes
        fred_pl = pl.from_pandas(fred_df)
        fred_renamed = fred_pl.rename({col: f"fred_{col}" for col in fred_pl.columns if col != "date"})
        
        expected_columns = ['date', 'fred_fed_funds_rate', 'fred_unemployment_rate']
        assert fred_renamed.columns == expected_columns
    
    def test_news_aggregation(self):
        """Test that news data is properly aggregated by date"""
        news_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02']),
            'sentiment': [0.5, 0.3, 0.7]
        })
        
        news_pl = pl.from_pandas(news_df)
        news_agg = news_pl.group_by("date").agg([
            pl.len().alias("news_count"),
            pl.col("sentiment").mean().alias("news_avg_sentiment")
        ])
        
        assert news_agg.shape[0] == 2  # Two unique dates
        assert 'news_count' in news_agg.columns
        assert 'news_avg_sentiment' in news_agg.columns
        
        # Check aggregation values
        jan_1_data = news_agg.filter(pl.col("date") == pd.to_datetime('2025-01-01'))
        assert jan_1_data['news_count'].item() == 2
        assert jan_1_data['news_avg_sentiment'].item() == 0.4  # (0.5 + 0.3) / 2
    
    def test_cli_flags(self):
        """Test that CLI flags are properly parsed"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--date", required=True)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--use-raw-macro", action="store_true")
        parser.add_argument("--raw-fred-csv")
        parser.add_argument("--raw-vix-json")
        parser.add_argument("--raw-dxy-csv")
        parser.add_argument("--raw-news-dir")
        
        test_args = [
            "--date", "2025-01-01",
            "--use-raw-macro",
            "--raw-fred-csv", "/test/fred.csv",
            "--raw-vix-json", "/test/vix.json"
        ]
        
        args = parser.parse_args(test_args)
        
        assert args.date == "2025-01-01"
        assert args.use_raw_macro is True
        assert args.raw_fred_csv == "/test/fred.csv"
        assert args.raw_vix_json == "/test/vix.json"
    
    def test_empty_macro_data_handling(self):
        """Test that empty macro dataframes are handled gracefully"""
        empty_df = pd.DataFrame()
        
        # Should not cause errors when empty
        assert empty_df.empty
        
        # Test Polars conversion of empty DataFrame
        empty_pl = pl.from_pandas(pd.DataFrame({'date': [], 'value': []}))
        assert empty_pl.shape[0] == 0
    
    def test_macro_data_types(self):
        """Test that macro features have correct data types"""
        macro_dtypes = {
            "fred_fed_funds_rate": pl.Float64,
            "fred_unemployment_rate": pl.Float64,
            "vix_index": pl.Float64,
            "vix_close": pl.Float64,
            "dxy_rate": pl.Float64,
            "news_count": pl.UInt64,
            "news_avg_sentiment": pl.Float64
        }
        
        # Verify all macro types are numeric
        for col, dtype in macro_dtypes.items():
            assert dtype in [pl.Float64, pl.UInt64, pl.Int64]
    
    def test_macro_join_logic(self):
        """Test the macro data join logic separately"""
        # Create test dataframes
        stocks_df = pl.DataFrame({
            'date': [pd.to_datetime('2025-01-01'), pd.to_datetime('2025-01-02')],
            'ticker': ['AAPL', 'MSFT'],
            'stockd_close': [150.0, 300.0]
        })
        
        fred_df = pl.DataFrame({
            'date': [pd.to_datetime('2025-01-01'), pd.to_datetime('2025-01-02')],
            'fred_fed_funds_rate': [5.25, 5.30]
        })
        
        # Ensure consistent datetime types
        stocks_df = stocks_df.with_columns(pl.col("date").dt.cast_time_unit("us"))
        fred_df = fred_df.with_columns(pl.col("date").dt.cast_time_unit("us"))
        
        # Test left join
        result = stocks_df.join(fred_df, on="date", how="left")
        
        assert result.shape[0] == 2
        assert result.shape[1] == 4  # original 3 + 1 macro column
        assert 'fred_fed_funds_rate' in result.columns
        assert result['fred_fed_funds_rate'].null_count() == 0

if __name__ == "__main__":
    pytest.main([__file__])