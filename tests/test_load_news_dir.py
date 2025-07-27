#!/usr/bin/env python3
"""
Tests for load_news_dir function from clean_macro_data module
"""
import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_macro_data import load_news_dir


class TestLoadNewsDir:
    """Test suite for load_news_dir function"""
    
    def test_load_news_dir_empty_directory(self, tmp_path):
        """Test loading from empty news directory"""
        result = load_news_dir(str(tmp_path))
        assert result.empty
    
    def test_load_news_dir_nonexistent_directory(self):
        """Test loading from non-existent directory"""
        result = load_news_dir("/nonexistent/path")
        assert result.empty
    
    def test_load_news_dir_single_file(self, tmp_path):
        """Test loading news from single CSV file"""
        # Create single news file with known values
        news_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "sentiment": [0.5, -0.3]
        })
        
        news_file = tmp_path / "news.csv"
        news_file.write_text(news_df.to_csv(index=False))
        
        result = load_news_dir(str(tmp_path))
        
        assert len(result) == 2
        assert "date" in result.columns
        assert "sentiment" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["sentiment"].tolist() == [0.5, -0.3]
    
    def test_load_news_dir_multiple_files(self, tmp_path):
        """Test loading news from multiple CSV files"""
        # Create first news file
        news1_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "sentiment": [0.5, -0.3]
        })
        (tmp_path / "news1.csv").write_text(news1_df.to_csv(index=False))
        
        # Create second news file
        news2_df = pd.DataFrame({
            "date": ["2025-01-03"],
            "sentiment": [0.8]
        })
        (tmp_path / "news2.csv").write_text(news2_df.to_csv(index=False))
        
        result = load_news_dir(str(tmp_path))
        
        # Should have combined data from both files
        assert len(result) == 3
        assert set(result["sentiment"].tolist()) == {0.5, -0.3, 0.8}
        
        # Check date parsing
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        expected_dates = {"2025-01-01", "2025-01-02", "2025-01-03"}
        actual_dates = set(result["date"].dt.strftime("%Y-%m-%d").tolist())
        assert actual_dates == expected_dates
    
    def test_load_news_dir_with_aggregation_ready_data(self, tmp_path):
        """Test that returned DataFrame has one row per unique date with expected columns"""
        # Create news data with multiple entries per date
        news_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "sentiment": [0.6, 0.4, 0.7]
        })
        (tmp_path / "news.csv").write_text(news_df.to_csv(index=False))
        
        result = load_news_dir(str(tmp_path))
        
        # Raw data should have 3 rows (not aggregated yet)
        assert len(result) == 3
        assert "date" in result.columns
        assert "sentiment" in result.columns
        
        # Convert to Polars and test aggregation logic
        news_pl = pl.from_pandas(result)
        news_agg = news_pl.group_by("date").agg([
            pl.len().alias("news_count"),
            pl.col("sentiment").mean().alias("news_avg_sentiment")
        ])
        
        # After aggregation should have one row per unique date
        assert news_agg.shape[0] == 2  # Two unique dates
        assert "news_count" in news_agg.columns
        assert "news_avg_sentiment" in news_agg.columns
        
        # Check aggregated values
        jan_1_data = news_agg.filter(pl.col("date") == pd.to_datetime("2025-01-01"))
        assert jan_1_data["news_count"].item() == 2
        assert jan_1_data["news_avg_sentiment"].item() == 0.5  # (0.6 + 0.4) / 2
        
        jan_2_data = news_agg.filter(pl.col("date") == pd.to_datetime("2025-01-02"))
        assert jan_2_data["news_count"].item() == 1
        assert jan_2_data["news_avg_sentiment"].item() == 0.7
    
    def test_load_news_dir_ignores_non_csv_files(self, tmp_path):
        """Test that non-CSV files are ignored"""
        # Create CSV file
        news_df = pd.DataFrame({
            "date": ["2025-01-01"],
            "sentiment": [0.5]
        })
        (tmp_path / "news.csv").write_text(news_df.to_csv(index=False))
        
        # Create non-CSV files that should be ignored
        (tmp_path / "readme.txt").write_text("This should be ignored")
        (tmp_path / "data.json").write_text('{"test": "data"}')
        
        result = load_news_dir(str(tmp_path))
        
        # Should only load from CSV file
        assert len(result) == 1
        assert result["sentiment"].iloc[0] == 0.5
    
    def test_load_news_dir_handles_malformed_csv(self, tmp_path, caplog):
        """Test that malformed CSV files are handled gracefully"""
        # Create valid CSV
        valid_df = pd.DataFrame({
            "date": ["2025-01-01"],
            "sentiment": [0.5]
        })
        (tmp_path / "valid.csv").write_text(valid_df.to_csv(index=False))
        
        # Create malformed CSV
        (tmp_path / "malformed.csv").write_text("invalid,csv,content\nwith,missing,headers")
        
        with caplog.at_level("WARNING"):
            result = load_news_dir(str(tmp_path))
        
        # Should load valid file and warn about malformed file
        assert len(result) == 1
        assert result["sentiment"].iloc[0] == 0.5
        assert "Failed to load" in caplog.text
        assert "malformed.csv" in caplog.text
    
    def test_load_news_dir_date_column_parsing(self, tmp_path):
        """Test that date column is properly parsed as datetime"""
        # Test simple date format that we know works
        news_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "sentiment": [0.1, 0.2, 0.3]
        })
        (tmp_path / "news.csv").write_text(news_df.to_csv(index=False))
        
        result = load_news_dir(str(tmp_path))
        
        # Check that date column is datetime type (load_news_dir calls pd.read_csv with parse_dates=["date"])
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        
        # Check that all dates were parsed correctly
        assert len(result) == 3
        expected_dates = [
            pd.to_datetime("2025-01-01"),
            pd.to_datetime("2025-01-02"),
            pd.to_datetime("2025-01-03")
        ]
        # Sort both series for comparison since order is not guaranteed
        result_sorted = result.sort_values("date").reset_index(drop=True)
        expected_sorted = pd.Series(expected_dates, name="date")
        pd.testing.assert_series_equal(result_sorted["date"], expected_sorted)


if __name__ == "__main__":
    pytest.main([__file__])