#!/usr/bin/env python3
"""
Tests for enhanced macro-data ingestion with raw source support
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_macro_data import detect_backshift, load_data_source, load_news_dir


class TestCleanMacroData:
    """Test suite for clean_macro_data module functions"""

    def test_load_news_dir_empty(self):
        """Test loading from non-existent news directory"""
        result = load_news_dir("/nonexistent/path")
        assert result.empty

    def test_load_news_dir_with_files(self, tmp_path):
        """Test loading news from directory with CSV files"""
        # Create test CSV files with known values
        df1 = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-02"], "sentiment": [0.5, -0.3]}
        )
        df2 = pd.DataFrame({"date": ["2025-01-03"], "sentiment": [0.8]})

        (tmp_path / "news1.csv").write_text(df1.to_csv(index=False))
        (tmp_path / "news2.csv").write_text(df2.to_csv(index=False))

        result = load_news_dir(str(tmp_path))
        assert len(result) == 3
        assert "date" in result.columns
        assert "sentiment" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_detect_backshift_no_backshift(self):
        """Test backshift detection when dates match"""
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})

        result = detect_backshift("TEST", proc_df, raw_df)
        assert result is False

    def test_detect_backshift_with_backshift(self):
        """Test backshift detection when raw is newer"""
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"])})

        result = detect_backshift("TEST", proc_df, raw_df)
        assert result is True

    def test_detect_backshift_empty_dataframes(self):
        """Test backshift detection with empty dataframes"""
        proc_df = pd.DataFrame()
        raw_df = pd.DataFrame()

        result = detect_backshift("TEST", proc_df, raw_df)
        assert result is False

    def test_load_data_source_force_raw(self, tmp_path):
        """Test loading data source with --use-raw-macro flag"""
        raw_path = tmp_path / "raw.csv"
        parquet_path = tmp_path / "processed.parquet"

        # Create raw CSV with known values
        raw_df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-02"], "value": [1.0, 2.0]}
        )
        raw_path.write_text(raw_df.to_csv(index=False))

        # Create processed parquet with different values
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "value": [1.5]})
        proc_df.to_parquet(parquet_path, index=False)

        result = load_data_source(
            "TEST", str(raw_path), str(parquet_path), use_raw=True
        )
        assert len(result) == 2  # Should use raw data
        assert result["value"].tolist() == [1.0, 2.0]

    def test_load_data_source_auto_fallback(self, tmp_path):
        """Test automatic fallback to raw when backshift detected"""
        raw_path = tmp_path / "raw.csv"
        parquet_path = tmp_path / "processed.parquet"

        # Create raw CSV with newer data
        raw_df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-03"], "value": [1.0, 3.0]}
        )
        raw_path.write_text(raw_df.to_csv(index=False))

        # Create processed parquet with older max date
        proc_df = pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "value": [1.0, 2.0]}
        )
        proc_df.to_parquet(parquet_path, index=False)

        result = load_data_source(
            "TEST", str(raw_path), str(parquet_path), use_raw=False
        )
        assert len(result) == 2  # Should fallback to raw due to backshift
        assert result["date"].max() == pd.to_datetime("2025-01-03")
        assert result["value"].tolist() == [1.0, 3.0]

    def test_load_data_source_json(self, tmp_path):
        """Test loading JSON data source"""
        raw_path = tmp_path / "raw.json"
        parquet_path = tmp_path / "processed.parquet"

        # Create raw JSON with known values
        json_data = [
            {"date": "2025-01-01", "value": 1.0},
            {"date": "2025-01-02", "value": 2.0},
        ]
        with open(raw_path, "w") as f:
            json.dump(json_data, f)

        result = load_data_source(
            "TEST", str(raw_path), str(parquet_path), use_raw=True, is_json=True
        )
        assert len(result) == 2
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["value"].tolist() == [1.0, 2.0]

    def test_load_data_source_no_processed_fallback(self, tmp_path):
        """Test fallback to raw when no processed data exists"""
        raw_path = tmp_path / "raw.csv"
        parquet_path = tmp_path / "nonexistent.parquet"

        # Create raw CSV
        raw_df = pd.DataFrame({"date": ["2025-01-01"], "value": [1.0]})
        raw_path.write_text(raw_df.to_csv(index=False))

        result = load_data_source(
            "TEST", str(raw_path), str(parquet_path), use_raw=False
        )
        assert len(result) == 1  # Should use raw data
        assert result["value"].iloc[0] == 1.0

    def test_load_data_source_use_processed(self, tmp_path):
        """Test using processed data when no backshift detected"""
        raw_path = tmp_path / "raw.csv"
        parquet_path = tmp_path / "processed.parquet"

        # Create raw CSV
        raw_df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-02"], "value": [1.0, 2.0]}
        )
        raw_path.write_text(raw_df.to_csv(index=False))

        # Create processed parquet with same max date but different values
        proc_df = pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "value": [1.5, 2.5]}
        )
        proc_df.to_parquet(parquet_path, index=False)

        result = load_data_source(
            "TEST", str(raw_path), str(parquet_path), use_raw=False
        )
        assert len(result) == 2
        assert result["value"].iloc[0] == 1.5  # Should use processed data
        assert result["value"].tolist() == [1.5, 2.5]

    def test_backshift_logging(self, caplog):
        """Test that backshift detection logs warnings"""
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})

        with caplog.at_level("WARNING"):
            result = detect_backshift("TEST", proc_df, raw_df)

        assert result is True
        assert "TEST backshift" in caplog.text
        assert "processed max=" in caplog.text
        assert "raw max=" in caplog.text

    def test_load_data_source_returns_correct_dataframe(self, tmp_path):
        """Test that load_data_source returns correct DataFrame for processed vs raw"""
        raw_path = tmp_path / "raw.csv"
        parquet_path = tmp_path / "processed.parquet"

        # Create raw CSV with specific values
        raw_df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-02"], "fed_funds_rate": [5.25, 5.30]}
        )
        raw_path.write_text(raw_df.to_csv(index=False))

        # Create processed parquet with different values
        proc_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "fed_funds_rate": [5.20, 5.25],
            }
        )
        proc_df.to_parquet(parquet_path, index=False)

        # Test processed path
        result_proc = load_data_source(
            "FRED", str(raw_path), str(parquet_path), use_raw=False
        )
        assert result_proc["fed_funds_rate"].tolist() == [5.20, 5.25]

        # Test raw path
        result_raw = load_data_source(
            "FRED", str(raw_path), str(parquet_path), use_raw=True
        )
        assert result_raw["fed_funds_rate"].tolist() == [5.25, 5.30]

    def test_detect_backshift_correctly_identifies_newer_raw(self):
        """Test that detect_backshift correctly identifies when raw max date > processed max date"""
        # Case 1: Raw is newer (should trigger fallback)
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"])})
        assert detect_backshift("TEST", proc_df, raw_df) is True

        # Case 2: Same max dates (should not trigger fallback)
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        assert detect_backshift("TEST", proc_df, raw_df) is False

        # Case 3: Processed is newer (should not trigger fallback)
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        assert (
            detect_backshift("TEST", proc_df, raw_df) is True
        )  # Different dates trigger fallback

    def test_load_news_dir_aggregates_correctly(self, tmp_path):
        """Test that load_news_dir returns DataFrame with correct aggregation"""
        # Create news files with known sentiment values
        news1_df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-01"], "sentiment": [0.5, 0.3]}
        )
        news2_df = pd.DataFrame({"date": ["2025-01-02"], "sentiment": [0.7]})

        (tmp_path / "news1.csv").write_text(news1_df.to_csv(index=False))
        (tmp_path / "news2.csv").write_text(news2_df.to_csv(index=False))

        result = load_news_dir(str(tmp_path))

        # Should have 3 total rows (2 from news1 + 1 from news2)
        assert len(result) == 3
        assert set(result["sentiment"].tolist()) == {0.5, 0.3, 0.7}

        # Check date parsing
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        expected_dates = {"2025-01-01", "2025-01-02"}
        actual_dates = set(result["date"].dt.strftime("%Y-%m-%d").unique())
        assert actual_dates == expected_dates


if __name__ == "__main__":
    pytest.main([__file__])
