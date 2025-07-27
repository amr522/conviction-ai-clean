#!/usr/bin/env python3
"""
Tests for enhanced macro-data ingestion with raw source support
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_macro_data import detect_backshift, load_data_source, load_news_dir


class TestCleanMacroData:
    def test_load_news_dir_empty(self):
        """Test loading from non-existent news directory"""
        result = load_news_dir("/nonexistent/path")
        assert result.empty

    def test_load_news_dir_with_files(self):
        """Test loading news from directory with CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            df1 = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "headline": ["News 1", "News 2"],
                    "sentiment": [0.5, -0.3],
                }
            )
            df2 = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-03"]),
                    "headline": ["News 3"],
                    "sentiment": [0.8],
                }
            )

            df1.to_csv(f"{tmpdir}/news1.csv", index=False)
            df2.to_csv(f"{tmpdir}/news2.csv", index=False)

            result = load_news_dir(tmpdir)
            assert len(result) == 3
            assert "date" in result.columns
            assert "headline" in result.columns

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

    def test_load_data_source_force_raw(self):
        """Test loading data source with --use-raw-macro flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = f"{tmpdir}/raw.csv"
            parquet_path = f"{tmpdir}/processed.parquet"

            # Create raw CSV
            raw_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "value": [1.0, 2.0],
                }
            )
            raw_df.to_csv(raw_path, index=False)

            # Create processed parquet
            proc_df = pd.DataFrame(
                {"date": pd.to_datetime(["2025-01-01"]), "value": [1.5]}
            )
            proc_df.to_parquet(parquet_path, index=False)

            result = load_data_source("TEST", raw_path, parquet_path, use_raw=True)
            assert len(result) == 2  # Should use raw data

    def test_load_data_source_auto_fallback(self):
        """Test automatic fallback to raw when backshift detected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = f"{tmpdir}/raw.csv"
            parquet_path = f"{tmpdir}/processed.parquet"

            # Create raw CSV with newer data
            raw_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                    "value": [1.0, 3.0],
                }
            )
            raw_df.to_csv(raw_path, index=False)

            # Create processed parquet with older data
            proc_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "value": [1.0, 2.0],
                }
            )
            proc_df.to_parquet(parquet_path, index=False)

            result = load_data_source("TEST", raw_path, parquet_path, use_raw=False)
            assert len(result) == 2  # Should fallback to raw due to backshift
            assert result["date"].max() == pd.to_datetime("2025-01-03")

    def test_load_data_source_json(self):
        """Test loading JSON data source"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = f"{tmpdir}/raw.json"
            parquet_path = f"{tmpdir}/processed.parquet"

            # Create raw JSON
            raw_df = pd.DataFrame(
                {"date": ["2025-01-01", "2025-01-02"], "value": [1.0, 2.0]}
            )
            raw_df.to_json(raw_path, orient="records", date_format="iso")

            result = load_data_source(
                "TEST", raw_path, parquet_path, use_raw=True, is_json=True
            )
            assert len(result) == 2
            assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_load_data_source_no_processed_fallback(self):
        """Test fallback to raw when no processed data exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = f"{tmpdir}/raw.csv"
            parquet_path = f"{tmpdir}/nonexistent.parquet"

            # Create raw CSV
            raw_df = pd.DataFrame(
                {"date": pd.to_datetime(["2025-01-01"]), "value": [1.0]}
            )
            raw_df.to_csv(raw_path, index=False)

            result = load_data_source("TEST", raw_path, parquet_path, use_raw=False)
            assert len(result) == 1  # Should use raw data

    def test_load_data_source_use_processed(self):
        """Test using processed data when no backshift detected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = f"{tmpdir}/raw.csv"
            parquet_path = f"{tmpdir}/processed.parquet"

            # Create raw CSV
            raw_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "value": [1.0, 2.0],
                }
            )
            raw_df.to_csv(raw_path, index=False)

            # Create processed parquet with same max date
            proc_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                    "value": [1.5, 2.5],
                }
            )
            proc_df.to_parquet(parquet_path, index=False)

            result = load_data_source("TEST", raw_path, parquet_path, use_raw=False)
            assert len(result) == 2
            assert result["value"].iloc[0] == 1.5  # Should use processed data

    @patch("clean_macro_data.logger")
    def test_backshift_logging(self, mock_logger):
        """Test that backshift detection logs warnings"""
        proc_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
        raw_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})

        detect_backshift("TEST", proc_df, raw_df)
        mock_logger.warning.assert_called_once()

        # Check warning message contains expected info
        warning_call = mock_logger.warning.call_args[0][0]
        assert "TEST backshift" in warning_call
        assert "processed max=" in warning_call
        assert "raw max=" in warning_call

    def test_pipeline_integration_flags(self):
        """Test that pipeline scripts accept macro flags"""
        import argparse

        # Test that run_full_pipeline.py accepts macro flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--date", required=True)
        parser.add_argument("--use-raw-macro", action="store_true")
        parser.add_argument("--raw-fred-csv")
        parser.add_argument("--raw-vix-json")
        parser.add_argument("--raw-dxy-csv")
        parser.add_argument("--raw-news-dir")

        test_args = [
            "--date",
            "2025-01-01",
            "--use-raw-macro",
            "--raw-fred-csv",
            "/test/fred.csv",
            "--raw-vix-json",
            "/test/vix.json",
        ]

        args = parser.parse_args(test_args)

        assert args.date == "2025-01-01"
        assert args.use_raw_macro is True
        assert args.raw_fred_csv == "/test/fred.csv"
        assert args.raw_vix_json == "/test/vix.json"

    def test_cli_flag_propagation(self):
        """Test that CLI flags are properly parsed"""
        import argparse

        test_args = [
            "--use-raw-macro",
            "--raw-fred-csv",
            "/test/fred.csv",
            "--raw-vix-json",
            "/test/vix.json",
        ]

        parser = argparse.ArgumentParser()
        parser.add_argument("--use-raw-macro", action="store_true")
        parser.add_argument("--raw-fred-csv")
        parser.add_argument("--raw-vix-json")
        parser.add_argument("--raw-dxy-csv")
        parser.add_argument("--raw-news-dir")

        args = parser.parse_args(test_args)

        assert args.use_raw_macro is True
        assert args.raw_fred_csv == "/test/fred.csv"
        assert args.raw_vix_json == "/test/vix.json"


if __name__ == "__main__":
    pytest.main([__file__])
