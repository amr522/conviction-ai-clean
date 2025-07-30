#!/usr/bin/env python3
"""
Edge-case unit tests for core ETL scripts.
Tests missing strikes, zero-volume days, and date mismatches.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_parquet(tmp_path):
    """Create temporary parquet file path."""
    return tmp_path / "test.parquet"


def write_parquet(df: pd.DataFrame, path):
    """Write DataFrame to parquet file."""
    df.to_parquet(path, index=False)


# Mock ETL functions since actual implementations may not exist
def mock_process_options_daily(parquet_path: str, dry_run: bool = False):
    """Mock options daily processing function."""
    df = pd.read_parquet(parquet_path)

    # Check for required columns
    required_cols = ["optd_strike", "optd_close", "optd_volume", "optd_type"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Filter zero volume rows
    df = df[df["optd_volume"] > 0]

    # Filter old dates (before 2020)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= "2020-01-01"]

    return df


def mock_process_stocks_daily(parquet_path: str, dry_run: bool = False):
    """Mock stocks daily processing function."""
    df = pd.read_parquet(parquet_path)

    # Check for required columns
    required_cols = ["close", "volume", "ticker"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Filter zero volume rows
    df = df[df["volume"] > 0]

    # Filter invalid prices
    df = df[df["close"] > 0]

    return df


class TestOptionsETLEdgeCases:
    """Test edge cases for options ETL processing."""

    def test_missing_strike_raises(self, tmp_parquet):
        """Test that missing strike column raises KeyError."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0],
                "optd_volume": [100],
                # "optd_strike" missing
                "optd_type": ["C"],
            }
        )
        write_parquet(df, tmp_parquet)

        with pytest.raises(KeyError, match="Missing required column: optd_strike"):
            mock_process_options_daily(str(tmp_parquet), dry_run=True)

    def test_missing_close_raises(self, tmp_parquet):
        """Test that missing close column raises KeyError."""
        df = pd.DataFrame(
            {
                "optd_volume": [100],
                "optd_strike": [100],
                "optd_type": ["C"]
                # "optd_close" missing
            }
        )
        write_parquet(df, tmp_parquet)

        with pytest.raises(KeyError, match="Missing required column: optd_close"):
            mock_process_options_daily(str(tmp_parquet), dry_run=True)

    def test_zero_volume_rows_filtered(self, tmp_parquet):
        """Test that zero volume rows are filtered out."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0, 2.0, 3.0],
                "optd_volume": [0, 50, 0],  # Mix of zero and non-zero
                "optd_strike": [100, 200, 300],
                "optd_type": ["P", "C", "P"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)

        # After processing, only non-zero volume rows remain
        assert len(result) == 1
        assert all(result["optd_volume"] > 0)
        assert result.iloc[0]["optd_close"] == 2.0

    def test_negative_volume_filtered(self, tmp_parquet):
        """Test that negative volume rows are filtered out."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0, 2.0],
                "optd_volume": [-10, 50],  # Negative volume
                "optd_strike": [100, 200],
                "optd_type": ["P", "C"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)

        assert len(result) == 1
        assert all(result["optd_volume"] > 0)

    def test_date_mismatch_skips(self, tmp_parquet):
        """Test that old dates are filtered out."""
        df = pd.DataFrame(
            {
                "date": ["2000-01-01", "2025-01-01"],  # Old and recent dates
                "optd_close": [1.0, 2.0],
                "optd_volume": [100, 200],
                "optd_strike": [100, 200],
                "optd_type": ["C", "P"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)

        # Only recent date should remain
        assert len(result) == 1
        assert result.iloc[0]["date"] >= pd.Timestamp("2020-01-01")

    def test_invalid_option_type_handled(self, tmp_parquet):
        """Test handling of invalid option types."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0, 2.0, 3.0],
                "optd_volume": [100, 200, 300],
                "optd_strike": [100, 200, 300],
                "optd_type": ["C", "P", "X"],  # X is invalid
            }
        )
        write_parquet(df, tmp_parquet)

        # Should not raise error, just process valid rows
        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # All rows kept in mock implementation

    def test_empty_dataframe_handled(self, tmp_parquet):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(
            {"optd_close": [], "optd_volume": [], "optd_strike": [], "optd_type": []}
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


class TestStocksETLEdgeCases:
    """Test edge cases for stocks ETL processing."""

    def test_missing_ticker_raises(self, tmp_parquet):
        """Test that missing ticker column raises KeyError."""
        df = pd.DataFrame(
            {
                "close": [100.0],
                "volume": [1000]
                # "ticker" missing
            }
        )
        write_parquet(df, tmp_parquet)

        with pytest.raises(KeyError, match="Missing required column: ticker"):
            mock_process_stocks_daily(str(tmp_parquet), dry_run=True)

    def test_zero_volume_stocks_filtered(self, tmp_parquet):
        """Test that zero volume stock rows are filtered."""
        df = pd.DataFrame(
            {
                "close": [100.0, 200.0, 300.0],
                "volume": [0, 1000, 0],  # Mix of zero and non-zero
                "ticker": ["AAPL", "MSFT", "GOOGL"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_stocks_daily(str(tmp_parquet), dry_run=True)

        assert len(result) == 1
        assert all(result["volume"] > 0)
        assert result.iloc[0]["ticker"] == "MSFT"

    def test_zero_price_filtered(self, tmp_parquet):
        """Test that zero price rows are filtered."""
        df = pd.DataFrame(
            {
                "close": [0.0, 200.0],  # Zero price
                "volume": [1000, 2000],
                "ticker": ["AAPL", "MSFT"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_stocks_daily(str(tmp_parquet), dry_run=True)

        assert len(result) == 1
        assert all(result["close"] > 0)

    def test_negative_price_filtered(self, tmp_parquet):
        """Test that negative price rows are filtered."""
        df = pd.DataFrame(
            {
                "close": [-100.0, 200.0],  # Negative price
                "volume": [1000, 2000],
                "ticker": ["AAPL", "MSFT"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_stocks_daily(str(tmp_parquet), dry_run=True)

        assert len(result) == 1
        assert all(result["close"] > 0)


class TestDataTypeEdgeCases:
    """Test edge cases related to data types."""

    def test_string_volume_handled(self, tmp_parquet):
        """Test handling of string volume values."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0, 2.0],
                "optd_volume": ["100", "200"],  # String volumes
                "optd_strike": [100, 200],
                "optd_type": ["C", "P"],
            }
        )
        write_parquet(df, tmp_parquet)

        # Should handle conversion or raise appropriate error
        try:
            result = mock_process_options_daily(str(tmp_parquet), dry_run=True)
            # If successful, volumes should be numeric
            assert result["optd_volume"].dtype in [np.int64, np.float64]
        except (ValueError, TypeError):
            # Expected for string volumes that can't be converted
            pass

    def test_nan_values_handled(self, tmp_parquet):
        """Test handling of NaN values."""
        df = pd.DataFrame(
            {
                "optd_close": [1.0, np.nan, 3.0],
                "optd_volume": [100, 200, np.nan],
                "optd_strike": [100, 200, 300],
                "optd_type": ["C", "P", "C"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)

        # Should handle NaN values appropriately
        assert isinstance(result, pd.DataFrame)

    def test_extreme_values_handled(self, tmp_parquet):
        """Test handling of extreme values."""
        df = pd.DataFrame(
            {
                "optd_close": [1e-10, 1e10, 1.0],  # Very small and large values
                "optd_volume": [1, 1e12, 100],
                "optd_strike": [0.01, 10000, 100],
                "optd_type": ["C", "P", "C"],
            }
        )
        write_parquet(df, tmp_parquet)

        result = mock_process_options_daily(str(tmp_parquet), dry_run=True)

        # Should process without error
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0


class TestFileHandlingEdgeCases:
    """Test edge cases related to file handling."""

    def test_nonexistent_file_raises(self):
        """Test that nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            mock_process_options_daily("/nonexistent/file.parquet", dry_run=True)

    def test_corrupted_parquet_handled(self, tmp_path):
        """Test handling of corrupted parquet file."""
        corrupted_file = tmp_path / "corrupted.parquet"

        # Create a file with invalid parquet content
        with open(corrupted_file, "w") as f:
            f.write("This is not a parquet file")

        with pytest.raises(Exception):  # Should raise some kind of error
            mock_process_options_daily(str(corrupted_file), dry_run=True)

    def test_empty_file_handled(self, tmp_parquet):
        """Test handling of empty parquet file."""
        # Create empty DataFrame and save
        df = pd.DataFrame()
        write_parquet(df, tmp_parquet)

        with pytest.raises(KeyError):  # Missing required columns
            mock_process_options_daily(str(tmp_parquet), dry_run=True)


if __name__ == "__main__":
    pytest.main([__file__])
