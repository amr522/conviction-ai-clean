#!/usr/bin/env python3
"""
Test Delta Lake utilities
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.delta_writer import (get_delta_table_history, read_delta_table,
                                write_delta_table)


class TestDeltaWriter:
    def setup_method(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=100, freq="H"),
                "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
                "price": range(100),
                "volume": range(1000, 1100),
            }
        )

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_write_delta_table_success(self, mock_spark_session):
        """Test successful Delta table write"""
        # Mock Spark session and DataFrame
        mock_spark = MagicMock()
        mock_spark_session.return_value = mock_spark

        mock_spark_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_spark_df

        mock_writer = MagicMock()
        mock_spark_df.write.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_writer.partitionBy.return_value = mock_writer

        # Test write operation
        result = write_delta_table(
            self.test_df,
            f"{self.temp_dir}/test.delta",
            partition_cols=["symbol"],
            merge_schema=True,
        )

        assert result is True
        mock_spark.createDataFrame.assert_called_once_with(self.test_df)
        mock_spark_df.write.format.assert_called_once_with("delta")
        mock_writer.save.assert_called_once()

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_read_delta_table_success(self, mock_spark_session):
        """Test successful Delta table read"""
        # Mock Spark session and DataFrame
        mock_spark = MagicMock()
        mock_spark_session.return_value = mock_spark

        mock_reader = MagicMock()
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_reader
        mock_reader.toPandas.return_value = self.test_df

        # Test read operation
        result = read_delta_table(f"{self.temp_dir}/test.delta")

        assert result is not None
        assert len(result) == len(self.test_df)
        mock_spark.read.format.assert_called_once_with("delta")
        mock_reader.load.assert_called_once()

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_read_delta_table_time_travel(self, mock_spark_session):
        """Test Delta table time-travel read"""
        # Mock Spark session and DataFrame
        mock_spark = MagicMock()
        mock_spark_session.return_value = mock_spark

        mock_reader = MagicMock()
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_reader
        mock_reader.toPandas.return_value = self.test_df

        # Test time-travel read
        result = read_delta_table(
            f"{self.temp_dir}/test.delta", timestamp_as_of="2025-01-16T00:00:00Z"
        )

        assert result is not None
        mock_reader.option.assert_called_with("timestampAsOf", "2025-01-16T00:00:00Z")

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_read_delta_table_version(self, mock_spark_session):
        """Test Delta table version read"""
        # Mock Spark session and DataFrame
        mock_spark = MagicMock()
        mock_spark_session.return_value = mock_spark

        mock_reader = MagicMock()
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_reader
        mock_reader.toPandas.return_value = self.test_df

        # Test version read
        result = read_delta_table(f"{self.temp_dir}/test.delta", version_as_of=0)

        assert result is not None
        mock_reader.option.assert_called_with("versionAsOf", "0")

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_get_delta_table_history(self, mock_spark_session):
        """Test Delta table history retrieval"""
        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark_session.return_value = mock_spark

        # Mock history DataFrame
        history_data = pd.DataFrame(
            {
                "version": [0, 1, 2],
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="D"),
                "operation": ["CREATE", "WRITE", "WRITE"],
            }
        )

        mock_history_df = MagicMock()
        mock_spark.sql.return_value = mock_history_df
        mock_history_df.toPandas.return_value = history_data

        # Test history retrieval
        result = get_delta_table_history(f"{self.temp_dir}/test.delta")

        assert result is not None
        assert len(result) == 3
        assert "version" in result.columns
        mock_spark.sql.assert_called_once()

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_write_delta_table_error_handling(self, mock_spark_session):
        """Test Delta table write error handling"""
        # Mock Spark session to raise exception
        mock_spark_session.side_effect = Exception("Spark initialization failed")

        # Test write operation with error
        result = write_delta_table(self.test_df, f"{self.temp_dir}/test.delta")

        assert result is False

    @patch("utils.delta_writer.get_delta_spark_session")
    def test_read_delta_table_error_handling(self, mock_spark_session):
        """Test Delta table read error handling"""
        # Mock Spark session to raise exception
        mock_spark_session.side_effect = Exception("Spark initialization failed")

        # Test read operation with error
        result = read_delta_table(f"{self.temp_dir}/test.delta")

        assert result is None


def test_delta_writer_imports():
    """Test that Delta writer utilities import correctly"""
    try:
        from utils.delta_writer import (get_delta_spark_session,
                                        read_delta_table, write_delta_table)

        assert callable(write_delta_table)
        assert callable(read_delta_table)
        assert callable(get_delta_spark_session)
    except ImportError as e:
        pytest.fail(f"Delta writer import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
