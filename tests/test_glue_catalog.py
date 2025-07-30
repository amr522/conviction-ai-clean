#!/usr/bin/env python3
"""
Test AWS Glue Data Catalog registration
"""
import os
import sys
from unittest.mock import MagicMock, patch

import boto3
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.glue_catalog import (get_parquet_schema, register_parquet_table,
                                register_pipeline_tables)


class TestGlueCatalog:
    @patch("utils.glue_catalog.boto3.client")
    @patch("utils.glue_catalog.get_parquet_schema")
    def test_register_parquet_table_success(self, mock_schema, mock_boto):
        """Test successful table registration"""
        # Mock Glue client
        mock_glue = MagicMock()
        mock_boto.return_value = mock_glue

        # Mock schema
        mock_schema.return_value = [
            {"Name": "timestamp", "Type": "timestamp"},
            {"Name": "symbol", "Type": "string"},
            {"Name": "price", "Type": "double"},
        ]

        # Mock database exists
        mock_glue.get_database.return_value = {"Database": {"Name": "test_db"}}

        # Test table registration
        result = register_parquet_table(
            database="test_db",
            table="test_table",
            s3_path="s3://test-bucket/test.parquet",
        )

        assert result is True
        mock_glue.update_table.assert_called_once()

        # Verify table input structure
        call_args = mock_glue.update_table.call_args[1]
        table_input = call_args["TableInput"]

        assert table_input["Name"] == "test_table"
        assert table_input["TableType"] == "EXTERNAL_TABLE"
        assert table_input["Parameters"]["classification"] == "parquet"
        assert len(table_input["StorageDescriptor"]["Columns"]) == 3

    @patch("utils.glue_catalog.boto3.client")
    @patch("utils.glue_catalog.get_parquet_schema")
    def test_register_parquet_table_create_database(self, mock_schema, mock_boto):
        """Test table registration with database creation"""
        # Mock Glue client
        mock_glue = MagicMock()
        mock_boto.return_value = mock_glue

        # Mock schema
        mock_schema.return_value = [{"Name": "col1", "Type": "string"}]

        # Mock database doesn't exist
        mock_glue.get_database.side_effect = (
            mock_glue.exceptions.EntityNotFoundException({}, "")
        )

        # Test table registration
        result = register_parquet_table(
            database="new_db",
            table="test_table",
            s3_path="s3://test-bucket/test.parquet",
        )

        assert result is True
        mock_glue.create_database.assert_called_once()

        # Verify database creation
        db_input = mock_glue.create_database.call_args[1]["DatabaseInput"]
        assert db_input["Name"] == "new_db"

    @patch("utils.glue_catalog.boto3.client")
    @patch("utils.glue_catalog.get_parquet_schema")
    def test_register_parquet_table_create_new_table(self, mock_schema, mock_boto):
        """Test creating new table when update fails"""
        # Mock Glue client
        mock_glue = MagicMock()
        mock_boto.return_value = mock_glue

        # Mock schema
        mock_schema.return_value = [{"Name": "col1", "Type": "string"}]

        # Mock database exists
        mock_glue.get_database.return_value = {"Database": {"Name": "test_db"}}

        # Mock update fails (table doesn't exist)
        mock_glue.update_table.side_effect = (
            mock_glue.exceptions.EntityNotFoundException({}, "")
        )

        # Test table registration
        result = register_parquet_table(
            database="test_db",
            table="new_table",
            s3_path="s3://test-bucket/test.parquet",
        )

        assert result is True
        mock_glue.update_table.assert_called_once()
        mock_glue.create_table.assert_called_once()

    @patch("utils.glue_catalog.register_parquet_table")
    def test_register_pipeline_tables(self, mock_register):
        """Test registering all pipeline tables"""
        # Mock successful registration
        mock_register.return_value = True

        # Test pipeline registration
        results = register_pipeline_tables(
            s3_bucket="test-bucket", s3_prefix="data/", database="test_db"
        )

        # Should register 5 tables
        assert len(results) == 5
        assert all(results.values())  # All should be True

        # Verify expected tables
        expected_tables = [
            "stocks_daily",
            "options_daily",
            "stocks_30min",
            "options_30min",
            "intraday_master",
        ]
        assert set(results.keys()) == set(expected_tables)

        # Verify register_parquet_table was called for each table
        assert mock_register.call_count == 5

    @patch("utils.glue_catalog.pq.ParquetDataset")
    def test_get_parquet_schema(self, mock_dataset):
        """Test Parquet schema extraction"""
        # Mock Arrow schema
        mock_schema = MagicMock()
        mock_schema.names = ["timestamp", "symbol", "price", "volume"]
        mock_schema.types = ["timestamp[us]", "string", "double", "int64"]

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.schema.to_arrow_schema.return_value = mock_schema
        mock_dataset.return_value = mock_dataset_instance

        # Test schema extraction
        columns = get_parquet_schema("s3://test-bucket/test.parquet")

        assert len(columns) == 4
        assert columns[0]["Name"] == "timestamp"
        assert columns[1]["Name"] == "symbol"
        assert columns[2]["Name"] == "price"
        assert columns[3]["Name"] == "volume"

    @patch("utils.glue_catalog.pq.ParquetDataset")
    def test_get_parquet_schema_fallback(self, mock_dataset):
        """Test schema extraction fallback on error"""
        # Mock dataset failure
        mock_dataset.side_effect = Exception("Failed to read")

        # Test schema extraction with fallback
        columns = get_parquet_schema("s3://test-bucket/test.parquet")

        # Should return fallback schema
        assert len(columns) == 3
        assert columns[0]["Name"] == "timestamp"
        assert columns[1]["Name"] == "symbol"
        assert columns[2]["Name"] == "value"


def test_glue_catalog_imports():
    """Test that Glue catalog utilities import correctly"""
    try:
        from utils.glue_catalog import (register_parquet_table,
                                        register_pipeline_tables)

        assert callable(register_parquet_table)
        assert callable(register_pipeline_tables)
    except ImportError as e:
        pytest.fail(f"Glue catalog import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
