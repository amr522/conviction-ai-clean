#!/usr/bin/env python3
"""
Test suite for schema validation functionality.
"""

import pytest
import tempfile
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validate_schemas import (
    validate_parquet_schema, 
    SchemaValidationError,
    load_expected_schema,
    compare_schemas,
    _types_match
)


def test_load_expected_schema():
    """Test loading schema from JSON specification."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {
            "test_dataset": {
                "col1": "float64",
                "col2": "string",
                "col3": "bool"
            }
        }
        json.dump(schema_spec, f)
        spec_file = f.name
    
    try:
        result = load_expected_schema(spec_file, "test_dataset")
        assert result == {"col1": "float64", "col2": "string", "col3": "bool"}
    finally:
        os.unlink(spec_file)


def test_load_expected_schema_missing_dataset():
    """Test error when dataset type not found in schema spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {"other_dataset": {"col1": "float64"}}
        json.dump(schema_spec, f)
        spec_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Dataset type 'missing_dataset' not found"):
            load_expected_schema(spec_file, "missing_dataset")
    finally:
        os.unlink(spec_file)


def test_compare_schemas():
    """Test schema comparison functionality."""
    actual = {
        "col1": "double",
        "col2": "string", 
        "col3": "int64",
        "extra_col": "bool"
    }
    
    expected = {
        "col1": "float64",
        "col2": "string",
        "missing_col": "uint64"
    }
    
    missing, extra, mismatches = compare_schemas(actual, expected)
    
    assert missing == ["missing_col"]
    assert set(extra) == {"col3", "extra_col"}
    assert len(mismatches) == 0  # col1 should match (double -> float64)


def test_types_match():
    """Test type matching logic."""
    assert _types_match("double", "float64") == True
    assert _types_match("string", "string") == True
    assert _types_match("bool", "bool") == True
    assert _types_match("uint64", "uint64") == True
    assert _types_match("date32[day]", "date32") == True
    assert _types_match("timestamp[ns]", "timestamp[ns]") == True
    
    # Mismatches
    assert _types_match("double", "string") == False
    assert _types_match("int64", "float64") == False


def test_validate_parquet_schema_missing_column():
    """Test schema validation with missing column."""
    # Mock PyArrow table with missing column
    mock_field1 = MagicMock()
    mock_field1.name = "col1"
    mock_field1.type = "double"
    
    mock_schema = MagicMock()
    mock_schema.__iter__ = lambda self: iter([mock_field1])
    
    mock_table = MagicMock()
    mock_table.schema = mock_schema
    
    # Create temporary spec file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {
            "test_dataset": {
                "col1": "float64",
                "missing_col": "string"  # This column is missing from actual
            }
        }
        json.dump(schema_spec, f)
        spec_file = f.name
    
    # Create temporary parquet file (just for file existence check)
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_file = f.name
    
    try:
        with patch('validate_schemas.pq.read_table', return_value=mock_table):
            with pytest.raises(SchemaValidationError) as exc_info:
                validate_parquet_schema(parquet_file, spec_file, "test_dataset")
            
            assert "Missing columns: missing_col" in str(exc_info.value)
    finally:
        os.unlink(spec_file)
        os.unlink(parquet_file)


def test_validate_parquet_schema_extra_column():
    """Test schema validation with extra column."""
    # Mock PyArrow table with extra column
    mock_field1 = MagicMock()
    mock_field1.name = "col1"
    mock_field1.type = "double"
    
    mock_field2 = MagicMock()
    mock_field2.name = "extra_col"
    mock_field2.type = "string"
    
    mock_schema = MagicMock()
    mock_schema.__iter__ = lambda self: iter([mock_field1, mock_field2])
    
    mock_table = MagicMock()
    mock_table.schema = mock_schema
    
    # Create temporary spec file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {
            "test_dataset": {
                "col1": "float64"  # extra_col is not expected
            }
        }
        json.dump(schema_spec, f)
        spec_file = f.name
    
    # Create temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_file = f.name
    
    try:
        with patch('validate_schemas.pq.read_table', return_value=mock_table):
            with pytest.raises(SchemaValidationError) as exc_info:
                validate_parquet_schema(parquet_file, spec_file, "test_dataset")
            
            assert "Extra columns: extra_col" in str(exc_info.value)
    finally:
        os.unlink(spec_file)
        os.unlink(parquet_file)


def test_validate_parquet_schema_type_mismatch():
    """Test schema validation with type mismatch."""
    # Mock PyArrow table with type mismatch
    mock_field1 = MagicMock()
    mock_field1.name = "col1"
    mock_field1.type = "string"  # Expected float64
    
    mock_schema = MagicMock()
    mock_schema.__iter__ = lambda self: iter([mock_field1])
    
    mock_table = MagicMock()
    mock_table.schema = mock_schema
    
    # Create temporary spec file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {
            "test_dataset": {
                "col1": "float64"  # Actual is string
            }
        }
        json.dump(schema_spec, f)
        spec_file = f.name
    
    # Create temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_file = f.name
    
    try:
        with patch('validate_schemas.pq.read_table', return_value=mock_table):
            with pytest.raises(SchemaValidationError) as exc_info:
                validate_parquet_schema(parquet_file, spec_file, "test_dataset")
            
            assert "Type mismatches: col1: expected float64, got string" in str(exc_info.value)
    finally:
        os.unlink(spec_file)
        os.unlink(parquet_file)


def test_validate_parquet_schema_success():
    """Test successful schema validation."""
    # Mock PyArrow table with matching schema
    mock_field1 = MagicMock()
    mock_field1.name = "col1"
    mock_field1.type = "double"
    
    mock_field2 = MagicMock()
    mock_field2.name = "col2"
    mock_field2.type = "string"
    
    mock_schema = MagicMock()
    mock_schema.__iter__ = lambda self: iter([mock_field1, mock_field2])
    
    mock_table = MagicMock()
    mock_table.schema = mock_schema
    
    # Create temporary spec file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_spec = {
            "test_dataset": {
                "col1": "float64",
                "col2": "string"
            }
        }
        json.dump(schema_spec, f)
        spec_file = f.name
    
    # Create temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_file = f.name
    
    try:
        with patch('validate_schemas.pq.read_table', return_value=mock_table):
            # Should not raise an exception
            validate_parquet_schema(parquet_file, spec_file, "test_dataset")
    finally:
        os.unlink(spec_file)
        os.unlink(parquet_file)


if __name__ == "__main__":
    pytest.main([__file__])