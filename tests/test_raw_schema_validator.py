#!/usr/bin/env python3
"""
Tests for raw schema validator
"""
import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.utils.raw_schema_validator import SchemaMismatchError, validate


def test_validate_happy_path():
    """Test successful validation with matching schema"""
    # Create test data
    test_data = pl.DataFrame({
        "ticker": ["AAPL220121C00150000", "MSFT220121P00140000"],
        "close": [5.0, 3.0],
        "volume": [1000, 500]
    })
    
    # Create test schema
    test_schema = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "close": {"type": "number"},
            "volume": {"type": "integer"}
        },
        "required": ["ticker", "close", "volume"]
    }
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as data_file:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as schema_file:
            # Write test files
            test_data.write_parquet(data_file.name)
            json.dump(test_schema, open(schema_file.name, 'w'))
            
            # Test validation
            result = validate(data_file.name, schema_file.name)
            assert result is True
            
            # Cleanup
            Path(data_file.name).unlink()
            Path(schema_file.name).unlink()


def test_validate_file_not_found():
    """Test FileNotFoundError when data file doesn't exist"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as schema_file:
        json.dump({"type": "object"}, open(schema_file.name, 'w'))
        
        with pytest.raises(FileNotFoundError):
            validate("nonexistent.parquet", schema_file.name)
            
        Path(schema_file.name).unlink()


def test_validate_schema_mismatch():
    """Test SchemaMismatchError when data doesn't match schema"""
    # Create test data with wrong types
    test_data = pl.DataFrame({
        "ticker": ["AAPL220121C00150000"],
        "close": ["not_a_number"],  # Should be number
        "volume": [1000]
    })
    
    # Create strict schema
    test_schema = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "close": {"type": "number"},
            "volume": {"type": "integer"}
        },
        "required": ["ticker", "close", "volume"]
    }
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as data_file:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as schema_file:
            # Write test files
            test_data.write_parquet(data_file.name)
            json.dump(test_schema, open(schema_file.name, 'w'))
            
            # Test validation should fail
            with pytest.raises(SchemaMismatchError):
                validate(data_file.name, schema_file.name)
            
            # Cleanup
            Path(data_file.name).unlink()
            Path(schema_file.name).unlink()


def test_validate_csv_format():
    """Test validation works with CSV files"""
    test_data = pl.DataFrame({
        "ticker": ["AAPL220121C00150000"],
        "close": [5.0],
        "volume": [1000]
    })
    
    test_schema = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "close": {"type": "number"},
            "volume": {"type": "integer"}
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as data_file:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as schema_file:
            # Write test files
            test_data.write_csv(data_file.name)
            json.dump(test_schema, open(schema_file.name, 'w'))
            
            # Test validation
            result = validate(data_file.name, schema_file.name)
            assert result is True
            
            # Cleanup
            Path(data_file.name).unlink()
            Path(schema_file.name).unlink()


def test_validate_missing_schema():
    """Test validation skips when schema file is missing"""
    test_data = pl.DataFrame({
        "ticker": ["AAPL220121C00150000"],
        "close": [5.0],
        "volume": [1000]
    })
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as data_file:
        test_data.write_parquet(data_file.name)
        
        # Test with non-existent schema
        result = validate(data_file.name, "nonexistent_schema.json")
        assert result is True  # Should skip validation and return True
        
        Path(data_file.name).unlink()