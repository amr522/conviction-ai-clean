"""Tests for feature validation utility."""

import pytest
import tempfile
import polars as pl
from pathlib import Path
from src.validate_features import load_expected_features, validate_feature_table


def test_load_expected_features():
    """Test loading features from markdown file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Features\n")
        f.write("feature1\n")
        f.write("## Section\n")
        f.write("feature2\n")
        f.write("# Comment\n")
        f.write("feature3\n")
        f.flush()
        
        features = load_expected_features(f.name)
        assert features == ["feature1", "feature2", "feature3"]
        
        Path(f.name).unlink()


def test_validate_feature_table_success():
    """Test successful feature validation."""
    # Create test parquet file
    df = pl.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "feature3": [7, 8, 9]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.write_parquet(f.name)
        
        success, message = validate_feature_table(f.name, ["feature1", "feature2"])
        assert success
        assert "2 features validated successfully" in message
        
        Path(f.name).unlink()


def test_validate_feature_table_missing_features():
    """Test validation with missing features."""
    df = pl.DataFrame({
        "feature1": [1, 2, 3]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.write_parquet(f.name)
        
        success, message = validate_feature_table(f.name, ["feature1", "feature2"])
        assert not success
        assert "Missing features" in message
        assert "feature2" in message
        
        Path(f.name).unlink()


def test_validate_feature_table_null_values():
    """Test validation with null values."""
    df = pl.DataFrame({
        "feature1": [1, None, 3],
        "feature2": [4, 5, 6]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.write_parquet(f.name)
        
        success, message = validate_feature_table(f.name, ["feature1", "feature2"])
        assert not success
        assert "Features with null values" in message
        assert "feature1 (1 nulls)" in message
        
        Path(f.name).unlink()


def test_validate_feature_table_file_not_found():
    """Test validation with missing file."""
    success, message = validate_feature_table("nonexistent.parquet", ["feature1"])
    assert not success
    assert "Feature table not found" in message