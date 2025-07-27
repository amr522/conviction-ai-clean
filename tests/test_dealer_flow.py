#!/usr/bin/env python3
"""
Tests for dealer flow feature computation
"""
import pytest
import polars as pl
from pathlib import Path
import tempfile
from src.feature_builder.dealer_flow import compute_gex_spx


def test_compute_gex_spx_happy_path():
    """Test normal GEX computation"""
    # Create test data
    test_data = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "gex_spx": [1000.0, 1200.0, 800.0]
    }).with_columns(pl.col("date").str.to_date())
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_input:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_output:
            # Write test CSV
            test_data.write_csv(tmp_input.name)
            
            # Run computation
            result = compute_gex_spx(tmp_input.name, tmp_output.name)
            
            # Verify results
            assert len(result) == 3
            assert "gex_spx_lag1" in result.columns
            assert result["gex_spx_lag1"][0] is None  # First row should be null
            assert result["gex_spx_lag1"][1] == 1000.0  # Lagged value
            
            # Cleanup
            Path(tmp_input.name).unlink()
            Path(tmp_output.name).unlink()


def test_compute_gex_spx_missing_days():
    """Test GEX computation with missing days (forward fill)"""
    test_data = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-04"],  # Missing 01-03
        "gex_spx": [1000.0, None, 800.0]  # Missing value
    }).with_columns(pl.col("date").str.to_date())
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_input:
        test_data.write_csv(tmp_input.name)
        
        result = compute_gex_spx(tmp_input.name, None)
        
        # Should forward-fill the None value
        assert result["gex_spx"][1] == 1000.0  # Forward filled
        
        Path(tmp_input.name).unlink()


def test_compute_gex_spx_missing_file():
    """Test behavior when input file doesn't exist"""
    result = compute_gex_spx("nonexistent.csv", None)
    
    # Should return empty DataFrame with correct schema
    assert len(result) == 0
    assert "date" in result.columns
    assert "gex_spx" in result.columns
    assert "gex_spx_lag1" in result.columns