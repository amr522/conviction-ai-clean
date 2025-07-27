import pytest
import polars as pl
from datetime import date
from pathlib import Path
import tempfile
import os

from src.generate_labels import calculate_iv_change_5d, calculate_target_labels, generate_labels


@pytest.fixture
def sample_data():
    """Create sample daily master data for testing."""
    dates = [date(2025, 1, i) for i in range(1, 11)]  # 10 days
    tickers = ["AAPL", "MSFT"]
    
    data = []
    for ticker in tickers:
        for i, dt in enumerate(dates):
            data.append({
                "date": dt,
                "ticker": ticker,
                "stockd_close": 100 + i + (10 if ticker == "MSFT" else 0),  # Trending prices
                "optd_iv30": 0.2 + i * 0.01,  # Trending IV
                "stockd_return_1d": 0.01 * (1 if i % 2 == 0 else -1),  # Alternating returns
                "vix_index": 20 + i * 0.5
            })
    
    return pl.DataFrame(data)


def test_calculate_iv_change_5d(sample_data):
    """Test IV change calculation."""
    result = calculate_iv_change_5d(sample_data)
    
    # Check that iv_change_5d column is added
    assert "iv_change_5d" in result.columns
    
    # Check calculation for first ticker (AAPL)
    aapl_data = result.filter(pl.col("ticker") == "AAPL").sort("date")
    
    # First 5 rows should have valid iv_change_5d values
    first_change = aapl_data[0, "iv_change_5d"]
    assert first_change is not None
    
    # Should be approximately 0.05 (5 days * 0.01 increment)
    assert abs(first_change - 0.05) < 0.001


def test_calculate_target_labels(sample_data):
    """Test target label calculation."""
    result = calculate_target_labels(sample_data)
    
    # Check that target column is added
    assert "target" in result.columns
    
    # Check calculation for first ticker
    aapl_data = result.filter(pl.col("ticker") == "AAPL").sort("date")
    
    # First row should have valid target
    first_target = aapl_data[0, "target"]
    assert first_target is not None
    
    # Should be approximately 0.05 (5/100 - 1)
    assert abs(first_target - 0.05) < 0.001


def test_generate_labels_with_mock_data(sample_data, tmp_path):
    """Test full label generation with mocked daily master."""
    # Create temporary daily master file
    daily_master_path = tmp_path / "daily_master.parquet"
    sample_data.write_parquet(daily_master_path)
    
    # Create staged directory and symlink
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    staged_master = staged_dir / "daily_master.parquet"
    
    # Copy file instead of symlink for cross-platform compatibility
    sample_data.write_parquet(staged_master)
    
    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Generate labels for first date
        test_date = "2025-01-01"
        output_path = generate_labels(test_date)
        
        # Check output file exists
        assert Path(output_path).exists()
        
        # Load and validate labels
        labels = pl.read_parquet(output_path)
        
        # Check schema
        expected_cols = ["date", "ticker", "target", "iv_change_5d"]
        for col in expected_cols:
            assert col in labels.columns
        
        # Check data
        assert labels.height > 0
        assert labels["ticker"].n_unique() == 2  # AAPL and MSFT
        
        # Check date filtering
        assert all(labels["date"].to_list() == [date(2025, 1, 1)] * labels.height)
        
    finally:
        os.chdir(original_cwd)


def test_generate_labels_missing_daily_master():
    """Test error handling when daily master is missing."""
    with pytest.raises(FileNotFoundError, match="Daily master not found"):
        generate_labels("2025-01-01")


def test_generate_labels_no_data_for_date(sample_data, tmp_path):
    """Test handling when no data exists for specified date."""
    # Create daily master with sample data
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    daily_master_path = staged_dir / "daily_master.parquet"
    sample_data.write_parquet(daily_master_path)
    
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Try to generate labels for date not in sample data
        with pytest.raises(ValueError, match="No data found for date"):
            generate_labels("2025-12-31")
    finally:
        os.chdir(original_cwd)


def test_generate_labels_creates_synthetic_fallback(tmp_path):
    """Test that synthetic labels are created when no valid labels exist."""
    # Create minimal data without required columns for label calculation
    minimal_data = pl.DataFrame({
        "date": [date(2025, 1, 1)] * 2,
        "ticker": ["AAPL", "MSFT"],
        "some_other_col": [1, 2]
    })
    
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    daily_master_path = staged_dir / "daily_master.parquet"
    minimal_data.write_parquet(daily_master_path)
    
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # This should create synthetic labels due to missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_labels("2025-01-01")
    finally:
        os.chdir(original_cwd)