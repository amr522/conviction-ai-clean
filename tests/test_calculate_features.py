#!/usr/bin/env python3
"""
Tests for calculate_features.py module
"""

import pytest
import polars as pl
import tempfile
from pathlib import Path
from datetime import date, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calculate_features import calculate_rolling_features, calculate_intraday_features, calculate_cross_sectional_features, parse_date_range


@pytest.fixture
def sample_daily_data():
    """Create sample daily master data"""
    dates = [date(2025, 1, 15) + timedelta(days=i) for i in range(10)]
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for ticker in tickers:
        for i, dt in enumerate(dates):
            data.append({
                "ticker": ticker,
                "date": dt,
                "fred_fed_funds_rate": 5.0 + i * 0.1,
                "vix_index": 20.0 + i * 0.5,
                "news_count": 10 + i,
                "optd_iv30": 0.25 + i * 0.01,
                "optd_volume": 1000 + i * 100,
                "stockd_return_1d": 0.01 + i * 0.001,
                "stockd_volume": 50000 + i * 1000
            })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_intraday_data():
    """Create sample intraday master data"""
    from datetime import datetime
    
    timestamps = [datetime(2025, 1, 15, 9, 30) + timedelta(minutes=30*i) for i in range(8)]
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for ticker in tickers:
        for i, ts in enumerate(timestamps):
            data.append({
                "ticker": ticker,
                "timestamp": ts,
                "date": ts.date(),
                "opt30_mid_price": 100.0 + i * 0.5,
                "opt30_volume": 500 + i * 50
            })
    
    return pl.DataFrame(data)


def test_parse_date_range():
    """Test date range parsing"""
    # Single date
    start, end = parse_date_range("2025-01-15")
    assert start == date(2025, 1, 15)
    assert end == date(2025, 1, 15)
    
    # Date range
    start, end = parse_date_range("2025-01-15,2025-01-20")
    assert start == date(2025, 1, 15)
    assert end == date(2025, 1, 20)


def test_calculate_rolling_features(sample_daily_data):
    """Test rolling feature calculations"""
    ticker_data = sample_daily_data.filter(pl.col("ticker") == "AAPL")
    result = calculate_rolling_features(ticker_data, window=5)
    
    # Check that rolling columns are created
    expected_cols = ["fred_rate_mean", "vix_std", "news_count_rolling", 
                    "optd_iv30_mean", "optd_volume_std", "stockd_vol_rolling", "stockd_volume_mean"]
    
    for col in expected_cols:
        assert col in result.columns
    
    # Check that rolling calculations work (non-null after window)
    assert result["fred_rate_mean"].drop_nulls().len() > 0


def test_calculate_intraday_features(sample_intraday_data):
    """Test intraday feature calculations"""
    result = calculate_intraday_features(sample_intraday_data)
    
    # Check required columns
    assert "ret_1h" in result.columns
    assert "ticker" in result.columns
    assert "timestamp" in result.columns
    
    # Check that we have data for all tickers
    tickers = result["ticker"].unique().sort()
    expected_tickers = ["AAPL", "GOOGL", "MSFT"]
    assert tickers.to_list() == expected_tickers


def test_calculate_cross_sectional_features(sample_daily_data):
    """Test cross-sectional feature calculations"""
    result = calculate_cross_sectional_features(sample_daily_data)
    
    # Check that cross-sectional columns are created
    expected_cols = ["vol_zscore", "iv_rank", "ret_relative"]
    
    for col in expected_cols:
        assert col in result.columns
    
    # Check that z-scores are calculated (should have mean ~0 per date)
    assert result["vol_zscore"].drop_nulls().len() > 0


def test_full_feature_calculation_integration(sample_daily_data, sample_intraday_data):
    """Test full feature calculation pipeline"""
    import subprocess
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample data
        daily_path = Path(tmpdir) / "daily.parquet"
        intraday_path = Path(tmpdir) / "intraday.parquet"
        output_path = Path(tmpdir) / "features.parquet"
        
        sample_daily_data.write_parquet(daily_path)
        sample_intraday_data.write_parquet(intraday_path)
        
        # Run calculate_features.py
        cmd = [
            "python", "src/calculate_features.py",
            "--daily-master-path", str(daily_path),
            "--intraday-master-path", str(intraday_path),
            "--output-path", str(output_path),
            "--date", "2025-01-15,2025-01-20",
            "--window-days", "5"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Check that command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Check output file exists
        assert output_path.exists()
        
        # Load and validate output
        features = pl.read_parquet(output_path)
        
        # Check key feature columns exist
        expected_features = ["fred_rate_mean", "vix_std", "news_count_rolling", "ret_1h", "vol_zscore"]
        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"
        
        # Check we have data
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__])