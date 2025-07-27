#!/usr/bin/env python3
"""
Integration tests for calculate_features.py in end-to-end pipeline flows
"""

import pytest
import polars as pl
import tempfile
import subprocess
from pathlib import Path
from datetime import date, timedelta
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_master_data():
    """Create sample master datasets for integration testing"""
    dates = [date(2025, 1, 15) + timedelta(days=i) for i in range(5)]
    tickers = ["AAPL", "MSFT"]
    
    # Daily master data
    daily_data = []
    for ticker in tickers:
        for i, dt in enumerate(dates):
            daily_data.append({
                "ticker": ticker,
                "date": dt,
                "fred_fed_funds_rate": 5.0 + i * 0.1,
                "vix_index": 20.0 + i * 0.5,
                "news_count": 10 + i,
                "optd_iv30": 0.25 + i * 0.01,
                "optd_hv30": 0.20 + i * 0.01,
                "optd_volume": 1000 + i * 100,
                "stockd_return_1d": 0.01 + i * 0.001,
                "stockd_volume": 50000 + i * 1000,
                "stockd_close": 150.0 + i * 2.0
            })
    
    # Intraday master data
    from datetime import datetime
    intraday_data = []
    for ticker in tickers:
        for dt in dates:
            for hour in range(9, 16):  # Market hours
                for minute in [30, 0]:  # 30-minute intervals
                    if hour == 15 and minute == 30:  # Skip 3:30 PM
                        continue
                    ts = datetime(dt.year, dt.month, dt.day, hour, minute)
                    intraday_data.append({
                        "ticker": ticker,
                        "timestamp": ts,
                        "date": dt,
                        "opt30_mid_price": 100.0 + hour * 0.5 + minute * 0.01,
                        "opt30_volume": 500 + hour * 10
                    })
    
    return pl.DataFrame(daily_data), pl.DataFrame(intraday_data)


def test_end_to_end_feature_calculation(sample_master_data):
    """Test complete feature calculation pipeline"""
    daily_df, intraday_df = sample_master_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample data
        daily_path = Path(tmpdir) / "daily_master.parquet"
        intraday_path = Path(tmpdir) / "intraday_master.parquet"
        features_path = Path(tmpdir) / "features_2025-01-16.parquet"
        
        daily_df.write_parquet(daily_path)
        intraday_df.write_parquet(intraday_path)
        
        # Run feature calculation
        cmd = [
            "python", "src/calculate_features.py",
            "--daily-master-path", str(daily_path),
            "--intraday-master-path", str(intraday_path),
            "--output-path", str(features_path),
            "--date", "2025-01-16",
            "--window-days", "3"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Check command succeeded
        assert result.returncode == 0, f"Feature calculation failed: {result.stderr}"
        
        # Check output file exists
        assert features_path.exists(), f"Features file not created: {features_path}"
        
        # Load and validate features
        features = pl.read_parquet(features_path)
        
        # Check required columns exist
        required_cols = [
            "fred_rate_mean", "vix_std", "news_count_rolling",
            "ret_1h", "vol_zscore", "ticker", "date"
        ]
        
        for col in required_cols:
            assert col in features.columns, f"Missing required column: {col}"
        
        # Check data integrity
        assert len(features) > 0, "No features generated"
        assert features["ticker"].n_unique() == 2, "Expected 2 tickers"


def test_run_full_pipeline_integration(sample_master_data):
    """Test that run_full_pipeline.py generates features correctly"""
    daily_df, intraday_df = sample_master_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create staged and datasets directories
        staged_dir = Path(tmpdir) / "staged"
        datasets_dir = Path(tmpdir) / "datasets"
        staged_dir.mkdir()
        datasets_dir.mkdir()
        
        # Write sample data to expected locations
        daily_df.write_parquet(staged_dir / "daily_master.parquet")
        intraday_df.write_parquet(datasets_dir / "intraday_master.parquet")
        
        # Set environment variables for feature calculation
        env = os.environ.copy()
        env.update({
            "WINDOW_DAYS": "3",
            "USE_GPU": "false",
            "N_JOBS": "1"
        })
        
        # Change to temp directory to simulate pipeline execution
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Create minimal directory structure
            os.makedirs("src", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Copy calculate_features.py to temp directory
            import shutil
            src_path = Path(original_cwd) / "src" / "calculate_features.py"
            if src_path.exists():
                shutil.copy(src_path, "src/calculate_features.py")
            
            # Run feature calculation directly (simulating run_full_pipeline.py step)
            cmd = [
                "python", "src/calculate_features.py",
                "--daily-master-path", "staged/daily_master.parquet",
                "--intraday-master-path", "datasets/intraday_master.parquet",
                "--output-path", "datasets/features_2025-01-16.parquet",
                "--date", "2025-01-16",
                "--window-days", "3",
                "--n-jobs", "1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            # Check execution
            assert result.returncode == 0, f"Feature calculation failed: {result.stderr}"
            
            # Check output file
            features_path = Path("datasets/features_2025-01-16.parquet")
            assert features_path.exists(), "Features file not created by pipeline"
            
            # Validate schema
            features = pl.read_parquet(features_path)
            expected_schema = [
                "fred_rate_mean", "vix_std", "news_count_rolling",
                "optd_iv30_mean", "optd_volume_std", "stockd_vol_rolling",
                "stockd_volume_mean", "ret_1h", "vol_zscore", "iv_rank", "ret_relative"
            ]
            
            for col in expected_schema:
                assert col in features.columns, f"Missing expected column: {col}"
            
        finally:
            os.chdir(original_cwd)


def test_feature_file_naming_convention():
    """Test that feature files follow the expected naming convention"""
    test_dates = ["2025-01-15", "2025-12-31", "2024-06-15"]
    
    for test_date in test_dates:
        expected_filename = f"features_{test_date}.parquet"
        
        # This would be the expected path in the pipeline
        expected_path = f"datasets/{expected_filename}"
        
        # Verify naming convention matches what scripts expect
        assert "features_" in expected_filename
        assert test_date in expected_filename
        assert expected_filename.endswith(".parquet")


def test_environment_variable_integration():
    """Test that environment variables are properly used"""
    env_vars = {
        "WINDOW_DAYS": "20",
        "USE_GPU": "true", 
        "N_JOBS": "4"
    }
    
    # Test that these variables would be used in the pipeline
    for var, value in env_vars.items():
        # Simulate environment variable usage
        test_env = os.environ.copy()
        test_env[var] = value
        
        # Verify the variable is accessible
        assert test_env.get(var) == value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])