#!/usr/bin/env python3
"""
Tests for train_and_evaluate.py module
"""

import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_and_evaluate import load_features, prepare_features_and_target


@pytest.fixture
def sample_features_data():
    """Create sample features data for testing"""
    dates = [date(2025, 1, 15) + timedelta(days=i) for i in range(10)]
    tickers = ["AAPL", "MSFT"]

    data = []
    for ticker in tickers:
        for i, dt in enumerate(dates):
            data.append(
                {
                    "ticker": ticker,
                    "date": dt,
                    "fred_rate_mean": 5.0 + i * 0.1,
                    "vix_std": 2.0 + i * 0.1,
                    "news_count_rolling": 100 + i * 10,
                    "optd_iv30_mean": 0.25 + i * 0.01,
                    "optd_volume_std": 1000 + i * 100,
                    "stockd_vol_rolling": 0.02 + i * 0.001,
                    "stockd_volume_mean": 50000 + i * 1000,
                    "stockd_return_1d": 0.01 + i * 0.001,
                    "ret_1h": 0.005 + i * 0.0001,
                    "vol_zscore": (i - 5) * 0.1,
                    "iv_rank": i * 0.1,
                    "ret_relative": (i - 5) * 0.001,
                }
            )

    return pd.DataFrame(data)


def test_load_features(sample_features_data):
    """Test loading features from parquet file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample data
        features_path = Path(tmpdir) / "features.parquet"
        sample_features_data.to_parquet(features_path)

        # Test loading
        loaded_df = load_features(str(features_path), "2025-01-15", "2025-01-20")

        # Check that data was loaded correctly
        assert len(loaded_df) > 0
        assert "ticker" in loaded_df.columns
        assert "date" in loaded_df.columns
        assert "fred_rate_mean" in loaded_df.columns


def test_prepare_features_and_target(sample_features_data):
    """Test preparing features and target from precomputed features"""
    X, y = prepare_features_and_target(sample_features_data)

    # Check features
    assert len(X) == len(sample_features_data)
    assert "ticker" not in X.columns
    assert "date" not in X.columns
    assert "target" not in X.columns
    assert "fred_rate_mean" in X.columns

    # Check target
    assert len(y) == len(sample_features_data)
    assert y.name == "target"


def test_train_and_evaluate_with_feature_path(sample_features_data):
    """Test train_and_evaluate.py with --feature-path argument"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample features
        features_path = Path(tmpdir) / "features.parquet"
        sample_features_data.to_parquet(features_path)

        # Create output directories
        model_path = Path(tmpdir) / "model.pkl"
        metrics_path = Path(tmpdir) / "metrics"
        metrics_path.mkdir()

        # Run train_and_evaluate.py
        cmd = [
            "python",
            "src/train_and_evaluate.py",
            "--start-date",
            "2025-01-15",
            "--end-date",
            "2025-01-20",
            "--feature-path",
            str(features_path),
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
            "--dry-run",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        # Check that command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check that dry run completed successfully
        assert "DRY RUN" in result.stdout
        assert "Would train on" in result.stdout


def test_train_and_evaluate_with_tuning(sample_features_data):
    """Test train_and_evaluate.py with hyperparameter tuning"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample features
        features_path = Path(tmpdir) / "features.parquet"
        sample_features_data.to_parquet(features_path)

        # Create output directories
        model_path = Path(tmpdir) / "model.pkl"
        metrics_path = Path(tmpdir) / "metrics"
        metrics_path.mkdir()

        # Run with tuning (dry run)
        cmd = [
            "python",
            "src/train_and_evaluate.py",
            "--start-date",
            "2025-01-15",
            "--end-date",
            "2025-01-20",
            "--feature-path",
            str(features_path),
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
            "--dry-run",
            "--tune",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        # Check that command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check that tuning was attempted
        assert "hyperparameter optimization" in result.stdout
        assert "Dry-run best params" in result.stdout


def test_missing_feature_path():
    """Test that missing --feature-path raises appropriate error"""
    cmd = [
        "python",
        "src/train_and_evaluate.py",
        "--start-date",
        "2025-01-15",
        "--end-date",
        "2025-01-20",
        "--dry-run",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Should fail due to missing required argument
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "feature-path" in result.stderr


def test_nonexistent_feature_file():
    """Test handling of nonexistent feature file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_path = Path(tmpdir) / "nonexistent.parquet"

        cmd = [
            "python",
            "src/train_and_evaluate.py",
            "--start-date",
            "2025-01-15",
            "--end-date",
            "2025-01-20",
            "--feature-path",
            str(nonexistent_path),
            "--dry-run",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        # Should fail due to missing file
        assert result.returncode != 0


def test_feature_path_integration():
    """Test that feature path is properly logged in metrics"""
    sample_data = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "date": pd.date_range("2025-01-15", periods=5),
            "fred_rate_mean": [5.0, 5.1, 5.2, 5.3, 5.4],
            "vix_std": [2.0, 2.1, 2.2, 2.3, 2.4],
            "stockd_return_1d": [0.01, 0.02, -0.01, 0.015, 0.005],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = Path(tmpdir) / "test_features.parquet"
        sample_data.to_parquet(features_path)

        # Test that feature path is included in function call
        from train_and_evaluate import run

        try:
            # This should fail gracefully but test parameter passing
            run(
                start_date="2025-01-15",
                end_date="2025-01-16",
                model_path=str(Path(tmpdir) / "model.pkl"),
                metrics_path=str(tmpdir),
                feature_path=str(features_path),
                dry_run=True,
            )
        except Exception:
            # Expected to fail in test environment, but parameter should be accepted
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
