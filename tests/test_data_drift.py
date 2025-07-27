#!/usr/bin/env python3
"""
Test data drift monitoring functionality
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_drift_monitor import monitor_data_drift, generate_drift_report

class TestDataDriftMonitoring:
    
    def setup_method(self):
        """Setup test data and temporary directories"""
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = os.path.join(self.temp_dir, "datasets", "processed")
        self.metrics_dir = os.path.join(self.temp_dir, "metrics")
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        
        # Create directories
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create synthetic reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.exponential(1, 1000),
            'target': np.random.normal(0.02, 0.01, 1000)
        })
        
        # Save reference data
        ref_path = os.path.join(self.datasets_dir, "reference.parquet")
        self.reference_data.to_parquet(ref_path)
    
    def test_no_drift_detection(self):
        """Test that identical data shows no drift"""
        # Create identical current data
        current_data = self.reference_data.copy()
        current_path = os.path.join(self.datasets_dir, "2025-01-01.parquet")
        current_data.to_parquet(current_path)
        
        # Change working directory for test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Monitor drift
            drift_detected, report_path = monitor_data_drift("2025-01-01")
            
            # Should detect no drift
            assert drift_detected is False
            assert report_path is not None
            assert os.path.exists(report_path)
            
            # Check log file
            log_path = os.path.join(self.logs_dir, "evidently_log.txt")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    assert "Drift detected: False" in log_content
        
        finally:
            os.chdir(original_cwd)
    
    def test_drift_detection(self):
        """Test that different data shows drift"""
        # Create drifted current data (shifted distribution)
        np.random.seed(123)  # Different seed for different data
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, 1000),  # Shifted mean
            'feature_2': np.random.normal(10, 2, 1000),  # Shifted mean
            'feature_3': np.random.exponential(2, 1000),  # Different scale
            'target': np.random.normal(0.05, 0.02, 1000)  # Different distribution
        })
        
        current_path = os.path.join(self.datasets_dir, "2025-01-02.parquet")
        current_data.to_parquet(current_path)
        
        # Change working directory for test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Monitor drift
            drift_detected, report_path = monitor_data_drift("2025-01-02")
            
            # Should detect drift
            assert drift_detected is True
            assert report_path is not None
            assert os.path.exists(report_path)
            
            # Check log file
            log_path = os.path.join(self.logs_dir, "evidently_log.txt")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    assert "Drift detected: True" in log_content
        
        finally:
            os.chdir(original_cwd)
    
    def test_missing_data_handling(self):
        """Test handling of missing data files"""
        # Change working directory for test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Try to monitor drift for non-existent date
            drift_detected, report_path = monitor_data_drift("2025-12-31")
            
            # Should handle gracefully
            assert drift_detected is False
            assert report_path is None
        
        finally:
            os.chdir(original_cwd)
    
    def test_generate_drift_report(self):
        """Test drift report generation directly"""
        # Create slightly different data
        current_data = self.reference_data.copy()
        current_data['feature_1'] += 0.1  # Small shift
        
        # Generate report
        report_path, drift_detected = generate_drift_report(
            current_data, self.reference_data, "2025-01-03", self.metrics_dir
        )
        
        # Check report was created
        assert os.path.exists(report_path)
        assert report_path.endswith("data_drift_report_2025-01-03.html")
        
        # Drift detection depends on Evidently's sensitivity
        assert isinstance(drift_detected, bool)

def test_smoke_training_with_drift():
    """Smoke test for training pipeline with drift monitoring"""
    # This would be run in CI to ensure the integration works
    # For now, just test imports work
    try:
        from src.train_and_evaluate import run
        from src.data_drift_monitor import monitor_data_drift
        assert True  # If imports work, basic integration is OK
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])