#!/usr/bin/env python3
"""
Test auto-retrain Prefect flow
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flows.auto_retrain_flow import check_drift_status, send_drift_notification

class TestAutoRetrainFlow:
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Change to temp directory for tests
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
    
    def test_check_drift_status_no_drift(self):
        """Test drift status check when no drift detected"""
        # Create log file with no drift
        log_content = """
        2025-01-16 10:00:00 - INFO - Running drift analysis for 2025-01-16
        2025-01-16 10:00:01 - INFO - Drift detected: False for 2025-01-16
        """
        
        with open("logs/evidently_log.txt", "w") as f:
            f.write(log_content)
        
        # Mock get_run_logger
        with patch('flows.auto_retrain_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            drift_detected = check_drift_status("2025-01-16")
            assert drift_detected is False
    
    def test_check_drift_status_with_drift(self):
        """Test drift status check when drift is detected"""
        # Create log file with drift
        log_content = """
        2025-01-16 10:00:00 - INFO - Running drift analysis for 2025-01-16
        2025-01-16 10:00:01 - WARNING - Drift detected: True for 2025-01-16
        """
        
        with open("logs/evidently_log.txt", "w") as f:
            f.write(log_content)
        
        # Mock get_run_logger
        with patch('flows.auto_retrain_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            drift_detected = check_drift_status("2025-01-16")
            assert drift_detected is True
    
    def test_check_drift_status_missing_log(self):
        """Test drift status check when log file is missing"""
        # Mock get_run_logger
        with patch('flows.auto_retrain_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            drift_detected = check_drift_status("2025-01-16")
            assert drift_detected is False
    
    @patch('flows.auto_retrain_flow.shell_run_command')
    def test_send_drift_notification_no_drift(self, mock_shell):
        """Test notification sending when no drift detected"""
        mock_shell.return_value = MagicMock(return_code=0)
        
        with patch('flows.auto_retrain_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            send_drift_notification("2025-01-16", drift_detected=False)
            
            # Check that shell command was called with correct parameters
            mock_shell.assert_called_once()
            call_args = mock_shell.call_args[1]['command']
            assert "NO_DRIFT" in call_args
            assert "No data drift detected" in call_args
    
    @patch('flows.auto_retrain_flow.shell_run_command')
    def test_send_drift_notification_with_drift_success(self, mock_shell):
        """Test notification sending when drift detected and backfill successful"""
        mock_shell.return_value = MagicMock(return_code=0)
        
        with patch('flows.auto_retrain_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            send_drift_notification("2025-01-16", drift_detected=True, backfill_success=True)
            
            # Check that shell command was called with correct parameters
            mock_shell.assert_called_once()
            call_args = mock_shell.call_args[1]['command']
            assert "DRIFT_RESOLVED" in call_args
            assert "backfill completed successfully" in call_args

def test_flow_imports():
    """Test that flow imports work correctly"""
    try:
        from flows.auto_retrain_flow import auto_retrain_flow, run_etl_and_train, check_drift_status
        assert callable(auto_retrain_flow)
        assert callable(run_etl_and_train)
        assert callable(check_drift_status)
    except ImportError as e:
        pytest.fail(f"Flow import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])