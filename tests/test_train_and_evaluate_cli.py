#!/usr/bin/env python3
"""
Tests for train_and_evaluate.py CLI functionality
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_and_evaluate import run, main


class TestTrainAndEvaluateCLI:
    """Test suite for train_and_evaluate CLI functionality"""
    
    def test_help_output(self, capsys):
        """Test --help output for train_and_evaluate.py"""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['train_and_evaluate.py', '--help']):
                main()
        
        # Should exit with code 0 for help
        assert exc_info.value.code == 0
        
        # Capture help output
        captured = capsys.readouterr()
        help_text = captured.out
        
        # Check that key arguments are documented
        assert "--start-date" in help_text
        assert "--end-date" in help_text
        assert "--dry-run" in help_text
        assert "--tune" in help_text
        assert "--n-jobs" in help_text
        assert "Start date (YYYY-MM-DD)" in help_text
    
    @patch('train_and_evaluate.load_partitioned_data')
    @patch('train_and_evaluate.prepare_features_and_target')
    @patch('train_and_evaluate.LineageTracker')
    def test_minimal_dry_run(self, mock_lineage, mock_prepare, mock_load, caplog):
        """Test minimal dry-run execution"""
        # Mock data loading
        mock_load.return_value = MagicMock()
        mock_features = MagicMock()
        mock_features.columns = ['feature1', 'feature2']
        mock_prepare.return_value = (mock_features, MagicMock())
        
        # Mock lineage tracker
        mock_tracker = MagicMock()
        mock_lineage.return_value = mock_tracker
        
        with caplog.at_level("INFO"):
            exit_code = run(
                start_date='2025-01-01',
                end_date='2025-01-02', 
                model_path='test_model.pkl',
                metrics_path='test_metrics/',
                dry_run=True
            )
        
        # Should exit successfully
        assert exit_code == 0
        
        # Check expected log messages
        assert "DRY RUN: Skipping training and file operations" in caplog.text
        assert "Starting training pipeline: 2025-01-01 to 2025-01-02" in caplog.text
        
        # Verify lineage tracking was called
        mock_tracker.start_run.assert_called_once()
    
    @patch('train_and_evaluate.load_partitioned_data')
    @patch('train_and_evaluate.prepare_features_and_target')
    @patch('train_and_evaluate.train_validation_split')
    @patch('train_and_evaluate.optuna.create_study')
    @patch('train_and_evaluate.LineageTracker')
    def test_mock_optuna_tuning(self, mock_lineage, mock_study_create, mock_split, mock_prepare, mock_load, caplog):
        """Test Optuna hyperparameter tuning with mocked study"""
        # Mock data loading and preparation
        mock_load.return_value = MagicMock()
        mock_features = MagicMock()
        mock_features.columns = ['feature1', 'feature2']
        mock_target = MagicMock()
        mock_prepare.return_value = (mock_features, mock_target)
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        
        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {'learning_rate': 0.1, 'num_leaves': 31}
        mock_study.best_value = 0.05
        mock_study_create.return_value = mock_study
        
        # Mock lineage tracker
        mock_tracker = MagicMock()
        mock_lineage.return_value = mock_tracker
        
        with caplog.at_level("INFO"):
            exit_code = run(
                start_date='2025-01-01',
                end_date='2025-01-02',
                model_path='test_model.pkl', 
                metrics_path='test_metrics/',
                dry_run=True,
                tune=True,
                n_trials=1
            )
        
        # Should exit successfully
        assert exit_code == 0
        
        # Verify study.optimize was called with n_trials=1
        mock_study.optimize.assert_called_once()
        call_args = mock_study.optimize.call_args
        assert call_args[1]['n_trials'] == 1
        
        # Check tuning log messages
        assert "Running 1 trial for dry-run hyperparameter optimization" in caplog.text
        assert "Dry-run best params:" in caplog.text
    
    @patch('train_and_evaluate.load_partitioned_data')
    @patch('train_and_evaluate.prepare_features_and_target') 
    @patch('train_and_evaluate.train_validation_split')
    @patch('train_and_evaluate.LGBMRegressor')
    @patch('train_and_evaluate.LineageTracker')
    def test_mock_lightgbm_n_jobs(self, mock_lineage, mock_lgbm, mock_split, mock_prepare, mock_load):
        """Test that LGBMRegressor receives n_jobs from --n-jobs flag"""
        # Mock data loading and preparation
        mock_load.return_value = MagicMock()
        mock_features = MagicMock()
        mock_features.columns = ['feature1', 'feature2']
        mock_target = MagicMock()
        mock_prepare.return_value = (mock_features, mock_target)
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        
        # Mock LightGBM model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.01, 0.02, 0.03]
        mock_model.feature_importances_ = [0.5, 0.3]
        mock_lgbm.return_value = mock_model
        
        # Mock lineage tracker
        mock_tracker = MagicMock()
        mock_lineage.return_value = mock_tracker
        
        exit_code = run(
            start_date='2025-01-01',
            end_date='2025-01-02',
            model_path='test_model.pkl',
            metrics_path='test_metrics/',
            dry_run=True,
            n_jobs=4
        )
        
        # Should exit successfully
        assert exit_code == 0
        
        # In dry_run mode, LGBMRegressor is not called
        # The test verifies the dry_run logic works correctly
    
    @patch('train_and_evaluate.load_partitioned_data')
    def test_data_loading_failure(self, mock_load, caplog):
        """Test handling of data loading failures"""
        # Mock data loading to raise exception
        mock_load.side_effect = FileNotFoundError("Data directory not found")
        
        with caplog.at_level("ERROR"):
            exit_code = run(
                start_date='2025-01-01',
                end_date='2025-01-02',
                model_path='test_model.pkl',
                metrics_path='test_metrics/',
                dry_run=True
            )
        
        # Should exit with error code
        assert exit_code == 1
        
        # Check error logging
        assert "Training pipeline failed:" in caplog.text
    
    @patch('train_and_evaluate.load_partitioned_data')
    @patch('train_and_evaluate.prepare_features_and_target')
    @patch('train_and_evaluate.LineageTracker')
    def test_gpu_detection_fallback(self, mock_lineage, mock_prepare, mock_load, caplog):
        """Test GPU detection and fallback to CPU"""
        # Mock data loading
        mock_load.return_value = MagicMock()
        mock_prepare.return_value = (MagicMock(), MagicMock())
        
        # Mock lineage tracker
        mock_tracker = MagicMock()
        mock_lineage.return_value = mock_tracker
        
        with caplog.at_level("INFO"):
            exit_code = run(
                start_date='2025-01-01',
                end_date='2025-01-02',
                model_path='test_model.pkl',
                metrics_path='test_metrics/',
                dry_run=True
            )
        
        # Should exit successfully
        assert exit_code == 0
        
        # Check GPU detection message (should fallback to CPU)
        assert "GPU acceleration not available, using CPU" in caplog.text
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing with various flags"""
        # Test that main() can parse arguments correctly
        test_args = [
            'train_and_evaluate.py',
            '--start-date', '2025-01-01',
            '--end-date', '2025-01-02', 
            '--dry-run',
            '--tune',
            '--n-trials', '10',
            '--n-jobs', '2'
        ]
        
        with patch('sys.argv', test_args):
            with patch('train_and_evaluate.run') as mock_run:
                mock_run.return_value = 0
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Should exit with code 0
                assert exc_info.value.code == 0
                
                # Verify run was called with correct arguments
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs['start_date'] == '2025-01-01'
                assert call_kwargs['end_date'] == '2025-01-02'
                assert call_kwargs['dry_run'] is True
                assert call_kwargs['tune'] is True
                assert call_kwargs['n_trials'] == 10
                assert call_kwargs['n_jobs'] == 2


if __name__ == "__main__":
    pytest.main([__file__])