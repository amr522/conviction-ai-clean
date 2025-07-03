#!/usr/bin/env python3
"""
Tests for HPO data source pinning functionality
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_hpo_launch import get_input_data_s3, validate_s3_uri


class TestDataSourcePinning(unittest.TestCase):
    """Test cases for data source pinning functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        self.valid_s3_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
        self.invalid_s3_uri = "invalid://not-s3-uri"
        self.nonexistent_s3_uri = "s3://nonexistent-bucket/nonexistent/path.csv"
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_get_input_data_s3_cli_precedence(self):
        """Test CLI argument takes highest precedence"""
        cli_arg = "s3://cli-bucket/cli-data/"
        result = get_input_data_s3(cli_arg)
        self.assertEqual(result, cli_arg)
    
    def test_get_input_data_s3_from_environment(self):
        """Test getting pinned dataset from environment variable"""
        with patch.dict(os.environ, {'PINNED_DATA_S3': self.valid_s3_uri}):
            result = get_input_data_s3()
            self.assertEqual(result, self.valid_s3_uri)
    
    def test_get_input_data_s3_from_file(self):
        """Test getting pinned dataset from file when env var not set"""
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(self.valid_s3_uri)
        
        with patch.dict(os.environ, {}, clear=True):
            result = get_input_data_s3()
            self.assertEqual(result, self.valid_s3_uri)
    
    def test_get_input_data_s3_fallback(self):
        """Test fallback to default when no pinned dataset source is available"""
        with patch.dict(os.environ, {}, clear=True):
            result = get_input_data_s3()
            self.assertTrue(result.startswith('s3://hpo-bucket-773934887314/data/'))
    
    def test_get_input_data_s3_file_read_error(self):
        """Test handling of file read errors"""
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(self.valid_s3_uri)
        os.chmod('last_dataset_uri.txt', 0o000)
        
        with patch.dict(os.environ, {}, clear=True):
            result = get_input_data_s3()
            self.assertTrue(result.startswith('s3://hpo-bucket-773934887314/data/'))
        
        os.chmod('last_dataset_uri.txt', 0o644)
    
    def test_validate_s3_uri_format_valid(self):
        """Test S3 URI format validation with valid URI"""
        with patch('boto3.client') as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            mock_s3.head_object.return_value = {}
            
            result = validate_s3_uri(self.valid_s3_uri)
            self.assertTrue(result)
            mock_s3.head_object.assert_called_once()
    
    def test_validate_s3_uri_format_invalid(self):
        """Test S3 URI format validation with invalid URI"""
        with self.assertRaises(SystemExit):
            validate_s3_uri(self.invalid_s3_uri)
    
    def test_validate_s3_uri_empty(self):
        """Test S3 URI validation with empty/None input"""
        self.assertFalse(validate_s3_uri(None))
        self.assertFalse(validate_s3_uri(""))
    
    def test_validate_s3_uri_access_error(self):
        """Test S3 URI validation when S3 access fails"""
        with patch('boto3.client') as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            mock_s3.head_object.side_effect = Exception("Access denied")
            
            result = validate_s3_uri(self.valid_s3_uri)
            self.assertFalse(result)
    
    def test_validate_s3_uri_directory_path(self):
        """Test S3 URI validation with directory path (not .csv file)"""
        directory_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/"
        
        with patch('boto3.client') as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            mock_s3.list_objects_v2.return_value = {'Contents': [{'Key': 'test.csv'}]}
            
            result = validate_s3_uri(directory_uri)
            self.assertTrue(result)
            mock_s3.list_objects_v2.assert_called_once()
    
    def test_validate_s3_uri_empty_directory(self):
        """Test S3 URI validation with empty directory"""
        directory_uri = "s3://hpo-bucket-773934887314/empty/directory/"
        
        with patch('boto3.client') as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            mock_s3.list_objects_v2.return_value = {}  # No Contents key
            
            result = validate_s3_uri(directory_uri)
            self.assertFalse(result)


class TestSageMakerMocking(unittest.TestCase):
    """Test cases for SageMaker mocking in dry-run mode"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    @patch('aws_hpo_launch.sagemaker')
    @patch('aws_hpo_launch.boto3')
    def test_launch_aapl_hpo_dry_run(self, mock_boto3, mock_sagemaker):
        """Test AAPL HPO launch in dry-run mode with mocked SageMaker"""
        mock_session = MagicMock()
        mock_sagemaker.Session.return_value = mock_session
        
        mock_estimator = MagicMock()
        mock_sagemaker.estimator.Estimator.return_value = mock_estimator
        
        mock_tuner = MagicMock()
        mock_sagemaker.tuner.HyperparameterTuner.return_value = mock_tuner
        
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {}
        
        pinned_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(pinned_uri)
        
        from aws_hpo_launch import launch_aapl_hpo
        
        with patch.dict(os.environ, {}, clear=True):
            result = launch_aapl_hpo(dry_run=True)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('-dry-run'))
    
    @patch('aws_hpo_launch.sagemaker')
    @patch('aws_hpo_launch.boto3')
    def test_launch_full_universe_hpo_dry_run(self, mock_boto3, mock_sagemaker):
        """Test Full Universe HPO launch in dry-run mode with mocked SageMaker"""
        mock_session = MagicMock()
        mock_sagemaker.Session.return_value = mock_session
        
        mock_estimator = MagicMock()
        mock_sagemaker.estimator.Estimator.return_value = mock_estimator
        
        mock_tuner = MagicMock()
        mock_sagemaker.tuner.HyperparameterTuner.return_value = mock_tuner
        
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {}
        
        pinned_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
        
        from aws_hpo_launch import launch_full_universe_hpo
        
        with patch.dict(os.environ, {'PINNED_DATA_S3': pinned_uri}):
            result = launch_full_universe_hpo(dry_run=True)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('-dry-run'))
    
    @patch('aws_hpo_launch.sagemaker')
    @patch('aws_hpo_launch.boto3')
    def test_fallback_to_default_data_source(self, mock_boto3, mock_sagemaker):
        """Test fallback to default data source when pinned dataset is invalid"""
        mock_session = MagicMock()
        mock_sagemaker.Session.return_value = mock_session
        
        mock_estimator = MagicMock()
        mock_sagemaker.estimator.Estimator.return_value = mock_estimator
        
        mock_tuner = MagicMock()
        mock_sagemaker.tuner.HyperparameterTuner.return_value = mock_tuner
        
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.side_effect = Exception("Access denied")
        
        invalid_uri = "s3://invalid-bucket/invalid/path.csv"
        
        from aws_hpo_launch import launch_aapl_hpo
        
        with patch.dict(os.environ, {'PINNED_DATA_S3': invalid_uri}):
            result = launch_aapl_hpo(dry_run=True)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('-dry-run'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
