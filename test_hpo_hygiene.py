#!/usr/bin/env python3
"""
Test script to verify HPO training script hygiene
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aws_hpo_launch import get_input_data_s3, validate_s3_uri, launch_aapl_hpo, launch_full_universe_hpo


class TestHPOHygiene(unittest.TestCase):
    """Test HPO training script hygiene"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        self.valid_s3_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
        self.invalid_s3_uri = "invalid://not-s3-uri"
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_cli_arg_precedence(self):
        """Test CLI argument has highest precedence"""
        cli_arg = "s3://cli-bucket/cli-data.csv"
        env_var = "s3://env-bucket/env-data.csv"
        file_data = "s3://file-bucket/file-data.csv"
        
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(file_data)
        
        with patch.dict(os.environ, {'PINNED_DATA_S3': env_var}):
            result = get_input_data_s3(cli_arg)
            self.assertEqual(result, cli_arg, "CLI argument should have highest precedence")
    
    def test_env_var_precedence(self):
        """Test environment variable has second precedence"""
        env_var = "s3://env-bucket/env-data.csv"
        file_data = "s3://file-bucket/file-data.csv"
        
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(file_data)
        
        with patch.dict(os.environ, {'PINNED_DATA_S3': env_var}):
            result = get_input_data_s3(None)
            self.assertEqual(result, env_var, "Environment variable should have second precedence")
    
    def test_file_fallback(self):
        """Test file has third precedence"""
        file_data = "s3://file-bucket/file-data.csv"
        
        with open('last_dataset_uri.txt', 'w') as f:
            f.write(file_data)
        
        with patch.dict(os.environ, {}, clear=True):
            result = get_input_data_s3(None)
            self.assertEqual(result, file_data, "File should be used as fallback")
    
    def test_s3_uri_regex_validation(self):
        """Test S3 URI regex validation"""
        valid_uris = [
            "s3://bucket/path/file.csv",
            "s3://my-bucket/deep/nested/path/data.parquet",
            "s3://bucket123/folder_name/file-name.csv"
        ]
        
        invalid_uris = [
            "http://bucket/path/file.csv",
            "s3://",
            "s3://bucket",
            "bucket/path/file.csv",
            "",
            None
        ]
        
        with patch('aws_hpo_launch.boto3.client') as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            mock_s3.head_object.return_value = {}
            
            for uri in valid_uris:
                with self.subTest(uri=uri):
                    try:
                        result = validate_s3_uri(uri)
                        self.assertTrue(result, f"Valid URI should pass: {uri}")
                    except SystemExit:
                        self.fail(f"Valid URI should not cause SystemExit: {uri}")
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                with self.assertRaises(SystemExit, msg=f"Invalid URI should cause SystemExit: {uri}"):
                    validate_s3_uri(uri)
    
    @patch('aws_hpo_launch.sagemaker')
    @patch('aws_hpo_launch.boto3')
    def test_dry_run_no_sagemaker_calls(self, mock_boto3, mock_sagemaker):
        """Test dry-run mode makes no SageMaker calls"""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {}
        
        mock_session = MagicMock()
        mock_estimator = MagicMock()
        mock_tuner = MagicMock()
        
        mock_sagemaker.Session.return_value = mock_session
        mock_sagemaker.estimator.Estimator.return_value = mock_estimator
        mock_sagemaker.tuner.HyperparameterTuner.return_value = mock_tuner
        
        result = launch_aapl_hpo(self.valid_s3_uri, dry_run=True)
        
        self.assertIsNotNone(result)
        self.assertIn("dry-run", result)
        
        mock_sagemaker.Session.assert_not_called()
        mock_sagemaker.estimator.Estimator.assert_not_called()
        mock_sagemaker.tuner.HyperparameterTuner.assert_not_called()
        mock_tuner.fit.assert_not_called()
    
    @patch('aws_hpo_launch.sagemaker')
    @patch('aws_hpo_launch.boto3')
    def test_real_run_makes_sagemaker_calls(self, mock_boto3, mock_sagemaker):
        """Test real run makes SageMaker calls"""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {}
        
        mock_session = MagicMock()
        mock_estimator = MagicMock()
        mock_tuner = MagicMock()
        
        mock_sagemaker.Session.return_value = mock_session
        mock_sagemaker.estimator.Estimator.return_value = mock_estimator
        mock_sagemaker.tuner.HyperparameterTuner.return_value = mock_tuner
        
        result = launch_aapl_hpo(self.valid_s3_uri, dry_run=False)
        
        self.assertIsNotNone(result)
        self.assertNotIn("dry-run", result)
        
        mock_sagemaker.Session.assert_called_once()
        mock_sagemaker.estimator.Estimator.assert_called_once()
        mock_sagemaker.tuner.HyperparameterTuner.assert_called_once()
        mock_tuner.fit.assert_called_once()


def run_hygiene_tests():
    """Run all hygiene tests"""
    print("🧪 Testing HPO Training Script Hygiene")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHPOHygiene)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All hygiene tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_hygiene_tests()
    sys.exit(0 if success else 1)
