#!/usr/bin/env python3
"""
Test that the AWS HPO launch script correctly uses the last successful HPO dataset
"""
import os
import sys
import unittest
import importlib
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataSourceSync(unittest.TestCase):
    """Test suite for data source synchronization"""
    
    def setUp(self):
        """Set up test environment - clear module cache before each test"""
        # Remove the module from sys.modules to force a fresh import each time
        if 'aws_hpo_launch' in sys.modules:
            del sys.modules['aws_hpo_launch']
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('boto3.client')
    @patch.dict(os.environ, {'PINNED_DATA_S3': 's3://pinned-bucket/pinned-data/'}, clear=True)
    def test_pinned_env_var_data_source(self, mock_boto3, mock_exists, mock_args, mock_exit):
        """Test that the script uses the PINNED_DATA_S3 environment variable (highest priority after CLI)"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked env var)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the env var value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://pinned-bucket/pinned-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('boto3.client')
    @patch.dict(os.environ, {'LAST_DATA_S3': 's3://test-bucket/test-data/'}, clear=True)
    def test_legacy_env_var_data_source(self, mock_boto3, mock_exists, mock_args, mock_exit):
        """Test that the script uses the LAST_DATA_S3 environment variable (legacy)"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file doesn't exist
        mock_exists.return_value = False
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked env var)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the env var value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://test-bucket/test-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='s3://file-bucket/file-data/')
    @patch('boto3.client')
    @patch.dict(os.environ, {}, clear=True)
    def test_file_data_source(self, mock_boto3, mock_file, mock_exists, mock_args, mock_exit):
        """Test that the script uses the last_dataset_uri.txt file"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file exists
        mock_exists.return_value = True
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked file)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the file value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://file-bucket/file-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('boto3.client')
    @patch.dict(os.environ, {'PINNED_DATA_S3': 's3://pinned-bucket/pinned-data/', 
                            'LAST_DATA_S3': 's3://test-bucket/test-data/'}, clear=True)
    def test_pinned_overrides_legacy(self, mock_boto3, mock_exists, mock_args, mock_exit):
        """Test that PINNED_DATA_S3 takes precedence over LAST_DATA_S3"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file doesn't exist
        mock_exists.return_value = False
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked env vars)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the PINNED_DATA_S3 value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://pinned-bucket/pinned-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='s3://file-bucket/file-data/')
    @patch('boto3.client')
    @patch.dict(os.environ, {'LAST_DATA_S3': 's3://test-bucket/test-data/'}, clear=True)
    def test_file_overrides_legacy(self, mock_boto3, mock_file, mock_exists, mock_args, mock_exit):
        """Test that file takes precedence over LAST_DATA_S3"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file exists
        mock_exists.return_value = True
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked file and env vars)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the file value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://file-bucket/file-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='s3://file-bucket/file-data/')
    @patch('boto3.client')
    @patch.dict(os.environ, {'PINNED_DATA_S3': 's3://pinned-bucket/pinned-data/'}, clear=True)
    def test_pinned_overrides_file(self, mock_boto3, mock_file, mock_exists, mock_args, mock_exit):
        """Test that PINNED_DATA_S3 takes precedence over file"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file exists
        mock_exists.return_value = True
        
        # Mock S3 client for existence check
        s3_client = MagicMock()
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked env vars and file)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the PINNED_DATA_S3 value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://pinned-bucket/pinned-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch.dict(os.environ, {'LAST_DATA_S3': 's3://test-bucket/test-data/'}, clear=True)
    def test_cli_arg_override(self, mock_args, mock_exit):
        """Test that CLI args override the environment variable"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3='s3://override-bucket/override-data/',
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )
        
        # Import the script (this will use the mocked args)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the CLI arg value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://override-bucket/override-data/')
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch.dict(os.environ, {'LAST_DATA_S3': 's3://test-bucket/test-data/'}, clear=True)
    def test_force_default_data(self, mock_args, mock_exit):
        """Test that force_default_data overrides env var and CLI arg"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3='s3://override-bucket/override-data/',
            dry_run=True,
            force_default_data=True,
            target_completed=138
        )
        
        # Import the script (this will use default due to force flag)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is set to the default value
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, aws_hpo_launch.DEFAULT_DATA_PREFIX)
        mock_exit.assert_not_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch.dict(os.environ, {'LAST_DATA_S3': 'invalid-uri'}, clear=True)
    def test_invalid_s3_uri_env_var(self, mock_args, mock_exit):
        """Test that the script validates S3 URIs from environment variables"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )
        
        # Import should call sys.exit due to invalid URI
        importlib.import_module('aws_hpo_launch')
        mock_exit.assert_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='invalid-uri')
    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_s3_uri_file(self, mock_file, mock_exists, mock_args, mock_exit):
        """Test that the script validates S3 URIs from the file"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )

        # Mock that the file exists
        mock_exists.return_value = True
        
        # Import should call sys.exit due to invalid URI in file
        importlib.import_module('aws_hpo_launch')
        mock_exit.assert_called()
    
    @patch('sys.exit')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('boto3.client')
    @patch.dict(os.environ, {'PINNED_DATA_S3': 's3://pinned-bucket/pinned-data/'}, clear=True)
    def test_s3_existence_check(self, mock_boto3, mock_args, mock_exit):
        """Test that the script checks if the S3 object exists"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True,
            force_default_data=False,
            target_completed=138
        )
        
        # Mock S3 client for existence check - object does not exist
        s3_client = MagicMock()
        s3_client.head_object.side_effect = Exception("Object does not exist")
        mock_boto3.return_value = s3_client
        
        # Import the script (this will use the mocked env var and S3 client)
        aws_hpo_launch = importlib.import_module('aws_hpo_launch')
        
        # Assert that the S3_DATA_PREFIX is still set to the env var value (warning only)
        self.assertEqual(aws_hpo_launch.S3_DATA_PREFIX, 's3://pinned-bucket/pinned-data/')
        mock_exit.assert_not_called()  # Should not exit on existence check failure
