#!/usr/bin/env python3
"""
Test that the AWS HPO launch script correctly uses the last successful HPO dataset
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataSourceSync(unittest.TestCase):
    """Test suite for data source synchronization"""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch.dict(os.environ, {'LAST_DATA_S3': 's3://test-bucket/test-data/'})
    def test_env_var_data_source(self, mock_args):
        """Test that the script uses the LAST_DATA_S3 environment variable"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3=None,
            dry_run=True
        )
        
        # Import the script (this will use the mocked env var)
        from aws_hpo_launch import S3_DATA_PREFIX
        
        # Assert that the S3_DATA_PREFIX is set to the env var value
        self.assertEqual(S3_DATA_PREFIX, 's3://test-bucket/test-data/')
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_arg_override(self, mock_args):
        """Test that CLI args override the environment variable"""
        # Mock the ArgumentParser
        mock_args.return_value = MagicMock(
            input_data_s3='s3://override-bucket/override-data/',
            dry_run=True
        )
        
        # Import the script (this will use the mocked args)
        from aws_hpo_launch import S3_DATA_PREFIX
        
        # Assert that the S3_DATA_PREFIX is set to the CLI arg value
        self.assertEqual(S3_DATA_PREFIX, 's3://override-bucket/override-data/')
    
if __name__ == '__main__':
    unittest.main()
