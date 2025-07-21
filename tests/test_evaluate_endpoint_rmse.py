#!/usr/bin/env python3
"""
test_evaluate_endpoint_rmse.py - Unit tests for evaluate_endpoint_rmse.py
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import io
import tempfile

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
import evaluate_endpoint_rmse

class TestEvaluateEndpointRMSE(unittest.TestCase):
    """Test cases for evaluate_endpoint_rmse.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary CSV file with holdout data
        self.holdout_data = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.holdout_csv = self.temp_file.name
        
        # Save the data to the temporary file
        self.holdout_data.to_csv(self.holdout_csv, index=False)
        
        # Mock predictions to return
        self.mock_predictions = [11.0, 21.0, 31.0, 41.0, 51.0]
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Remove the temporary file
        os.unlink(self.holdout_csv)
        
        # Remove output file if it exists
        if os.path.exists('predictions_with_labels.csv'):
            os.unlink('predictions_with_labels.csv')
    
    @patch('boto3.Session')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_evaluate_endpoint(self, mock_stdout, mock_boto3_session):
        """Test the evaluation process with mocked boto3"""
        # Mock the boto3 session and runtime client
        mock_runtime = MagicMock()
        mock_boto3_session.return_value.client.return_value = mock_runtime
        
        # Mock the response from the endpoint
        mock_response = {
            'Body': MagicMock()
        }
        # Create a proper CSV format for the response
        mock_response['Body'].read.return_value = '\n'.join(str(p) for p in self.mock_predictions).encode('utf-8')
        mock_runtime.invoke_endpoint.return_value = mock_response
        
        # Set up command line arguments
        test_args = [
            '--endpoint-name', 'test-endpoint',
            '--holdout-csv', self.holdout_csv,
            '--target-col', 'target',
            '--batch-size', '2'
        ]
        
        # Patch sys.argv
        with patch('sys.argv', ['evaluate_endpoint_rmse.py'] + test_args):
            # Run the main function
            result = evaluate_endpoint_rmse.main()
            
            # Check that the function returned True (success)
            self.assertTrue(result)
            
            # Check that the boto3 client was called with the right endpoint
            mock_boto3_session.return_value.client.assert_called_with('sagemaker-runtime')
            
            # Check that invoke_endpoint was called
            self.assertTrue(mock_runtime.invoke_endpoint.called)
            
            # Check that metrics were printed
            output = mock_stdout.getvalue()
            self.assertIn('RMSE:', output)
            self.assertIn('MAE:', output)
            self.assertIn('R²:', output)
            
            # Check that predictions_with_labels.csv was created
            self.assertTrue(os.path.exists('predictions_with_labels.csv'))
            
            # Load the predictions file and check its content
            predictions_df = pd.read_csv('predictions_with_labels.csv')
            self.assertEqual(len(predictions_df), len(self.holdout_data))
            self.assertIn('prediction', predictions_df.columns)

if __name__ == '__main__':
    unittest.main()
