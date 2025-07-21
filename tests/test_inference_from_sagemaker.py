"""
Tests for inference_from_sagemaker.py
"""
import sys
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import logging

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add aws_pipeline directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aws_pipeline'))

# Skip tests if module not found
try:
    from aws_pipeline.inference_from_sagemaker import (
        load_input_data,
        invoke_endpoint,
        main
    )
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True

@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session."""
    mock_session = MagicMock()
    mock_runtime = MagicMock()
    mock_s3 = MagicMock()
    
    # Set up session to return mocked clients
    mock_session.client.side_effect = lambda service, **kwargs: {
        'sagemaker-runtime': mock_runtime,
        's3': mock_s3
    }.get(service)
    
    return mock_session, mock_runtime, mock_s3

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'symbol': ['AAPL'] * 5 + ['MSFT'] * 5,
        'open': np.random.rand(10) * 100,
        'high': np.random.rand(10) * 100,
        'low': np.random.rand(10) * 100,
        'close': np.random.rand(10) * 100,
        'volume': np.random.randint(1000, 10000, 10)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_endpoint_response():
    """Create a sample endpoint response."""
    prediction_result = {
        'predictions': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    mock_response = {
        'Body': io.BytesIO(json.dumps(prediction_result).encode())
    }
    return mock_response

@pytest.mark.skipif(SKIP_TESTS, reason="inference_from_sagemaker module not found")
def test_load_input_data():
    """Test loading input data from file."""
    # Create a temporary CSV file
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5),
        'symbol': ['AAPL'] * 5,
        'price': [150.0, 151.0, 152.0, 153.0, 154.0]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
        sample_df.to_csv(temp_file.name, index=False)
        temp_file.flush()
        
        # Test loading from CSV
        result = load_input_data(temp_file.name, 3)
        assert len(result) == 3
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    # Test with no input file (dummy data generation)
    with patch('aws_pipeline.inference_from_sagemaker.logger.info') as mock_log:
        result = load_input_data(None, 5)
        assert len(result) == 5
        assert mock_log.called

@pytest.mark.skipif(SKIP_TESTS, reason="inference_from_sagemaker module not found")
def test_invoke_endpoint():
    """Test invoking the SageMaker endpoint."""
    # Setup mock boto3 client and response
    mock_response = {
        'Body': io.BytesIO(json.dumps([0.1, 0.2, 0.3]).encode())
    }
    
    with patch('boto3.Session') as mock_session:
        mock_runtime = MagicMock()
        mock_runtime.invoke_endpoint.return_value = mock_response
        mock_session.return_value.client.return_value = mock_runtime
        
        # Test invoke endpoint
        payload = [{'feature1': 1.0, 'feature2': 2.0}]
        result = invoke_endpoint('test-endpoint', payload, 'us-east-1')
        
        # Verify result and calls
        assert result == [0.1, 0.2, 0.3]
        mock_session.return_value.client.assert_called_once_with('sagemaker-runtime')
        mock_runtime.invoke_endpoint.assert_called_once_with(
            EndpointName='test-endpoint', 
            ContentType='application/json',
            Body=json.dumps(payload)
        )

@pytest.mark.skipif(SKIP_TESTS, reason="inference_from_sagemaker module not found")
def test_main():
    """Test the main function."""
    test_args = ['inference_from_sagemaker.py', 
                 '--endpoint-name', 'test-endpoint', 
                 '--sample-size', '3',
                 '--region', 'us-east-1']
    
    sample_records = [{'feature1': 1.0}, {'feature2': 2.0}, {'feature3': 3.0}]
    sample_predictions = [0.1, 0.2, 0.3]
    
    with patch('sys.argv', test_args), \
         patch('aws_pipeline.inference_from_sagemaker.load_input_data', return_value=sample_records), \
         patch('aws_pipeline.inference_from_sagemaker.invoke_endpoint', return_value=sample_predictions), \
         patch('logging.basicConfig'), \
         patch('pandas.DataFrame.to_csv'), \
         patch('logging.info'):
        
        # Execute main function
        result = main()
        assert result is True