"""
Tests for inspect_parquet_data.py script.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock, call

# Add the parent directory to the path so we can import the script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the script
from inspect_parquet_data import inspect_parquet

@pytest.fixture
def sample_parquet_data(tmp_path):
    """Create a sample parquet file with test data."""
    # Create sample data
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOG', 'AMZN'],
        'timestamp': pd.date_range(start='2023-01-01', periods=4),
        'open': [150.0, 250.0, 180.0, 130.0],
        'high': [155.0, 255.0, 185.0, 135.0],
        'low': [145.0, 245.0, 175.0, 125.0],
        'close': [153.0, 253.0, 183.0, 133.0],
        'volume': [10000, 5000, 7500, 12000]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet file
    file_path = os.path.join(tmp_path, 'test_data.parquet')
    df.to_parquet(file_path)
    
    return {'path': file_path, 'df': df}

def test_inspect_parquet_data(sample_parquet_data):
    """Test the inspect_parquet_data function."""
    # Mock the necessary functions
    with patch('inspect_parquet_data.pd.read_parquet') as mock_read_parquet, \
         patch('inspect_parquet_data.s3fs.S3FileSystem') as mock_s3fs, \
         patch('builtins.print') as mock_print:
        
        # Configure mocks
        mock_s3fs.return_value.glob.return_value = [sample_parquet_data['path']]
        mock_read_parquet.return_value = sample_parquet_data['df']
        
        # Call the function
        inspect_parquet('s3://test-bucket/test-prefix/', 5)
        
        # Verify the function called the expected methods
        mock_s3fs.return_value.glob.assert_called_once()
        mock_read_parquet.assert_called_once()
        
        # Verify print was called multiple times (schema, nulls, sample data)
        assert mock_print.call_count > 5