"""
Tests for inspect_parquet_data.py
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
import json
import io
import tempfile

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add aws_pipeline directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aws_pipeline'))

# Skip tests if module not found
try:
    from aws_pipeline.inspect_parquet_data import (
        inspect_parquet,
        format_inspection_results,
        main
    )
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True

@pytest.fixture
def sample_parquet_data():
    """Create a sample DataFrame that would be read from a Parquet file."""
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'symbol': ['AAPL'] * 5 + ['MSFT'] * 5,
        'open': np.random.rand(10) * 100,
        'high': np.random.rand(10) * 100,
        'low': np.random.rand(10) * 100,
        'close': np.random.rand(10) * 100,
        'volume': np.random.randint(1000, 10000, 10),
        'null_col': [None, 1, 2, None, 4, 5, None, 7, 8, 9]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_s3fs():
    """Create a mock S3FileSystem."""
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ['s3://bucket/path/file.parquet']
    return mock_fs

@pytest.mark.skipif(SKIP_TESTS, reason="inspect_parquet_data module not found")
def test_inspect_parquet(sample_parquet_data, mock_s3fs):
    """Test inspecting Parquet data."""
    with patch('s3fs.S3FileSystem', return_value=mock_s3fs), \
         patch('pandas.read_parquet', return_value=sample_parquet_data):
        
        # Test inspection
        results = inspect_parquet('s3://bucket/path/', 5)
        
        # Verify results
        assert results is not None
        assert results['file_path'] == 's3://bucket/path/file.parquet'
        assert results['num_rows'] == 10
        assert results['num_columns'] == 8
        assert 'timestamp' in results['columns']
        assert 'symbol' in results['columns']
        assert 'open' in results['columns']
        assert results['null_percentages']['null_col'] == 30.0  # 3 out of 10 are None
        assert 'open' in results['numeric_columns']
        assert 'high' in results['numeric_columns']
        assert 'low' in results['numeric_columns']
        assert 'close' in results['numeric_columns']
        assert 'volume' in results['numeric_columns']
        assert len(results['sample_data']) == 5

@pytest.mark.skipif(SKIP_TESTS, reason="inspect_parquet_data module not found")
def test_format_inspection_results():
    """Test formatting inspection results."""
    # Create sample inspection results
    results = {
        'file_path': 's3://bucket/path/file.parquet',
        'num_rows': 10,
        'num_columns': 5,
        'columns': {
            'timestamp': 'datetime64[ns]',
            'symbol': 'object',
            'open': 'float64',
            'close': 'float64',
            'volume': 'int64'
        },
        'null_percentages': {
            'timestamp': 0.0,
            'symbol': 0.0,
            'open': 10.0,
            'close': 0.0,
            'volume': 0.0
        },
        'numeric_columns': ['open', 'close', 'volume'],
        'sample_data': [
            {'timestamp': '2023-01-01', 'symbol': 'AAPL', 'open': 150.0, 'close': 155.0, 'volume': 1000},
            {'timestamp': '2023-01-02', 'symbol': 'AAPL', 'open': 155.0, 'close': 160.0, 'volume': 1200}
        ]
    }
    
    # Format results
    formatted = format_inspection_results(results)
    
    # Verify formatting
    assert isinstance(formatted, str)
    assert '🔍 Inspection Results' in formatted
    assert '📊 Dataset Size: 10 rows, 5 columns' in formatted
    assert '📋 Column List' in formatted
    assert '🔢 Null Percentages' in formatted
    assert '📝 Sample Data' in formatted
    assert '✅ Potential Numeric/Target Columns' in formatted
    assert 'open' in formatted
    assert 'close' in formatted
    assert 'volume' in formatted

@pytest.mark.skipif(SKIP_TESTS, reason="inspect_parquet_data module not found")
def test_main():
    """Test the main function."""
    test_args = ['inspect_parquet_data.py', 
                 '--s3-uri', 's3://bucket/path/', 
                 '--sample-size', '3',
                 '--region', 'us-east-1']
    
    # Create sample inspection results
    sample_results = {
        'file_path': 's3://bucket/path/file.parquet',
        'num_rows': 10,
        'num_columns': 5,
        'columns': {'col1': 'int64', 'col2': 'float64'},
        'null_percentages': {'col1': 0.0, 'col2': 10.0},
        'numeric_columns': ['col1', 'col2'],
        'sample_data': [{'col1': 1, 'col2': 1.1}, {'col1': 2, 'col2': 2.2}]
    }
    
    with patch('sys.argv', test_args), \
         patch('aws_pipeline.inspect_parquet_data.inspect_parquet', return_value=sample_results), \
         patch('aws_pipeline.inspect_parquet_data.format_inspection_results', return_value='Formatted Results'), \
         patch('builtins.print') as mock_print:
        
        # Execute main function
        result = main()
        
        # Verify result and function calls
        assert result is True
        mock_print.assert_called_once_with('Formatted Results')