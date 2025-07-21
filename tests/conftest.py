"""
Common test fixtures for AWS ML pipeline tests.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


@pytest.fixture
def mock_spark_session():
    """Create a mock SparkSession for testing."""
    mock_session = MagicMock(spec=SparkSession)
    
    # Configure read.parquet to return a dataframe
    mock_read = MagicMock()
    mock_session.read = mock_read
    mock_read.parquet = MagicMock()
    
    return mock_session


@pytest.fixture
def mock_glue_context():
    """Create a mock GlueContext for testing."""
    mock_context = MagicMock()
    mock_context.spark_session = MagicMock(spec=SparkSession)
    return mock_context


@pytest.fixture
def mock_spark_context():
    """Create a mock SparkContext for testing."""
    return MagicMock()


@pytest.fixture
def mock_job():
    """Create a mock Job for testing."""
    mock_job = MagicMock()
    mock_job.init = MagicMock()
    mock_job.commit = MagicMock()
    return mock_job


@pytest.fixture
def mock_boto3_sagemaker():
    """Create a mock boto3 sagemaker client."""
    mock_client = MagicMock()
    
    # Setup responses
    mock_client.create_auto_ml_job.return_value = {
        'AutoMLJobArn': 'arn:aws:sagemaker:us-west-2:123456789012:auto-ml-job/test-job'
    }
    
    mock_client.describe_auto_ml_job.return_value = {
        'AutoMLJobStatus': 'Completed',
        'AutoMLJobName': 'test-job',
        'BestCandidate': {
            'CandidateName': 'candidate-1',
            'InferenceContainers': [{'Image': 'test-image'}],
            'CandidateProperties': {
                'ModelInsightsConfig': {}
            }
        }
    }
    
    mock_client.list_endpoints.return_value = {
        'Endpoints': [
            {'EndpointName': 'test-endpoint-1'},
            {'EndpointName': 'test-endpoint-2'}
        ]
    }
    
    mock_client.list_endpoint_configs.return_value = {
        'EndpointConfigs': [
            {'EndpointConfigName': 'test-config-1'},
            {'EndpointConfigName': 'test-config-2'}
        ]
    }
    
    mock_client.list_models.return_value = {
        'Models': [
            {'ModelName': 'test-model-1'},
            {'ModelName': 'test-model-2'}
        ]
    }
    
    return mock_client


@pytest.fixture
def mock_boto3_sagemaker_runtime():
    """Create a mock boto3 sagemaker runtime client."""
    mock_client = MagicMock()
    
    # Setup response for invoke_endpoint
    mock_client.invoke_endpoint.return_value = {
        'Body': MagicMock(read=MagicMock(return_value=b'{"predictions": [1, 0, 1]}'))
    }
    
    return mock_client


@pytest.fixture
def sample_minute_df():
    """Create a sample minute-level DataFrame."""
    # Create a mock DataFrame with schema
    schema = StructType([
        StructField("symbol", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("open", StringType(), True),
        StructField("high", StringType(), True),
        StructField("low", StringType(), True),
        StructField("close", StringType(), True),
        StructField("volume", StringType(), True),
        StructField("dayofweek", IntegerType(), True),
    ])
    
    # Create mock data
    data = [
        ("AAPL", "2023-01-01 09:30:00", "150.0", "151.5", "149.5", "151.0", "10000", 1),
        ("AAPL", "2023-01-01 09:31:00", "151.0", "152.0", "150.8", "151.8", "8500", 1),
        ("MSFT", "2023-01-01 09:30:00", "250.0", "251.0", "249.5", "250.5", "5000", 1),
        ("MSFT", "2023-01-01 09:31:00", "250.5", "251.5", "250.0", "251.0", "4800", 1),
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.schema = schema
    mock_df.columns = [field.name for field in schema.fields]
    mock_df.count.return_value = len(data)
    mock_df.withColumn.return_value = mock_df  # Return self for chaining
    mock_df.join.return_value = mock_df  # Return self for chaining
    mock_df.printSchema.return_value = None
    mock_df.write.mode.return_value.parquet = MagicMock()
    
    return mock_df


@pytest.fixture
def sample_daily_df():
    """Create a sample daily-level DataFrame."""
    # Create a mock DataFrame with schema
    schema = StructType([
        StructField("symbol", StringType(), False),
        StructField("date", StringType(), False),
        StructField("open", StringType(), True),
        StructField("high", StringType(), True),
        StructField("low", StringType(), True),
        StructField("close", StringType(), True),
        StructField("volume", StringType(), True),
        StructField("dayofweek", IntegerType(), True),
    ])
    
    # Create mock data
    data = [
        ("AAPL", "2023-01-01", "150.0", "155.0", "148.0", "153.0", "1000000", 1),
        ("MSFT", "2023-01-01", "250.0", "255.0", "248.0", "253.0", "800000", 1),
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.schema = schema
    mock_df.columns = [field.name for field in schema.fields]
    mock_df.count.return_value = len(data)
    mock_df.withColumn.return_value = mock_df  # Return self for chaining
    mock_df.join.return_value = mock_df  # Return self for chaining
    mock_df.printSchema.return_value = None
    mock_df.write.mode.return_value.parquet = MagicMock()
    
    return mock_df


@pytest.fixture
def sample_options_df():
    """Create a sample options DataFrame."""
    # Create a mock DataFrame with schema
    schema = StructType([
        StructField("symbol", StringType(), False),
        StructField("date", StringType(), False),
        StructField("strike", StringType(), True),
        StructField("option_type", StringType(), True),
        StructField("expiration", StringType(), True),
        StructField("price", StringType(), True),
        StructField("volume", StringType(), True),
        StructField("open_interest", StringType(), True),
    ])
    
    # Create mock data
    data = [
        ("AAPL", "2023-01-01", "150.0", "call", "2023-01-15", "5.0", "1000", "5000"),
        ("AAPL", "2023-01-01", "150.0", "put", "2023-01-15", "4.5", "800", "4500"),
        ("MSFT", "2023-01-01", "250.0", "call", "2023-01-15", "7.0", "600", "3000"),
        ("MSFT", "2023-01-01", "250.0", "put", "2023-01-15", "6.5", "500", "2800"),
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.schema = schema
    mock_df.columns = [field.name for field in schema.fields]
    mock_df.count.return_value = len(data)
    mock_df.withColumn.return_value = mock_df  # Return self for chaining
    mock_df.join.return_value = mock_df  # Return self for chaining
    mock_df.printSchema.return_value = None
    mock_df.write.mode.return_value.parquet = MagicMock()
    
    return mock_df


@pytest.fixture
def sample_news_df():
    """Create a sample news DataFrame."""
    # Create a mock DataFrame with schema
    schema = StructType([
        StructField("symbol", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("headline", StringType(), True),
        StructField("source", StringType(), True),
        StructField("url", StringType(), True),
        StructField("sentiment", StringType(), True),
    ])
    
    # Create mock data
    data = [
        ("AAPL", "2023-01-01 08:00:00", "Apple Announces New iPhone", "CNBC", "http://example.com/1", "0.75"),
        ("MSFT", "2023-01-01 09:15:00", "Microsoft Cloud Revenue Grows", "Bloomberg", "http://example.com/2", "0.85"),
    ]
    
    # Create a mock DataFrame
    mock_df = MagicMock()
    mock_df.schema = schema
    mock_df.columns = [field.name for field in schema.fields]
    mock_df.count.return_value = len(data)
    mock_df.withColumn.return_value = mock_df  # Return self for chaining
    mock_df.join.return_value = mock_df  # Return self for chaining
    mock_df.printSchema.return_value = None
    mock_df.write.mode.return_value.parquet = MagicMock()
    
    return mock_df


@pytest.fixture
def sample_parquet_df():
    """Create a sample pandas DataFrame for parquet inspection."""
    # Create a pandas DataFrame
    data = {
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'timestamp': ['2023-01-01 09:30:00', '2023-01-01 09:31:00', '2023-01-01 09:30:00', '2023-01-01 09:31:00'],
        'open': [150.0, 151.0, 250.0, 250.5],
        'high': [151.5, 152.0, 251.0, 251.5],
        'low': [149.5, 150.8, 249.5, 250.0],
        'close': [151.0, 151.8, 250.5, 251.0],
        'volume': [10000, 8500, 5000, 4800],
    }
    
    # Create DataFrame with some null values
    df = pd.DataFrame(data)
    df.loc[1, 'volume'] = None
    
    return df
