"""
Tests for run_sagemaker_autopilot.py
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call, ANY
import json
import datetime

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip tests if module not found
try:
    from run_sagemaker_autopilot import run_sagemaker_autopilot, parse_arguments
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True

@pytest.fixture
def mock_args():
    """Create mock args for the test."""
    args = MagicMock()
    args.role_arn = 'arn:aws:iam::123456789012:role/SageMakerRole'
    args.bucket = 'test-bucket'
    args.input_s3_uri = 's3://test-bucket/input'
    args.output_s3_uri = 's3://test-bucket/output'
    args.target_column = 'target'
    args.problem_type = 'Regression'
    args.metric_name = 'RMSE'
    args.max_candidates = 10
    args.max_runtime_hours = 1
    args.max_runtime_per_job_minutes = 30
    args.instance_type = 'ml.m5.large'
    args.region = 'us-east-1'
    args.split_type = 'RANDOM'
    args.timestamp_col = None
    args.validation_fraction = 0.2
    args.use_automl_v2 = False
    return args

@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session with responses."""
    mock_session = MagicMock()
    mock_sagemaker_client = MagicMock()
    mock_sm_runtime = MagicMock()
    
    # Setup responses for API calls
    mock_sagemaker_client.create_auto_ml_job.return_value = {
        'AutoMLJobArn': 'arn:aws:sagemaker:us-east-1:123456789012:automl-job/test-job'
    }
    
    mock_sagemaker_client.create_auto_ml_job_v2.return_value = {
        'AutoMLJobArn': 'arn:aws:sagemaker:us-east-1:123456789012:automl-job/test-job-v2'
    }
    
    mock_sagemaker_client.describe_auto_ml_job.return_value = {
        'AutoMLJobStatus': 'Completed',
        'AutoMLJobSecondaryStatus': 'Completed',
        'BestCandidate': {
            'CandidateName': 'test-candidate',
            'FinalAutoMLJobObjectiveMetric': {
                'MetricName': 'RMSE',
                'Value': 0.123
            },
            'InferenceContainers': [
                {'Image': 'test-image'}
            ],
            'ModelDataUrl': 's3://test-bucket/model-data'
        }
    }
    
    mock_sagemaker_client.describe_auto_ml_job_v2.return_value = {
        'AutoMLJobStatus': 'Completed',
        'AutoMLJobSecondaryStatus': 'Completed',
        'BestCandidate': {
            'CandidateName': 'test-candidate-v2',
            'FinalAutoMLJobObjectiveMetric': {
                'MetricName': 'RMSE',
                'Value': 0.123
            },
            'InferenceContainers': [
                {'Image': 'test-image-v2'}
            ],
            'ModelDataUrl': 's3://test-bucket/model-data-v2'
        }
    }
    
    mock_sagemaker_client.create_model.return_value = {
        'ModelArn': 'arn:aws:sagemaker:us-east-1:123456789012:model/test-model'
    }
    
    mock_sagemaker_client.create_endpoint_config.return_value = {
        'EndpointConfigArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/test-config'
    }
    
    mock_sagemaker_client.create_endpoint.return_value = {
        'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint'
    }
    
    mock_sagemaker_client.describe_endpoint.return_value = {
        'EndpointStatus': 'InService'
    }
    
    mock_session.client.side_effect = lambda service: {
        'sagemaker': mock_sagemaker_client,
        'sagemaker-runtime': mock_sm_runtime
    }.get(service)
    
    return mock_session

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_run_sagemaker_autopilot_v1(mock_args, mock_boto3_session):
    """Test running SageMaker Autopilot with V1 API."""
    # Set up mocks
    mock_args.use_automl_v2 = False
    
    # Mock the CSV dataset check
    with patch('boto3.Session', return_value=mock_boto3_session), \
         patch('run_sagemaker_autopilot.ensure_csv_dataset', return_value=mock_args.input_s3_uri), \
         patch('time.sleep'), \
         patch('json.dump'), \
         patch('builtins.open', MagicMock()):
        
        # Run the function
        result = run_sagemaker_autopilot(mock_args)
        
        # Verify the result
        assert result is True
        
        # Verify API calls
        sagemaker_client = mock_boto3_session.client('sagemaker')
        assert sagemaker_client.create_auto_ml_job.call_count == 1
        assert sagemaker_client.create_auto_ml_job_v2.call_count == 0
        assert sagemaker_client.describe_auto_ml_job.call_count >= 1
        assert sagemaker_client.describe_auto_ml_job_v2.call_count == 0
        assert sagemaker_client.create_model.call_count == 1
        assert sagemaker_client.create_endpoint_config.call_count == 1
        assert sagemaker_client.create_endpoint.call_count == 1
        assert sagemaker_client.describe_endpoint.call_count >= 1

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_run_sagemaker_autopilot_v2(mock_args, mock_boto3_session):
    """Test running SageMaker Autopilot with V2 API."""
    # Set up mocks
    mock_args.use_automl_v2 = True
    
    # Mock the CSV dataset check
    with patch('boto3.Session', return_value=mock_boto3_session), \
         patch('run_sagemaker_autopilot.ensure_csv_dataset', return_value=mock_args.input_s3_uri), \
         patch('time.sleep'), \
         patch('json.dump'), \
         patch('builtins.open', MagicMock()):
        
        # Run the function
        result = run_sagemaker_autopilot(mock_args)
        
        # Verify the result
        assert result is True
        
        # Verify API calls
        sagemaker_client = mock_boto3_session.client('sagemaker')
        assert sagemaker_client.create_auto_ml_job.call_count == 0
        assert sagemaker_client.create_auto_ml_job_v2.call_count == 1
        assert sagemaker_client.describe_auto_ml_job.call_count == 0
        assert sagemaker_client.describe_auto_ml_job_v2.call_count >= 1
        assert sagemaker_client.create_model.call_count == 1
        assert sagemaker_client.create_endpoint_config.call_count == 1
        assert sagemaker_client.create_endpoint.call_count == 1
        assert sagemaker_client.describe_endpoint.call_count >= 1

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_run_sagemaker_autopilot_v2_timestamp(mock_args, mock_boto3_session):
    """Test running SageMaker Autopilot with V2 API and timestamp split."""
    # Set up mocks
    mock_args.use_automl_v2 = True
    mock_args.split_type = 'TIMESTAMP'
    mock_args.timestamp_col = 'date'
    
    # Mock the CSV dataset check
    with patch('boto3.Session', return_value=mock_boto3_session), \
         patch('run_sagemaker_autopilot.ensure_csv_dataset', return_value=mock_args.input_s3_uri), \
         patch('time.sleep'), \
         patch('json.dump'), \
         patch('builtins.open', MagicMock()):
        
        # Run the function
        result = run_sagemaker_autopilot(mock_args)
        
        # Verify the result
        assert result is True
        
        # Verify API calls
        sagemaker_client = mock_boto3_session.client('sagemaker')
        assert sagemaker_client.create_auto_ml_job.call_count == 0
        assert sagemaker_client.create_auto_ml_job_v2.call_count == 1
        
        # Verify TabularJobConfig contains TimestampAttributeName
        call_args = sagemaker_client.create_auto_ml_job_v2.call_args[1]
        tabular_job_config = call_args['AutoMLProblemTypeConfig']['TabularJobConfig']
        assert 'TimestampAttributeName' in tabular_job_config
        assert tabular_job_config['TimestampAttributeName'] == 'date'

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_run_sagemaker_autopilot_v2_fallback_to_v1(mock_args, mock_boto3_session):
    """Test falling back to V1 API when V2 fails."""
    # Set up mocks
    mock_args.use_automl_v2 = True
    mock_args.split_type = 'RANDOM'
    
    # Make V2 API fail
    sagemaker_client = mock_boto3_session.client('sagemaker')
    sagemaker_client.create_auto_ml_job_v2.side_effect = Exception("V2 API not available")
    
    # Mock the CSV dataset check
    with patch('boto3.Session', return_value=mock_boto3_session), \
         patch('run_sagemaker_autopilot.ensure_csv_dataset', return_value=mock_args.input_s3_uri), \
         patch('time.sleep'), \
         patch('json.dump'), \
         patch('builtins.open', MagicMock()):
        
        # Run the function
        result = run_sagemaker_autopilot(mock_args)
        
        # Verify the result
        assert result is True
        
        # Verify API calls - should have tried V2 then fallen back to V1
        assert sagemaker_client.create_auto_ml_job_v2.call_count == 1
        assert sagemaker_client.create_auto_ml_job.call_count == 1
        assert sagemaker_client.describe_auto_ml_job.call_count >= 1
        assert sagemaker_client.describe_auto_ml_job_v2.call_count == 0

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_parse_arguments_default():
    """Test parse_arguments with default values."""
    with patch('argparse.ArgumentParser.parse_args',
               return_value=MagicMock(
                   role_arn='test-role',
                   bucket='test-bucket',
                   input_s3_uri=None,
                   output_s3_uri=None,
                   target_column='return',
                   problem_type='Regression',
                   metric_name='RMSE',
                   max_candidates=10,
                   max_runtime_hours=2,
                   max_runtime_per_job_minutes=30,
                   instance_type='ml.m5.large',
                   region='us-east-1',
                   split_type='random',
                   timestamp_col=None,
                   validation_fraction=0.2,
                   use_automl_v2=False
               )):
        args = parse_arguments()
        assert args.role_arn == 'test-role'
        assert args.bucket == 'test-bucket'
        assert args.target_column == 'return'
        assert args.problem_type == 'Regression'
        assert args.split_type == 'RANDOM'  # Should be normalized to uppercase
        assert args.use_automl_v2 is False

@pytest.mark.skipif(SKIP_TESTS, reason="run_sagemaker_autopilot module not found")
def test_parse_arguments_timestamp_split():
    """Test parse_arguments with timestamp split."""
    with patch('argparse.ArgumentParser.parse_args',
               return_value=MagicMock(
                   role_arn='test-role',
                   bucket='test-bucket',
                   input_s3_uri=None,
                   output_s3_uri=None,
                   target_column='return',
                   problem_type='Regression',
                   metric_name='RMSE',
                   max_candidates=10,
                   max_runtime_hours=2,
                   max_runtime_per_job_minutes=30,
                   instance_type='ml.m5.large',
                   region='us-east-1',
                   split_type='timestamp',
                   timestamp_col='date',
                   validation_fraction=0.2,
                   use_automl_v2=False  # This should be auto-set to True
               )):
        args = parse_arguments()
        assert args.role_arn == 'test-role'
        assert args.split_type == 'TIMESTAMP'  # Should be normalized to uppercase
        assert args.timestamp_col == 'date'
        assert args.use_automl_v2 is True  # Should be automatically set to True
