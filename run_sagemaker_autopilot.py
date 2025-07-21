#!/usr/bin/env python3
"""
run_sagemaker_autopilot.py - Script to run SageMaker Autopilot for the Conviction-AI project

This script launches a SageMaker Autopilot job to train and deploy a model using the 
cleaned Parquet data in S3. It supports both AutoML V1 and V2 APIs, and can handle 
timestamp-based splits for time series data.

Key features:
- Supports both V1 and V2 SageMaker Autopilot APIs
- Handles time-based splits for time series data using V2 API
- Automatically converts Parquet data to CSV format required by SageMaker
- Deploys the best model to an endpoint for inference
"""

import argparse
import boto3
import sagemaker
import time
import json
import logging
from datetime import datetime
import uuid
import os
import sys
from dotenv import load_dotenv
# Import our dataset format checking module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from check_dataset_format import ensure_csv_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)  # Force override of existing env vars

# Print debug information about environment variables
logger.info(f"Using S3 bucket from env: {os.environ.get('S3_BUCKET_NAME', 'No bucket in env')}")

# Default values - can be overridden by command line arguments
TARGET_COLUMN = "return"        # Default target column to predict
PROBLEM_TYPE = "Regression"     # Default problem type (Regression for continuous values)
METRIC_NAME = "RMSE"            # Default metric for regression problems
MAX_CANDIDATES = 10             # Maximum number of model candidates to try
MAX_RUNTIME_HOURS = 2           # Maximum runtime for the AutoML job in hours
MAX_RUNTIME_PER_JOB_MINUTES = 30  # Maximum runtime per training job in minutes

# Define global variables for bucket names
S3_BUCKET_CANDIDATES = ['sagemaker-us-east-1-773934887314', 'convictionai-data']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run SageMaker Autopilot for Conviction-AI')
    
    # Use role ARN from environment if available
    default_role_arn = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
    
    parser.add_argument('--role-arn', type=str, default=default_role_arn,
                        required=default_role_arn is None,
                        help='ARN of the IAM role for SageMaker execution')
    
    # Get S3 bucket name from environment if available, but force it if needed
    # Try a list of known bucket names that may have proper permissions
    
    # You can override this with the --bucket argument
    parser.add_argument('--bucket', type=str, default=None,
                        help='S3 bucket name to use for input/output (overrides environment)')
    
    parser.add_argument('--input-s3-uri', type=str, 
                        default=None,  # Will be set later
                        help='S3 URI for input training data')
    
    parser.add_argument('--output-s3-uri', type=str,
                        default=None,  # Will be set later
                        help='S3 URI for output data')
    
    parser.add_argument('--target-column', type=str, default=TARGET_COLUMN,
                        help=f'Target column to predict (default: {TARGET_COLUMN})')
    
    parser.add_argument('--problem-type', type=str, default=PROBLEM_TYPE,
                        choices=['Regression', 'BinaryClassification', 'MulticlassClassification'],
                        help=f'Problem type (default: {PROBLEM_TYPE})')
    
    parser.add_argument('--metric-name', type=str, default=METRIC_NAME,
                        help=f'Metric to optimize (default: {METRIC_NAME})')
    
    parser.add_argument('--max-candidates', type=int, default=MAX_CANDIDATES,
                        help=f'Maximum number of model candidates (default: {MAX_CANDIDATES})')
    
    parser.add_argument('--max-runtime-hours', type=int, default=MAX_RUNTIME_HOURS,
                        help=f'Maximum runtime for AutoML job in hours (default: {MAX_RUNTIME_HOURS})')
    
    parser.add_argument('--max-runtime-per-job-minutes', type=int, default=MAX_RUNTIME_PER_JOB_MINUTES,
                        help=f'Maximum runtime per training job in minutes (default: {MAX_RUNTIME_PER_JOB_MINUTES})')
    
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                        help='Instance type for model deployment (default: ml.m5.large)')
    
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region for SageMaker resources (default: us-east-1)')
    
    # Add arguments for data splitting
    
    parser.add_argument('--split-type', type=str, default='random',
                        choices=['random', 'RANDOM', 'timestamp', 'TIMESTAMP'],
                        help='Method to split the data (default: random)')
    
    parser.add_argument('--timestamp-col', type=str, default=None,
                        help='Column containing timestamps for time-based split (required if split-type=TIMESTAMP)')
    
    parser.add_argument('--validation-fraction', type=float, default=0.2,
                        help='Fraction of data to use for validation (default: 0.2)')
    
    parser.add_argument('--use-automl-v2', action='store_true',
                        help='Use SageMaker Autopilot V2 API (required for timestamp-based splits)')
    
    args = parser.parse_args()
    
    # Normalize split type to uppercase
    args.split_type = args.split_type.upper()
    
    # Sanity check for timestamp-based split
    if args.split_type == 'TIMESTAMP' and not args.timestamp_col:
        parser.error("--timestamp-col is required when --split-type=TIMESTAMP")
    
    # Force use-automl-v2 for TIMESTAMP split
    if args.split_type == 'TIMESTAMP' and not args.use_automl_v2:
        logger.info("Enabling AutoML V2 API since it's required for timestamp-based splits")
        args.use_automl_v2 = True
    
    return args

def run_sagemaker_autopilot(args):
    """
    Run a SageMaker Autopilot job to train and deploy a model.
    
    This function:
    1. Initializes a SageMaker Autopilot job using either V1 or V2 API
    2. Waits for the job to complete
    3. Deploys the best model to an endpoint
    
    The function automatically determines whether to use the V1 or V2 API based on:
    - The --use-automl-v2 flag
    - The split type (TIMESTAMP requires V2)
    
    For V2 API with timestamp-based splits, the TabularJobConfig will include
    a TimestampAttributeName parameter.
    
    If V2 API fails and a timestamp split was not requested, it will fall back to
    the V1 API with a random split.
    
    Args:
        args: Command line arguments containing job configuration
        
    Returns:
        bool: True if the job completed successfully, False otherwise
    """
    try:
        # Initialize boto3 clients
        region = args.region  # Use the region provided in arguments or from .env
        if not region and os.environ.get('AWS_REGION'):
            region = os.environ.get('AWS_REGION')
            
        # Create boto3 session with credentials from .env
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        
        # Print the AWS credentials being used (masked for security)
        access_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
        if access_key:
            masked_access_key = access_key[:4] + '****' + access_key[-4:] if len(access_key) > 8 else '****'
            logger.info(f"Using AWS access key: {masked_access_key}")
        else:
            logger.warning("AWS_ACCESS_KEY_ID not found in environment variables")
            
        if secret_key:
            logger.info(f"AWS secret access key is present (masked for security)")
        else:
            logger.warning("AWS_SECRET_ACCESS_KEY not found in environment variables")
        
        sagemaker_client = session.client('sagemaker')
        sm_runtime = session.client('sagemaker-runtime')
        
        # Generate a unique job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        autopilot_job_name = f"conviction-automl-{timestamp}"
        endpoint_name = f"conviction-ai-endpoint-{timestamp}"
        
        logger.info(f"🚀 Launching AutoML job: {autopilot_job_name}")
        
        # Check if the dataset is in CSV format with UTF-8 encoding
        logger.info(f"Checking dataset format at {args.input_s3_uri}...")
        region_str = region if region else 'us-east-1'
        csv_dataset_uri = ensure_csv_dataset(args.input_s3_uri, region_str)
        
        if csv_dataset_uri:
            logger.info(f"Using CSV dataset at {csv_dataset_uri}")
            # Update the input URI to point to the CSV dataset
            args.input_s3_uri = csv_dataset_uri
        else:
            logger.error("Failed to find or convert to CSV dataset. SageMaker Autopilot requires CSV format with UTF-8 encoding.")
            logger.error("Please convert your dataset to CSV format with UTF-8 encoding and try again.")
            return False
        
        # Calculate job runtime in seconds
        max_runtime_seconds = args.max_runtime_hours * 60 * 60
        max_runtime_per_job_seconds = args.max_runtime_per_job_minutes * 60
        
        # Track which API version we're using for later status checks
        is_v2_job = args.use_automl_v2
        
        # Use different implementations based on whether V2 API is requested
        if args.use_automl_v2:
            logger.info("Using SageMaker Autopilot V2 API")
            
            # Prepare job completion criteria
            completion_criteria = {
                'MaxCandidates': args.max_candidates,
                'MaxRuntimePerTrainingJobInSeconds': max_runtime_per_job_seconds,
                'MaxAutoMLJobRuntimeInSeconds': max_runtime_seconds
            }
            
            # Configure TabularJobConfig
            tabular_job_config = {
                'TargetAttributeName': args.target_column,
                'ProblemType': args.problem_type,
                'CompletionCriteria': completion_criteria
            }
            
            # Handle timestamp-based split
            if args.split_type == "TIMESTAMP":
                logger.info(f"Configuring timestamp-based split with column: {args.timestamp_col}")
                # Add timestamp column to TabularJobConfig
                tabular_job_config['TimestampAttributeName'] = args.timestamp_col
            
            # Set up V2 job parameters
            auto_ml_job_v2_request = {
                'AutoMLJobName': autopilot_job_name,
                'AutoMLJobInputDataConfig': [
                    {
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': args.input_s3_uri
                            }
                        }
                    }
                ],
                'OutputDataConfig': {
                    'S3OutputPath': args.output_s3_uri
                },
                'RoleArn': args.role_arn,
                'AutoMLJobObjective': {
                    'MetricName': args.metric_name
                },
                'AutoMLProblemTypeConfig': {
                    'TabularJobConfig': tabular_job_config
                },
                'DataSplitConfig': {
                    'ValidationFraction': args.validation_fraction
                }
            }
            
            try:
                # Create AutoML job with V2 API
                logger.info("Creating AutoML job with V2 API")
                response = sagemaker_client.create_auto_ml_job_v2(**auto_ml_job_v2_request)
                logger.info(f"AutoML job created: {response['AutoMLJobArn']}")
            except Exception as e:
                logger.error(f"Error creating AutoML job with V2 API: {str(e)}")
                
                if args.split_type == "TIMESTAMP":
                    logger.error("Timestamp-based split requires AutoML V2 API. Cannot fall back to V1.")
                    return False
                
                logger.warning("Falling back to V1 API")
                is_v2_job = False
                
                # Fall through to V1 API code below
                # Specify input data - for V1 API
                input_data_config = [
                    {
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': args.input_s3_uri
                            }
                        },
                        'TargetAttributeName': args.target_column
                    }
                ]
                
                # Prepare the AutoML job config for V1 API
                automl_job_config = {
                    'CompletionCriteria': {
                        'MaxCandidates': args.max_candidates,
                        'MaxRuntimePerTrainingJobInSeconds': max_runtime_per_job_seconds,
                        'MaxAutoMLJobRuntimeInSeconds': max_runtime_seconds
                    },
                    'DataSplitConfig': {
                        'ValidationFraction': args.validation_fraction
                    }
                }
                
                # Create the AutoML job with V1 API
                logger.info(f"Creating AutoML job with V1 API (random split)")
                response = sagemaker_client.create_auto_ml_job(
                    AutoMLJobName=autopilot_job_name,
                    InputDataConfig=input_data_config,
                    OutputDataConfig={
                        'S3OutputPath': args.output_s3_uri
                    },
                    ProblemType=args.problem_type,
                    AutoMLJobObjective={
                        'MetricName': args.metric_name
                    },
                    AutoMLJobConfig=automl_job_config,
                    RoleArn=args.role_arn
                )
                logger.info(f"AutoML job created: {response['AutoMLJobArn']}")
        else:
            # Default to V1 API
            logger.info(f"Using SageMaker Autopilot V1 API with RANDOM split")
            
            # Specify input data - for V1 API
            input_data_config = [
                {
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': args.input_s3_uri
                        }
                    },
                    'TargetAttributeName': args.target_column
                }
            ]
            
            # Prepare the AutoML job config for V1 API
            automl_job_config = {
                'CompletionCriteria': {
                    'MaxCandidates': args.max_candidates,
                    'MaxRuntimePerTrainingJobInSeconds': max_runtime_per_job_seconds,
                    'MaxAutoMLJobRuntimeInSeconds': max_runtime_seconds
                },
                'DataSplitConfig': {
                    'ValidationFraction': args.validation_fraction
                }
            }
            
            # Create the AutoML job with V1 API
            logger.info(f"Creating AutoML job with V1 API (random split)")
            response = sagemaker_client.create_auto_ml_job(
                AutoMLJobName=autopilot_job_name,
                InputDataConfig=input_data_config,
                OutputDataConfig={
                    'S3OutputPath': args.output_s3_uri
                },
                ProblemType=args.problem_type,
                AutoMLJobObjective={
                    'MetricName': args.metric_name
                },
                AutoMLJobConfig=automl_job_config,
                RoleArn=args.role_arn
            )
            logger.info(f"AutoML job created: {response['AutoMLJobArn']}")
        
        # Wait for the AutoML job to complete
        logger.info("Waiting for AutoML job to complete. This may take a while...")
        
        # Poll for job status
        status = "InProgress"
        secondary_status = ""
        start_time = time.time()
        
        while status in ["InProgress", "Stopping"]:
            # Sleep before polling again to avoid API throttling
            time.sleep(30)
            
            # Get job status using appropriate API version
            if is_v2_job:
                response = sagemaker_client.describe_auto_ml_job_v2(AutoMLJobName=autopilot_job_name)
            else:
                response = sagemaker_client.describe_auto_ml_job(AutoMLJobName=autopilot_job_name)
            
            prev_status = status
            prev_secondary_status = secondary_status
            
            # Status fields are the same in both V1 and V2 APIs
            status = response['AutoMLJobStatus']
            secondary_status = response['AutoMLJobSecondaryStatus']
            
            # Only log when status changes
            if status != prev_status or secondary_status != prev_secondary_status:
                elapsed_time = int(time.time() - start_time)
                logger.info(f"Job status: {status} / {secondary_status} (Elapsed: {elapsed_time}s)")
            
            # If failed or stopped, break out of loop
            if status in ["Failed", "Stopped"]:
                failure_reason = response.get('FailureReason', 'Unknown')
                logger.error(f"AutoML job failed or stopped: {failure_reason}")
                return False
        
        # Job completed successfully
        elapsed_time = int(time.time() - start_time)
        logger.info(f"AutoML job completed in {elapsed_time}s")
        
        # Get best candidate
        best_candidate = response.get('BestCandidate')
        
        if not best_candidate:
            logger.error("No best candidate found in AutoML job results")
            return False
        
        # Log best candidate metrics
        metrics = best_candidate.get('FinalAutoMLJobObjectiveMetric', {})
        logger.info(f"Best candidate metrics: {metrics}")
        
        # Deploy the model
        logger.info(f"Deploying best model to endpoint: {endpoint_name}")
        
        # Create model
        model_name = f"conviction-ai-model-{timestamp}"
        
        # Structure is similar between V1 and V2 APIs for these properties
        inference_containers = best_candidate.get('InferenceContainers', [])
        model_data_url = best_candidate.get('ModelDataUrl')
        
        if not inference_containers:
            logger.error("No inference containers found in best candidate")
            return False
        
        # Create model
        logger.info(f"Creating model from best candidate: {model_name}")
        
        create_model_response = sagemaker_client.create_model(
            ModelName=model_name,
            Containers=inference_containers,
            ExecutionRoleArn=args.role_arn
        )
        logger.info(f"Model created: {create_model_response}")
        
        # Create endpoint configuration
        endpoint_config_name = f"{model_name}-config"
        logger.info(f"Creating endpoint configuration: {endpoint_config_name}")
        
        endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Default',
                    'ModelName': model_name,
                    'InstanceType': args.instance_type,
                    'InitialInstanceCount': 1
                }
            ]
        )
        logger.info(f"Endpoint configuration created: {endpoint_config_response}")
        
        # Create endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        create_endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint creation initiated: {create_endpoint_response}")
        
        # Wait for the endpoint to be in service
        logger.info("Waiting for endpoint deployment to complete...")
        
        endpoint_status = "Creating"
        endpoint_start_time = time.time()
        
        while endpoint_status == "Creating":
            # Sleep before polling again to avoid API throttling
            time.sleep(30)
            
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = response['EndpointStatus']
            
            if endpoint_status == "Failed":
                failure_reason = response.get('FailureReason', 'Unknown reason')
                logger.error(f"❌ Endpoint creation failed: {failure_reason}")
                return False
            
            endpoint_elapsed_time = int(time.time() - endpoint_start_time)
            logger.info(f"Endpoint status: {endpoint_status} (Elapsed: {endpoint_elapsed_time}s)")
        
        if endpoint_status != "InService":
            logger.error(f"❌ Endpoint did not deploy successfully. Status: {endpoint_status}")
            return False
        
        logger.info(f"✅ Deployed endpoint: {endpoint_name}")
        
        # Get metrics from best candidate
        metric_value = None
        candidate_name = None
        
        try:
            metric_value = best_candidate['FinalAutoMLJobObjectiveMetric']['Value']
            candidate_name = best_candidate.get('CandidateName', 'Unknown')
            logger.info(f"🎯 Best candidate: {candidate_name} with {args.metric_name}={metric_value:.4f}")
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not extract metrics from best candidate: {str(e)}")
        
        # Save endpoint information for future reference
        endpoint_info = {
            "job_name": autopilot_job_name,
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "best_candidate": candidate_name,
            "problem_type": args.problem_type,
            "target_column": args.target_column,
            "split_type": args.split_type,
            "validation_fraction": args.validation_fraction,
            "api_version": "v2" if is_v2_job else "v1",
            "created_at": datetime.now().isoformat()
        }
        
        # Add timestamp column if using TIMESTAMP split
        if args.split_type == "TIMESTAMP" and args.timestamp_col:
            endpoint_info["timestamp_column"] = args.timestamp_col
        
        # Add metric value if available
        if metric_value is not None:
            endpoint_info["metric_value"] = float(metric_value)
        
        with open("endpoint_info.json", "w") as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info(f"Endpoint information saved to endpoint_info.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in SageMaker AutoML process: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set the bucket and S3 URIs if not provided
    if args.bucket:
        s3_bucket = args.bucket
    else:
        # Try to find a bucket that has the correct permissions
        s3_bucket = os.environ.get('S3_BUCKET_NAME', 'convictionai-data')
        for candidate in S3_BUCKET_CANDIDATES:
            try:
                # Try to list the bucket to see if we have access
                session = boto3.Session(
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=args.region
                )
                s3 = session.client('s3')
                s3.list_objects_v2(Bucket=candidate, MaxKeys=1)
                logger.info(f"Successfully accessed bucket: {candidate}")
                s3_bucket = candidate
                break
            except Exception as e:
                logger.warning(f"Could not access bucket {candidate}: {str(e)}")
    
    logger.info(f"Using S3 bucket: {s3_bucket}")
    
    # Set input and output URIs if not provided
    if not args.input_s3_uri:
        args.input_s3_uri = f's3://{s3_bucket}/conviction-ai/clean/train_dataset/'
    
    if not args.output_s3_uri:
        args.output_s3_uri = f's3://{s3_bucket}/conviction-ai/automl-out/'
    
    logger.info(f"Input S3 URI: {args.input_s3_uri}")
    logger.info(f"Output S3 URI: {args.output_s3_uri}")
    
    success = run_sagemaker_autopilot(args)
    if success:
        logger.info("SageMaker AutoML process completed successfully")
    else:
        logger.error("SageMaker AutoML process failed")
        exit(1)
