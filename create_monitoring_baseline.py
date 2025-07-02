#!/usr/bin/env python3
"""
create_monitoring_baseline.py - Create baseline statistics for model monitoring

This script:
1. Captures input and output data from a SageMaker endpoint
2. Generates statistics and constraints for model monitoring
3. Uploads the baseline to S3 for use in monitoring
"""

import os
import sys
import argparse
import logging
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs/baseline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)

def capture_endpoint_data(endpoint_name, sample_count=500):
    """Capture data from a SageMaker endpoint for baseline creation"""
    logger.info(f"Capturing data from endpoint {endpoint_name}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # Create a data capture job
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        capture_job_name = f"{endpoint_name}-capture-{timestamp}"
        
        capture_config = {
            "EndpointName": endpoint_name,
            "VariantName": "AllTraffic",
            "SamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')}/data-capture/{endpoint_name}/"
        }
        
        # Update the endpoint to enable data capture
        endpoint_config = sm_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_config['EndpointConfigName']
        
        # Get the current config
        current_config = sm_client.describe_endpoint_config(EndpointConfigName=config_name)
        
        # Create a new config with data capture enabled
        new_config_name = f"{config_name}-capture-{timestamp}"
        
        # Create new config with data capture
        create_args = {
            "EndpointConfigName": new_config_name,
            "ProductionVariants": current_config['ProductionVariants'],
            "DataCaptureConfig": {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": capture_config["DestinationS3Uri"],
                "CaptureOptions": [
                    {
                        "CaptureMode": "Input"
                    },
                    {
                        "CaptureMode": "Output"
                    }
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": [
                        "text/csv"
                    ],
                    "JsonContentTypes": [
                        "application/json"
                    ]
                }
            }
        }
        
        # Add tags if present in current config
        if 'Tags' in current_config:
            create_args["Tags"] = current_config['Tags']
        
        # Create the new config
        sm_client.create_endpoint_config(**create_args)
        
        # Update the endpoint to use the new config
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name
        )
        
        logger.info(f"Endpoint updated to capture data to {capture_config['DestinationS3Uri']}")
        
        # Wait for the endpoint to update
        sm_client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
        
        # Generate sample invocations
        logger.info(f"Generating {sample_count} sample invocations")
        
        # Load some sample data for invocations
        # This is a placeholder - you would need to replace with actual loading of test data
        s3_client = boto3.client('s3')
        bucket = os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')
        
        # Try to find the test dataset
        try:
            test_data_key = None
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix='56_stocks/test'):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.csv'):
                        test_data_key = obj['Key']
                        break
                if test_data_key:
                    break
            
            if test_data_key:
                # Download the test data
                local_test_file = "/tmp/test_data.csv"
                s3_client.download_file(bucket, test_data_key, local_test_file)
                
                # Load the test data
                test_data = pd.read_csv(local_test_file)
                
                # Create a runtime client for endpoint invocation
                runtime_client = boto3.client('sagemaker-runtime')
                
                # Invoke the endpoint with samples
                for i in range(min(sample_count, len(test_data))):
                    # Convert the row to CSV format
                    row = test_data.iloc[i]
                    payload = ','.join([str(val) for val in row.values])
                    
                    # Invoke the endpoint
                    response = runtime_client.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType='text/csv',
                        Body=payload
                    )
                    
                    # Small delay to avoid overwhelming the endpoint
                    if i % 10 == 0:
                        logger.info(f"Processed {i} samples")
                
                logger.info(f"Successfully invoked endpoint with {min(sample_count, len(test_data))} samples")
            else:
                logger.error(f"Could not find test data in S3 bucket {bucket}")
                return None
        except Exception as e:
            logger.error(f"Error loading or using test data: {e}")
            return None
        
        # Return the S3 URI where data is captured
        return capture_config["DestinationS3Uri"]
    except Exception as e:
        logger.error(f"Failed to capture endpoint data: {e}")
        return None

def create_baseline(endpoint_name, capture_data_uri=None, output_s3_uri=None):
    """Create baseline statistics and constraints for model monitoring"""
    logger.info(f"Creating baseline for endpoint {endpoint_name}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # Set default output URI if not provided
        if not output_s3_uri:
            output_s3_uri = f"s3://{os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')}/baselines/{endpoint_name}/"
        
        # If capture data URI not provided, try to find the most recent
        if not capture_data_uri:
            # List data capture locations
            s3_client = boto3.client('s3')
            bucket = os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')
            prefix = f"data-capture/{endpoint_name}/"
            
            # Try to find captured data
            latest_prefix = None
            latest_time = None
            
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
                for prefix_obj in page.get('CommonPrefixes', []):
                    prefix_path = prefix_obj.get('Prefix')
                    if prefix_path:
                        try:
                            # Extract timestamp from prefix
                            date_part = prefix_path.split('/')[-2]
                            prefix_time = datetime.strptime(date_part, "%Y/%m/%d")
                            
                            if latest_time is None or prefix_time > latest_time:
                                latest_time = prefix_time
                                latest_prefix = prefix_path
                        except:
                            continue
            
            if latest_prefix:
                capture_data_uri = f"s3://{bucket}/{latest_prefix}"
                logger.info(f"Found captured data at {capture_data_uri}")
            else:
                logger.error(f"Could not find captured data for endpoint {endpoint_name}")
                return None
        
        # Create the baseline processing job
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        baseline_job_name = f"{endpoint_name}-baseline-{timestamp}"
        
        # Create the baseline job
        sm_client.create_processing_job(
            ProcessingJobName=baseline_job_name,
            ProcessingInputs=[
                {
                    "InputName": "endpoint-data",
                    "S3Input": {
                        "S3Uri": capture_data_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "baseline",
                        "S3Output": {
                            "S3Uri": output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 20
                }
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": 3600
            },
            AppSpecification={
                "ImageUri": f"{boto3.client('sts').get_caller_identity().get('Account')}.dkr.ecr.{boto3.session.Session().region_name}.amazonaws.com/sagemaker-model-monitor-analyzer"
            },
            RoleArn=os.environ.get("SAGEMAKER_ROLE_ARN"),
            ProcessingJobName=baseline_job_name
        )
        
        # Wait for the processing job to complete
        sm_client.get_waiter('processing_job_completed').wait(ProcessingJobName=baseline_job_name)
        
        logger.info(f"Successfully created baseline at {output_s3_uri}")
        return output_s3_uri
    except Exception as e:
        logger.error(f"Failed to create baseline: {e}")
        return None

def main():
    """Main function to parse arguments and execute commands"""
    parser = argparse.ArgumentParser(description="Create baseline for model monitoring")
    
    parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    parser.add_argument("--capture-data", help="S3 URI to existing captured data (if available)")
    parser.add_argument("--output-s3-uri", help="S3 URI where to store the baseline")
    parser.add_argument("--sample-count", type=int, default=500, help="Number of samples to capture if needed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure the logs directory exists
    ensure_directory("pipeline_logs")
    
    # Capture data if needed
    capture_data_uri = args.capture_data
    if not capture_data_uri:
        logger.info("No capture data provided, capturing new data")
        capture_data_uri = capture_endpoint_data(args.endpoint_name, args.sample_count)
        
        if not capture_data_uri:
            logger.error("Failed to capture data, exiting")
            sys.exit(1)
    
    # Create baseline
    baseline_uri = create_baseline(args.endpoint_name, capture_data_uri, args.output_s3_uri)
    
    if baseline_uri:
        logger.info(f"Baseline created successfully at {baseline_uri}")
        sys.exit(0)
    else:
        logger.error("Failed to create baseline")
        sys.exit(1)

if __name__ == "__main__":
    main()
