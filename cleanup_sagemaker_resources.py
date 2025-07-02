#!/usr/bin/env python3
"""
cleanup_sagemaker_resources.py - Clean up unused SageMaker resources

This script:
1. Identifies and deletes endpoints, training jobs, tuning jobs, and models > 12 hours old
2. Ensures no resources are billing after training is complete
"""

import os
import sys
import argparse
import logging
import boto3
from datetime import datetime, timedelta, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs/cleanup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)

def clean_up_endpoints(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker endpoints older than the cutoff time"""
    logger.info("Cleaning up SageMaker endpoints")
    
    try:
        # List all endpoints
        endpoints = sagemaker_client.list_endpoints()
        
        count = 0
        for endpoint in endpoints.get('Endpoints', []):
            endpoint_name = endpoint['EndpointName']
            creation_time = endpoint['CreationTime']
            
            if creation_time < cutoff_time:
                logger.info(f"Found old endpoint: {endpoint_name} (created {creation_time})")
                
                if not dry_run:
                    logger.info(f"Deleting endpoint: {endpoint_name}")
                    try:
                        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would delete endpoint: {endpoint_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} endpoints")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up endpoints: {e}")
        return 0

def clean_up_endpoint_configs(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker endpoint configs older than the cutoff time"""
    logger.info("Cleaning up SageMaker endpoint configs")
    
    try:
        # List all endpoint configs
        endpoint_configs = sagemaker_client.list_endpoint_configs()
        
        count = 0
        for config in endpoint_configs.get('EndpointConfigs', []):
            config_name = config['EndpointConfigName']
            creation_time = config['CreationTime']
            
            if creation_time < cutoff_time:
                logger.info(f"Found old endpoint config: {config_name} (created {creation_time})")
                
                if not dry_run:
                    logger.info(f"Deleting endpoint config: {config_name}")
                    try:
                        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete endpoint config {config_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would delete endpoint config: {config_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} endpoint configs")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up endpoint configs: {e}")
        return 0

def clean_up_models(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker models older than the cutoff time"""
    logger.info("Cleaning up SageMaker models")
    
    try:
        # List all models
        models = sagemaker_client.list_models()
        
        count = 0
        for model in models.get('Models', []):
            model_name = model['ModelName']
            creation_time = model['CreationTime']
            
            if creation_time < cutoff_time:
                logger.info(f"Found old model: {model_name} (created {creation_time})")
                
                if not dry_run:
                    logger.info(f"Deleting model: {model_name}")
                    try:
                        sagemaker_client.delete_model(ModelName=model_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete model {model_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would delete model: {model_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} models")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up models: {e}")
        return 0

def clean_up_training_jobs(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker training jobs older than the cutoff time"""
    logger.info("Cleaning up SageMaker training jobs")
    
    try:
        # List all training jobs
        training_jobs = sagemaker_client.list_training_jobs()
        
        count = 0
        for job in training_jobs.get('TrainingJobSummaries', []):
            job_name = job['TrainingJobName']
            creation_time = job['CreationTime']
            status = job['TrainingJobStatus']
            
            if creation_time < cutoff_time and status in ['InProgress', 'Stopping']:
                logger.info(f"Found old training job: {job_name} (created {creation_time}, status {status})")
                
                if not dry_run:
                    logger.info(f"Stopping training job: {job_name}")
                    try:
                        sagemaker_client.stop_training_job(TrainingJobName=job_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to stop training job {job_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would stop training job: {job_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} training jobs")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up training jobs: {e}")
        return 0

def clean_up_processing_jobs(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker processing jobs older than the cutoff time"""
    logger.info("Cleaning up SageMaker processing jobs")
    
    try:
        # List all processing jobs
        processing_jobs = sagemaker_client.list_processing_jobs()
        
        count = 0
        for job in processing_jobs.get('ProcessingJobSummaries', []):
            job_name = job['ProcessingJobName']
            creation_time = job['CreationTime']
            status = job['ProcessingJobStatus']
            
            if creation_time < cutoff_time and status in ['InProgress', 'Stopping']:
                logger.info(f"Found old processing job: {job_name} (created {creation_time}, status {status})")
                
                if not dry_run:
                    logger.info(f"Stopping processing job: {job_name}")
                    try:
                        sagemaker_client.stop_processing_job(ProcessingJobName=job_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to stop processing job {job_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would stop processing job: {job_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} processing jobs")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up processing jobs: {e}")
        return 0

def clean_up_transform_jobs(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker transform jobs older than the cutoff time"""
    logger.info("Cleaning up SageMaker transform jobs")
    
    try:
        # List all transform jobs
        transform_jobs = sagemaker_client.list_transform_jobs()
        
        count = 0
        for job in transform_jobs.get('TransformJobSummaries', []):
            job_name = job['TransformJobName']
            creation_time = job['CreationTime']
            status = job['TransformJobStatus']
            
            if creation_time < cutoff_time and status in ['InProgress', 'Stopping']:
                logger.info(f"Found old transform job: {job_name} (created {creation_time}, status {status})")
                
                if not dry_run:
                    logger.info(f"Stopping transform job: {job_name}")
                    try:
                        sagemaker_client.stop_transform_job(TransformJobName=job_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to stop transform job {job_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would stop transform job: {job_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} transform jobs")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up transform jobs: {e}")
        return 0

def clean_up_hyperparameter_tuning_jobs(sagemaker_client, cutoff_time, dry_run=False):
    """Clean up SageMaker hyperparameter tuning jobs older than the cutoff time"""
    logger.info("Cleaning up SageMaker hyperparameter tuning jobs")
    
    try:
        # List all hyperparameter tuning jobs
        tuning_jobs = sagemaker_client.list_hyper_parameter_tuning_jobs()
        
        count = 0
        for job in tuning_jobs.get('HyperParameterTuningJobSummaries', []):
            job_name = job['HyperParameterTuningJobName']
            creation_time = job['CreationTime']
            status = job['HyperParameterTuningJobStatus']
            
            if creation_time < cutoff_time and status in ['InProgress', 'Stopping']:
                logger.info(f"Found old hyperparameter tuning job: {job_name} (created {creation_time}, status {status})")
                
                if not dry_run:
                    logger.info(f"Stopping hyperparameter tuning job: {job_name}")
                    try:
                        sagemaker_client.stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to stop hyperparameter tuning job {job_name}: {e}")
                else:
                    logger.info(f"[DRY RUN] Would stop hyperparameter tuning job: {job_name}")
                    count += 1
        
        logger.info(f"Cleaned up {count} hyperparameter tuning jobs")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up hyperparameter tuning jobs: {e}")
        return 0

def main():
    """Main function to clean up SageMaker resources"""
    parser = argparse.ArgumentParser(description='Clean up unused SageMaker resources')
    parser.add_argument('--hours', type=int, default=12,
                        help='Delete resources older than this many hours')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do not actually delete resources, just report what would be deleted')
    args = parser.parse_args()
    
    # Ensure log directory exists
    ensure_directory('pipeline_logs')
    
    # Calculate cutoff time (with timezone awareness to match AWS timestamps)
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    logger.info(f"Cleaning up resources older than {cutoff_time}")
    
    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    
    # Clean up resources
    total_count = 0
    
    # Clean up endpoints
    total_count += clean_up_endpoints(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up endpoint configs
    total_count += clean_up_endpoint_configs(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up models
    total_count += clean_up_models(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up training jobs
    total_count += clean_up_training_jobs(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up processing jobs
    total_count += clean_up_processing_jobs(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up transform jobs
    total_count += clean_up_transform_jobs(sagemaker_client, cutoff_time, args.dry_run)
    
    # Clean up hyperparameter tuning jobs
    total_count += clean_up_hyperparameter_tuning_jobs(sagemaker_client, cutoff_time, args.dry_run)
    
    logger.info(f"Total resources cleaned up: {total_count}")
    
    if args.dry_run:
        logger.info("This was a dry run. No resources were actually deleted.")

if __name__ == "__main__":
    main()
