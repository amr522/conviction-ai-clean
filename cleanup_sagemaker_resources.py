#!/usr/bin/env python3
"""
cleanup_sagemaker_resources.py - Script to clean up SageMaker resources

This script identifies and deletes SageMaker resources like endpoints, endpoint configs,
and models to avoid unnecessary AWS charges.
"""

import argparse
import boto3
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean up SageMaker resources')
    
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region for SageMaker resources (default: from env or us-east-1)')
    
    parser.add_argument('--prefix', type=str, default='conviction',
                        help='Prefix to filter resources (default: conviction)')
    
    parser.add_argument('--older-than-days', type=int, default=7,
                        help='Delete resources older than specified days (default: 7)')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='List resources without deleting them')
    
    parser.add_argument('--force', action='store_true',
                        help='Force deletion without confirmation')
    
    parser.add_argument('--include-endpoints', action='store_true',
                        help='Include endpoints in cleanup')
    
    parser.add_argument('--include-models', action='store_true',
                        help='Include models in cleanup')
    
    parser.add_argument('--include-endpoint-configs', action='store_true',
                        help='Include endpoint configs in cleanup')
    
    parser.add_argument('--include-all', action='store_true',
                        help='Include all resource types in cleanup')
    
    return parser.parse_args()

def list_resources(prefix=None, region=None):
    """
    List SageMaker resources, optionally filtered by prefix.
    
    Args:
        prefix: Optional prefix to filter resources
        region: AWS region
        
    Returns:
        Tuple of (endpoints, endpoint_configs, models)
    """
    try:
        # Create boto3 session with credentials from environment
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        
        # Create SageMaker client
        sm_client = session.client('sagemaker')
        
        # List endpoints
        endpoints = []
        endpoint_response = sm_client.list_endpoints()
        for endpoint in endpoint_response['Endpoints']:
            if not prefix or prefix in endpoint['EndpointName']:
                endpoints.append(endpoint['EndpointName'])
        
        # List endpoint configs
        configs = []
        config_response = sm_client.list_endpoint_configs()
        for config in config_response['EndpointConfigs']:
            if not prefix or prefix in config['EndpointConfigName']:
                configs.append(config['EndpointConfigName'])
        
        # List models
        models = []
        model_response = sm_client.list_models()
        for model in model_response['Models']:
            if not prefix or prefix in model['ModelName']:
                models.append(model['ModelName'])
        
        return endpoints, configs, models
        
    except Exception as e:
        logger.error(f"Error listing resources: {str(e)}")
        return [], [], []

def cleanup_resources(endpoints, configs, models, region=None, dry_run=False):
    """
    Delete the specified SageMaker resources.
    
    Args:
        endpoints: List of endpoint names to delete
        configs: List of endpoint config names to delete
        models: List of model names to delete
        region: AWS region
        dry_run: If True, only list resources without deleting
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create boto3 session with credentials from environment
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        
        # Create SageMaker client
        sm_client = session.client('sagemaker')
        
        # Delete endpoints
        for endpoint in endpoints:
            if dry_run:
                logger.info(f"Would delete endpoint: {endpoint}")
            else:
                logger.info(f"Deleting endpoint: {endpoint}")
                sm_client.delete_endpoint(EndpointName=endpoint)
        
        # Wait for endpoints to be deleted before deleting configs
        if endpoints and not dry_run:
            logger.info("Waiting for endpoints to be deleted...")
            time.sleep(30)
        
        # Delete endpoint configs
        for config in configs:
            if dry_run:
                logger.info(f"Would delete endpoint config: {config}")
            else:
                logger.info(f"Deleting endpoint config: {config}")
                sm_client.delete_endpoint_config(EndpointConfigName=config)
        
        # Delete models
        for model in models:
            if dry_run:
                logger.info(f"Would delete model: {model}")
            else:
                logger.info(f"Deleting model: {model}")
                sm_client.delete_model(ModelName=model)
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning up resources: {str(e)}")
        return False

def main():
    """Main function to clean up SageMaker resources."""
    args = parse_arguments()
    
    try:
        # Determine which resource types to include
        include_endpoints = args.include_endpoints or args.include_all
        include_configs = args.include_endpoint_configs or args.include_all
        include_models = args.include_models or args.include_all
        
        # If no specific resource types are selected, include all
        if not (include_endpoints or include_configs or include_models):
            include_endpoints = include_configs = include_models = True
            logger.info("No specific resource types selected, including all types")
        
        # List resources
        logger.info(f"Listing SageMaker resources in {args.region} with prefix '{args.prefix}'...")
        all_endpoints, all_configs, all_models = list_resources(args.prefix, args.region)
        
        logger.info(f"Found {len(all_endpoints)} endpoints, {len(all_configs)} endpoint configs, {len(all_models)} models")
        
        # Filter resources to delete
        endpoints_to_delete = all_endpoints if include_endpoints else []
        configs_to_delete = all_configs if include_configs else []
        models_to_delete = all_models if include_models else []
        
        # Check if there are resources to delete
        if not (endpoints_to_delete or configs_to_delete or models_to_delete):
            logger.info("No resources to delete")
            return True
        
        # Confirm deletion if not forced
        if not args.force and not args.dry_run:
            confirmation = input(f"Delete {len(endpoints_to_delete)} endpoints, {len(configs_to_delete)} configs, {len(models_to_delete)} models? (y/n): ")
            if confirmation.lower() != 'y':
                logger.info("Cleanup cancelled")
                return False
        
        # Clean up resources
        success = cleanup_resources(
            endpoints_to_delete, 
            configs_to_delete, 
            models_to_delete, 
            args.region, 
            args.dry_run
        )
        
        if success:
            if args.dry_run:
                logger.info("Dry run completed successfully")
            else:
                logger.info("Cleanup completed successfully")
        else:
            logger.error("Cleanup failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in cleanup process: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)