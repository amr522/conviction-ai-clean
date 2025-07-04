#!/usr/bin/env python3
"""
Teardown specific SageMaker endpoint
"""
import argparse
import boto3
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def teardown_endpoint(endpoint_name, yes=False):
    """Delete specific SageMaker endpoint"""
    sagemaker = boto3.client('sagemaker')
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        
        logger.info(f"Found endpoint {endpoint_name} with status: {status}")
        
        if not yes:
            confirm = input(f"Delete endpoint {endpoint_name}? (y/N): ")
            if confirm.lower() != 'y':
                logger.info("Deletion cancelled")
                return False
        
        logger.info(f"🗑️ Deleting endpoint: {endpoint_name}")
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        
        logger.info(f"✅ Endpoint {endpoint_name} deletion initiated")
        return True
        
    except sagemaker.exceptions.ClientError as e:
        if 'ValidationException' in str(e):
            logger.info(f"Endpoint {endpoint_name} does not exist")
            return True
        else:
            logger.error(f"❌ Failed to delete endpoint: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Teardown SageMaker endpoint')
    parser.add_argument('--name', required=True, help='Endpoint name to delete')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    success = teardown_endpoint(args.name, args.yes)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
