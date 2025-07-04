#!/usr/bin/env python3
"""
Check Current HPO Job Status
Quick status check for all running jobs
"""

import boto3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        sagemaker = boto3.client('sagemaker')
        
        logger.info("=== Current HPO Jobs Status ===")
        response = sagemaker.list_hyper_parameter_tuning_jobs(
            StatusEquals='InProgress',
            MaxResults=10
        )
        
        if response['HyperParameterTuningJobSummaries']:
            for job in response['HyperParameterTuningJobSummaries']:
                logger.info(f"Job: {job['HyperParameterTuningJobName']}")
                logger.info(f"Status: {job['HyperParameterTuningJobStatus']}")
                logger.info(f"Created: {job['CreationTime']}")
                logger.info("---")
        else:
            logger.info("No InProgress HPO jobs found")
        
        logger.info("=== SageMaker SDK Check ===")
        import sagemaker
        logger.info(f"SageMaker SDK version: {sagemaker.__version__}")
        
        logger.info("=== AWS Credentials Check ===")
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"Account: {identity['Account']}")
        logger.info(f"User: {identity['Arn']}")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")

if __name__ == "__main__":
    main()
