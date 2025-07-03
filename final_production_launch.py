#!/usr/bin/env python3
"""
Final production HPO launcher - simplified and focused
"""

import os
import sys
import time
import logging
import argparse
import boto3
import sagemaker
from sagemaker.tuner import HyperparameterTuner
from sagemaker.estimator import Estimator
from sagemaker.parameter import ContinuousParameter, IntegerParameter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Launch production HPO job"""
    parser = argparse.ArgumentParser(description='Launch production HPO job')
    parser.add_argument('--dry-run', action='store_true', help='Skip HPO job creation, log what would be done')
    args = parser.parse_args()
    
    try:
        pinned_data = os.environ.get('PINNED_DATA_S3')
        if not pinned_data:
            logger.error("PINNED_DATA_S3 environment variable not set")
            return False
        
        logger.info(f"Using pinned dataset: {pinned_data}")
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = Estimator(
            image_uri='811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1',
            entry_point='xgboost_train.py',
            role='arn:aws:iam::773934887314:role/SageMakerExecutionRole',
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            output_path='s3://hpo-bucket-773934887314/models/',
            sagemaker_session=session
        )
        
        hyperparameter_ranges = {
            'max_depth': IntegerParameter(3, 10),
            'eta': ContinuousParameter(0.01, 0.3),
            'subsample': ContinuousParameter(0.5, 1.0),
            'colsample_bytree': ContinuousParameter(0.5, 1.0),
        }
        
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=50,
            max_parallel_jobs=4
        )
        
        timestamp = int(time.time())
        job_name = f"prod-hpo-{timestamp}"
        
        logger.info(f"Launching production HPO job: {job_name}")
        
        if args.dry_run:
            logger.info(f"🔍 [dry-run] Would launch HPO with dataset: {pinned_data}")
            logger.info(f"🔍 [dry-run] Job name would be: {job_name}")
            return True
        
        # Launch tuning job
        tuner.fit({'training': pinned_data}, job_name=job_name)
        
        job_arn = f"arn:aws:sagemaker:us-east-1:773934887314:hyper-parameter-tuning-job/{job_name}"
        
        logger.info(f"✅ Production HPO job launched successfully!")
        logger.info(f"Job Name: {job_name}")
        logger.info(f"Job ARN: {job_arn}")
        logger.info(f"Dataset: {pinned_data}")
        
        with open('production_hpo_results.txt', 'w') as f:
            f.write(f"SUCCESS\n")
            f.write(f"Job Name: {job_name}\n")
            f.write(f"Job ARN: {job_arn}\n")
            f.write(f"Dataset: {pinned_data}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        
        print(f"SUCCESS: {job_arn}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to launch production HPO job: {e}")
        with open('production_hpo_results.txt', 'w') as f:
            f.write(f"FAILED\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
