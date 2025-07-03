#!/usr/bin/env python3
"""
AWS HPO Launch Script for AAPL and Full Universe

This script launches two staged HPO jobs on AWS SageMaker:
1. First on AAPL only (as a test)
2. Then on the full filtered universe
"""

import os
import sys
import time
import logging
import boto3
import sagemaker
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aws_hpo_launch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# AWS Configuration
ROLE_ARN = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'
BUCKET = 'hpo-bucket-773934887314'
S3_DATA_PREFIX = f's3://{BUCKET}/56_stocks/2025-07-03-05-41-43/'
S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/'

def get_hyperparameter_ranges():
    """Define hyperparameter ranges for RandomForest tuning"""
    return {
        'n_estimators': IntegerParameter(50, 300),
        'max_depth': IntegerParameter(3, 15),
        'min_samples_split': IntegerParameter(2, 10),
        'min_samples_leaf': IntegerParameter(1, 5),
        'max_features': CategoricalParameter(['sqrt', 'log2', 'auto']),
        'feature_fraction': ContinuousParameter(0.1, 1.0),
        'bagging_fraction': ContinuousParameter(0.1, 1.0),
    }

def launch_aapl_hpo():
    """Launch AWS SageMaker HPO job for AAPL only"""
    try:
        logger.info("Launching AWS SageMaker HPO job for AAPL only")
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = SKLearn(
            entry_point='sagemaker_train.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            framework_version='0.23-1',
            py_version='py3',
            hyperparameters={},
            sagemaker_session=session,
            output_path=S3_OUTPUT_PREFIX
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'validation:auc', 'Regex': 'validation:auc=([0-9\\.]+)'}],
            max_jobs=20,
            max_parallel_jobs=4
        )
        
        # Launch tuning job
        job_name = f"cf-rf-aapl-{int(time.time())}"
        tuner.fit({
            'training': TrainingInput(f'{S3_DATA_PREFIX}train.csv', content_type='text/csv')
        }, job_name=job_name)
        
        logger.info(f"Successfully launched AAPL HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch AAPL HPO job: {e}")
        return None

def launch_full_universe_hpo():
    """Launch AWS SageMaker HPO job for the full filtered universe"""
    try:
        logger.info("Launching AWS SageMaker HPO job for full filtered universe")
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = SKLearn(
            entry_point='sagemaker_train.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            framework_version='0.23-1',
            py_version='py3',
            hyperparameters={},
            sagemaker_session=session,
            output_path=S3_OUTPUT_PREFIX
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'validation:auc', 'Regex': 'validation:auc=([0-9\\.]+)'}],
            max_jobs=50,
            max_parallel_jobs=4
        )
        
        # Launch tuning job
        job_name = f"cf-rf-full-{int(time.time())}"
        tuner.fit({
            'training': TrainingInput(f'{S3_DATA_PREFIX}train.csv', content_type='text/csv')
        }, job_name=job_name)
        
        logger.info(f"Successfully launched full universe HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch full universe HPO job: {e}")
        return None

def main():
    """Main function to launch AWS HPO jobs"""
    logger.info("Verification complete – launching AWS HPO on AAPL for test.")
    
    # Step 1: Launch AAPL HPO job
    aapl_job = launch_aapl_hpo()
    
    if not aapl_job:
        logger.error("Failed to launch AAPL HPO job. Aborting.")
        sys.exit(1)
    
    # Wait for AAPL job to complete (or at least start successfully)
    logger.info("Waiting 5 minutes for AAPL job to initialize...")
    time.sleep(300)  # Wait 5 minutes
    
    # Step 2: Launch full universe HPO job
    universe_job = launch_full_universe_hpo()
    
    if not universe_job:
        logger.error("Failed to launch full universe HPO job.")
        sys.exit(1)
    
    logger.info("Successfully launched both HPO jobs:")
    logger.info(f"AAPL job: {aapl_job}")
    logger.info(f"Full universe job: {universe_job}")
    logger.info("Monitor progress in the AWS SageMaker console.")

if __name__ == "__main__":
    main()
