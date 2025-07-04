#!/usr/bin/env python3
"""
AWS HPO Launch Script for AAPL and Full Universe

This script launches two staged HPO jobs on AWS SageMaker:
1. First on AAPL only (as a test)
2. Then on the full filtered universe

It reuses the same data source from the last successful HPO job with 138 completed models.
"""

import os
import sys
import time
import logging
import argparse
import boto3
import sagemaker
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.xgboost import XGBoost
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
S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/'
# XGBOOST_IMAGE_URI = '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3'

def get_input_data_s3(cli_arg=None):
    """Get input data S3 URI with proper precedence order"""
    if cli_arg:
        logger.info(f"🔗 Using CLI argument: {cli_arg}")
        return cli_arg
    
    pinned_data = os.environ.get('PINNED_DATA_S3')
    if pinned_data:
        logger.info(f"🔗 Using PINNED_DATA_S3 from environment: {pinned_data}")
        return pinned_data
    
    last_data_s3 = os.environ.get('LAST_DATA_S3')
    if last_data_s3:
        logger.info(f"🔗 Using LAST_DATA_S3 from environment: {last_data_s3}")
        return last_data_s3
    
    dataset_file = "last_dataset_uri.txt"
    if os.path.exists(dataset_file):
        try:
            with open(dataset_file, 'r') as f:
                pinned_data = f.read().strip()
            if pinned_data:
                logger.info(f"🔗 Using pinned dataset from {dataset_file}: {pinned_data}")
                return pinned_data
        except Exception as e:
            logger.warning(f"Failed to read {dataset_file}: {e}")
    
    logger.warning("No pinned dataset found. Using default data prefix.")
    return f's3://{BUCKET}/data/'

def validate_s3_uri(s3_uri):
    """Validate S3 URI format and accessibility with startup validation"""
    import re
    
    if not s3_uri:
        logger.error("❌ Empty S3 URI provided")
        return False
    
    if not re.match(r"^s3://[^/]+/.+", s3_uri):
        logger.error(f"❌ Invalid S3 URI format: {s3_uri}")
        sys.exit(f"❌ Invalid S3 URI: {s3_uri}")
    
    logger.info(f"🔗 Using dataset: {s3_uri}")
    
    try:
        s3_client = boto3.client('s3')
        
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        if key.endswith('.csv'):
            s3_client.head_object(Bucket=bucket, Key=key)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            if 'Contents' not in response:
                logger.error(f"No objects found at S3 URI: {s3_uri}")
                return False
        
        logger.info(f"✅ S3 URI validated: {s3_uri}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate S3 URI {s3_uri}: {e}")
        return False
def get_hyperparameter_ranges():
    """Define hyperparameter ranges for tuning - only XGBoost native parameters"""
    return {
        # XGBoost model parameters (tunable in script mode)
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.3),
        'min_child_weight': IntegerParameter(1, 10),
        'subsample': ContinuousParameter(0.5, 1.0),
        'gamma': ContinuousParameter(0, 10),
        'alpha': ContinuousParameter(0, 10),
        'lambda': ContinuousParameter(0, 10),
        'colsample_bytree': ContinuousParameter(0.5, 1.0),
    }

def launch_aapl_hpo(input_data_s3=None, dry_run=False):
    """Launch AWS SageMaker HPO job for AAPL only"""
    try:
        logger.info("Launching AWS SageMaker HPO job for AAPL only")
        
        training_data = get_input_data_s3(input_data_s3)
        
        if not validate_s3_uri(training_data):
            logger.error("❌ Invalid training data S3 URI")
            return None
        
        if dry_run:
            logger.info("🧪 DRY RUN MODE - No SageMaker calls will be made")
            job_name = f"options-hpo-aapl-{int(time.time())}-dry-run"
            logger.info(f"✅ DRY RUN: Would launch AAPL HPO job: {job_name}")
            logger.info(f"✅ DRY RUN: Would use training data: {training_data}")
            return job_name
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = XGBoost(
            entry_point='xgboost_train.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            framework_version='1.0-1',
            hyperparameters={
                'symbol': 'AAPL',
                'model': 'xgb',
                'debug': True,
                'iv_rank_window': 30,
                'iv_rank_weight': 0.5,
                'term_slope_window': 15,
                'term_slope_weight': 0.5,
                'oi_window': 15,
                'oi_weight': 0.5,
                'theta_window': 15,
                'theta_weight': 0.5,
                'vix_mom_window': 10,
                'vix_regime_thresh': 25.0,
                'event_lag': 2,
                'event_lead': 2,
                'news_threshold': 0.05,
                'lookback_window': 5,
                'reuters_weight': 1.0,
                'sa_weight': 1.0,
            },
            sagemaker_session=session
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'validation:auc', 'Regex': 'validation-auc:([0-9\\.]+)'}],
            max_jobs=20,
            max_parallel_jobs=3
        )
        
        # Launch tuning job
        job_name = f"hpo-aapl-{int(time.time())}"
        tuner.fit({
            'train': TrainingInput(
                s3_data=training_data,
                content_type='text/csv',
                input_mode='File'
            )
        }, job_name=job_name)
        
        logger.info(f"Successfully launched AAPL HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch AAPL HPO job: {e}")
        return None

def launch_full_universe_hpo(input_data_s3=None, dry_run=False):
    """Launch AWS SageMaker HPO job for the full filtered universe"""
    try:
        logger.info("Launching AWS SageMaker HPO job for full filtered universe")
        
        training_data = get_input_data_s3(input_data_s3)
        
        if not validate_s3_uri(training_data):
            logger.error("❌ Invalid training data S3 URI")
            return None
        
        if dry_run:
            logger.info("🧪 DRY RUN MODE - No SageMaker calls will be made")
            job_name = f"options-hpo-full-universe-{int(time.time())}-dry-run"
            logger.info(f"✅ DRY RUN: Would launch full universe HPO job: {job_name}")
            logger.info(f"✅ DRY RUN: Would use training data: {training_data}")
            return job_name
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = XGBoost(
            entry_point='xgboost_train.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            framework_version='1.0-1',
            hyperparameters={
                'symbol': 'ALL',  # Special keyword for all filtered symbols
                'model': 'xgb',
                'apply_universe_filter': True,
                'iv_rank_window': 30,
                'iv_rank_weight': 0.5,
                'term_slope_window': 15,
                'term_slope_weight': 0.5,
                'oi_window': 15,
                'oi_weight': 0.5,
                'theta_window': 15,
                'theta_weight': 0.5,
                'vix_mom_window': 10,
                'vix_regime_thresh': 25.0,
                'event_lag': 2,
                'event_lead': 2,
                'news_threshold': 0.05,
                'lookback_window': 5,
                'reuters_weight': 1.0,
                'sa_weight': 1.0,
            },
            sagemaker_session=session
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'validation:auc', 'Regex': 'validation-auc:([0-9\\.]+)'}],
            max_jobs=50,
            max_parallel_jobs=3
        )
        
        # Launch tuning job
        job_name = f"hpo-full-{int(time.time())}"
        tuner.fit({
            'train': TrainingInput(
                s3_data=training_data,
                content_type='text/csv',
                input_mode='File'
            )
        }, job_name=job_name)
        
        logger.info(f"Successfully launched full universe HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch full universe HPO job: {e}")
        return None

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description='Launch AWS SageMaker HPO jobs')
    parser.add_argument('--input-data-s3', type=str, help='S3 URI for training data (highest precedence)')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode without making SageMaker calls')
    parser.add_argument('--job-type', choices=['aapl', 'full', 'both'], default='both', help='Type of HPO job to launch')
    
    args = parser.parse_args()
    
    logger.info("Starting HPO pipeline...")
    
    if args.job_type in ['aapl', 'both']:
        aapl_job = launch_aapl_hpo(args.input_data_s3, args.dry_run)
        if aapl_job:
            logger.info(f"AAPL HPO job launched: {aapl_job}")
        else:
            logger.error("Failed to launch AAPL HPO job")
            if not args.dry_run:
                sys.exit(1)
    
    if args.job_type in ['full', 'both']:
        full_job = launch_full_universe_hpo(args.input_data_s3, args.dry_run)
        if full_job:
            logger.info(f"Full universe HPO job launched: {full_job}")
        else:
            logger.error("Failed to launch full universe HPO job")
            if not args.dry_run:
                sys.exit(1)
    
    logger.info("All HPO jobs completed successfully!")

if __name__ == "__main__":
    main()
