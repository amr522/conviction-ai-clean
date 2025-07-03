#!/usr/bin/env python3
"""
AWS HPO Launch Script for AAPL and Full Universe

This script launches two staged HPO jobs on AWS SageMaker:
1. First on AAPL only (as a test)
2. Then on the full filtered universe

It reuses the same data source from the last successful HPO job using the following priority:
1. Command line argument: --input-data-s3
2. Environment variable: PINNED_DATA_S3 (set by scripts/get_last_hpo_dataset.sh)
3. File content: last_dataset_uri.txt (created by scripts/get_last_hpo_dataset.sh)
4. Environment variable: LAST_DATA_S3 (legacy, for backward compatibility)
5. Default fallback path (if forced with --force-default-data or if no other source available)

Before running this script, execute:
    source ./scripts/get_last_hpo_dataset.sh
to automatically set the PINNED_DATA_S3 environment variable from the most recent
successful HPO job with sufficient completed training jobs.
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
from sagemaker.estimator import Estimator

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Launch AWS SageMaker HPO jobs with consistent data source')
parser.add_argument('--input-data-s3', type=str, help='S3 URI for input data', 
                    default=None)  # Will check environment variables in order of priority
parser.add_argument('--dry-run', action='store_true', help='Print config without launching jobs')
parser.add_argument('--force-default-data', action='store_true', help='Use default data path even if PINNED_DATA_S3 is set')
parser.add_argument('--target-completed', type=int, default=138, 
                    help='Target number of completed jobs for HPO job selection (only used if getting a new data source)')
args = parser.parse_args()

# AWS Configuration
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
BUCKET = 'your-hpo-bucket'
DEFAULT_DATA_PREFIX = f's3://{BUCKET}/data/'

# Function to validate S3 URI
def validate_s3_uri(uri):
    """Validate that a string is a proper S3 URI"""
    if not uri:
        return False
    if not isinstance(uri, str):
        return False
    if not uri.startswith('s3://'):
        return False
    # Check that it has at least a bucket name after the s3://
    parts = uri.split('/')
    if len(parts) < 3 or not parts[2]:
        return False
    return True

# Function to check S3 URI existence
def check_s3_object_exists(uri):
    """Check if an S3 object exists and is accessible"""
    try:
        if not uri or not uri.startswith('s3://'):
            return False
            
        # Extract bucket and key
        s3_parts = uri.replace('s3://', '').split('/', 1)
        if len(s3_parts) < 2:
            return False
            
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        # Check if object exists
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        logger.warning(f"S3 object check failed: {e}")
        return False

# Determine which data source to use
data_source_description = ""
S3_DATA_PREFIX = None  # Initialize variable to avoid NameError in error cases

# Priority 1: Command line argument
if args.input_data_s3 and not args.force_default_data:
    if validate_s3_uri(args.input_data_s3):
        S3_DATA_PREFIX = args.input_data_s3
        data_source_description = "command line argument"
    else:
        logger.error(f"❌ Invalid S3 URI in command line argument: {args.input_data_s3}")
        logger.error("Please provide a valid S3 URI starting with s3://")
        sys.exit(1)
        
# Priority 2: PINNED_DATA_S3 environment variable
elif os.getenv('PINNED_DATA_S3') and not args.force_default_data:
    if validate_s3_uri(os.getenv('PINNED_DATA_S3')):
        S3_DATA_PREFIX = os.getenv('PINNED_DATA_S3')
        data_source_description = "environment variable PINNED_DATA_S3"
    else:
        logger.error(f"❌ Invalid S3 URI in environment variable PINNED_DATA_S3: {os.getenv('PINNED_DATA_S3')}")
        logger.error("Please run 'source ./scripts/get_last_hpo_dataset.sh' to set the correct data source")
        sys.exit(1)
        
# Priority 3: last_dataset_uri.txt file
elif os.path.exists('last_dataset_uri.txt') and not args.force_default_data:
    try:
        with open('last_dataset_uri.txt', 'r') as f:
            file_uri = f.read().strip()
        
        if validate_s3_uri(file_uri):
            S3_DATA_PREFIX = file_uri
            data_source_description = "pinned dataset file (last_dataset_uri.txt)"
        else:
            logger.error(f"❌ Invalid S3 URI in last_dataset_uri.txt: {file_uri}")
            logger.error("Please run 'source ./scripts/get_last_hpo_dataset.sh' to set the correct data source")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error reading last_dataset_uri.txt: {e}")
        logger.error("Please run 'source ./scripts/get_last_hpo_dataset.sh' to set the correct data source")
        sys.exit(1)
        
# Priority 4: Legacy LAST_DATA_S3 environment variable (for backward compatibility)
elif os.getenv('LAST_DATA_S3') and not args.force_default_data:
    if validate_s3_uri(os.getenv('LAST_DATA_S3')):
        S3_DATA_PREFIX = os.getenv('LAST_DATA_S3')
        data_source_description = "environment variable LAST_DATA_S3 (legacy)"
        logger.warning("Using legacy LAST_DATA_S3 environment variable. Consider updating to PINNED_DATA_S3.")
    else:
        logger.error(f"❌ Invalid S3 URI in environment variable LAST_DATA_S3: {os.getenv('LAST_DATA_S3')}")
        logger.error("Please run 'source ./scripts/get_last_hpo_dataset.sh' to set the correct data source")
        sys.exit(1)
        
# Priority 5: Default fallback
else:
    S3_DATA_PREFIX = DEFAULT_DATA_PREFIX
    data_source_description = "default data path"
    if args.force_default_data:
        logger.info("Using default data source as requested by --force-default-data")
    else:
        logger.warning("⚠️ No pinned dataset found, using default. Run 'source ./scripts/get_last_hpo_dataset.sh' first.")

# Log the data source selection
logger.info(f"📂 Using data source from {data_source_description}: {S3_DATA_PREFIX}")

# Final validation of S3 data path
if not validate_s3_uri(S3_DATA_PREFIX):
    logger.error(f"❌ Invalid S3 data path: {S3_DATA_PREFIX}")
    logger.error("Please run 'source ./scripts/get_last_hpo_dataset.sh' to set the correct data source")
    logger.error("Or specify a valid S3 URI with --input-data-s3")
    sys.exit(1)

# Check if the S3 object exists
if not check_s3_object_exists(S3_DATA_PREFIX):
    logger.warning(f"⚠️ S3 path may not exist or is not accessible: {S3_DATA_PREFIX}")
    logger.warning("Jobs may fail if the data source is not valid.")
    # Optionally exit if you want to enforce existence
    # sys.exit(1)

# Log the final data source used
logger.info(f"🔗 Using dataset: {S3_DATA_PREFIX}")

S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/'

def get_hyperparameter_ranges():
    """Define hyperparameter ranges for tuning"""
    return {
        # Options-specific parameters
        'iv_rank_window': IntegerParameter(10, 60),
        'iv_rank_weight': ContinuousParameter(0.1, 1.0),
        'term_slope_window': IntegerParameter(5, 30),
        'term_slope_weight': ContinuousParameter(0.1, 1.0),
        'oi_window': IntegerParameter(5, 30),
        'oi_weight': ContinuousParameter(0.1, 1.0),
        'theta_window': IntegerParameter(5, 30),
        'theta_weight': ContinuousParameter(0.1, 1.0),
        
        # VIX parameters
        'vix_mom_window': IntegerParameter(5, 20),
        'vix_regime_thresh': ContinuousParameter(15.0, 35.0),
        
        # Event parameters
        'event_lag': IntegerParameter(1, 5),
        'event_lead': IntegerParameter(1, 5),
        
        # News parameters
        'news_threshold': ContinuousParameter(0.01, 0.2),
        'lookback_window': IntegerParameter(1, 10),
        'reuters_weight': ContinuousParameter(0.5, 2.0),
        'sa_weight': ContinuousParameter(0.5, 2.0),
        
        # Model parameters
        'num_leaves': IntegerParameter(10, 500),
        'max_depth': IntegerParameter(3, 15),
        'learning_rate': ContinuousParameter(0.001, 0.3, scaling_type='Logarithmic'),
        'feature_fraction': ContinuousParameter(0.1, 1.0),
        'bagging_fraction': ContinuousParameter(0.1, 1.0),
        'min_child_samples': IntegerParameter(5, 200),
        'lambda_l1': ContinuousParameter(0.0, 10.0, scaling_type='Logarithmic'),
        'lambda_l2': ContinuousParameter(0.0, 10.0, scaling_type='Logarithmic'),
    }

def launch_aapl_hpo():
    """Launch AWS SageMaker HPO job for AAPL only"""
    try:
        logger.info("Launching AWS SageMaker HPO job for AAPL only")
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Set up estimator
        estimator = Estimator(
            entry_point='run_hpo_with_macro.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            hyperparameters={
                'symbol': 'AAPL',
                'model': 'lgb',
                'debug': True
            },
            sagemaker_session=session
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=20,
            max_parallel_jobs=5
        )
        
        # Launch tuning job
        job_name = f"options-hpo-aapl-{int(time.time())}"
        tuner.fit({
            'training': f'{S3_DATA_PREFIX}AAPL/'
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
        estimator = Estimator(
            entry_point='run_hpo_with_macro.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            hyperparameters={
                'symbol': 'ALL',  # Special keyword for all filtered symbols
                'model': 'lgb',
                'apply_universe_filter': True
            },
            sagemaker_session=session
        )
        
        # Set up tuner
        hyperparameter_ranges = get_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=50,
            max_parallel_jobs=10
        )
        
        # Launch tuning job
        job_name = f"options-hpo-full-universe-{int(time.time())}"
        tuner.fit({
            'training': f'{S3_DATA_PREFIX}filtered_universe/'
        }, job_name=job_name)
        
        logger.info(f"Successfully launched full universe HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch full universe HPO job: {e}")
        return None

def main():
    """Main function to launch AWS HPO jobs"""
    logger.info(f"Starting AWS HPO launch process with data source: {S3_DATA_PREFIX}")
    
    # Print configuration for dry run
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - Configuration Summary:")
        logger.info(f"Data source: {S3_DATA_PREFIX}")
        logger.info(f"Output location: {S3_OUTPUT_PREFIX}")
        logger.info(f"AWS Role ARN: {ROLE_ARN}")
        logger.info(f"Target completed jobs threshold: {args.target_completed}")
        logger.info("Job would use the above data source for training")
        logger.info("Exiting due to --dry-run flag")
        sys.exit(0)
    
    # Additional data source verification
    try:
        # Use boto3 to check if the S3 bucket/prefix exists
        s3_client = boto3.client('s3')
        
        # Extract bucket and prefix from S3 URI
        s3_parts = S3_DATA_PREFIX.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ''
        
        # Check if bucket exists and we have access
        try:
            s3_client.head_bucket(Bucket=bucket)
            logger.info(f"✅ Verified S3 bucket access: {bucket}")
        except Exception as e:
            logger.error(f"❌ Cannot access S3 bucket {bucket}: {e}")
            logger.error("Please check your AWS credentials and bucket permissions")
            sys.exit(1)
        
        # List objects to check if prefix exists and has content
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
        if 'Contents' not in response or len(response['Contents']) == 0:
            logger.warning(f"⚠️ No objects found at {S3_DATA_PREFIX}")
            logger.warning("The data path exists but may be empty. Continuing anyway...")
        else:
            object_count = len(response['Contents'])
            logger.info(f"✅ Found {object_count} objects at data source path")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not fully verify S3 data path: {e}")
        logger.warning("Continuing with launch, but jobs may fail if path is invalid")
    
    # Step 1: Launch AAPL HPO job
    logger.info("🚀 Launching AAPL HPO job as initial test...")
    aapl_job = launch_aapl_hpo()
    
    if not aapl_job:
        logger.error("❌ Failed to launch AAPL HPO job. Aborting.")
        sys.exit(1)
    
    # Wait for AAPL job to reach "InProgress" status
    logger.info("⏳ Waiting for AAPL job to initialize...")
    
    sagemaker_client = boto3.client('sagemaker')
    max_retries = 30  # 5 minutes with 10-second interval
    retry_count = 0
    job_status = None
    
    while retry_count < max_retries:
        try:
            response = sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=aapl_job
            )
            job_status = response['HyperParameterTuningJobStatus']
            
            if job_status == 'InProgress':
                logger.info(f"✅ AAPL job is now {job_status}")
                break
            elif job_status in ['Completed', 'Stopped', 'Failed']:
                logger.error(f"❌ AAPL job unexpectedly {job_status}")
                sys.exit(1)
                
            logger.info(f"⏳ AAPL job status: {job_status}. Waiting...")
            retry_count += 1
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"❌ Error checking AAPL job status: {e}")
            retry_count += 1
            time.sleep(10)
    
    if job_status != 'InProgress':
        logger.error(f"❌ AAPL job did not reach InProgress status after 5 minutes. Current status: {job_status}")
        logger.error("Proceeding with full universe job anyway...")
    
    # Step 2: Launch full universe HPO job
    logger.info("🚀 Launching full universe HPO job...")
    universe_job = launch_full_universe_hpo()
    
    if not universe_job:
        logger.error("❌ Failed to launch full universe HPO job.")
        logger.error("AAPL job is still running, but full universe job failed to start.")
        sys.exit(1)
    
    logger.info("✅ Successfully launched both HPO jobs:")
    logger.info(f"AAPL job: {aapl_job}")
    logger.info(f"Full universe job: {universe_job}")
    logger.info("")
    logger.info("🔍 Monitor progress in the AWS SageMaker console:")
    logger.info("https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs")
    logger.info("")
    logger.info("🔄 After these jobs complete, run:")
    logger.info("source ./scripts/get_last_hpo_dataset.sh")
    logger.info("to update your environment with the latest successful dataset")

if __name__ == "__main__":
    main()
