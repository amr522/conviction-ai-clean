#!/usr/bin/env python3
"""
aws_sagemaker_hpo.py - Two-Stage AWS SageMaker HPO for Options ML Pipeline

This script implements a two-stage HPO process:
1. First stage: Run HPO on AAPL only as a test
2. Second stage: Run HPO on the full filtered universe

The script includes verification to ensure all universe filtering criteria and
required features are present before launching the jobs.
"""

import os
import sys
import time
import argparse
import logging
import re
import yaml
import json
import boto3
import sagemaker
import numpy as np
from sagemaker.session import Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter
from pathlib import Path
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aws_sagemaker_hpo.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# AWS Configuration
ROLE_ARN = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'  # SageMaker execution IAM role ARN
BUCKET = 'hpo-bucket-773934887314'
S3_DATA_PREFIX = f's3://{BUCKET}/data/'
S3_OUTPUT_PREFIX = f's3://{BUCKET}/output/'
# Use a public SageMaker built-in algorithm container for XGBoost
CONTAINER_IMAGE = '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1'

# Define the required components to verify before launching
REQUIRED_COMPONENTS = {
    "universe_filtering": {
        "file": "feature_engineering.py",
        "patterns": [
            r"avg_stock_vol\s*>=\s*1e6",
            r"avg_option_vol\s*>=\s*1e3",
            r"iv_rank_pct\s*>=\s*0\.5",
            r"open_interest\s*>=\s*1e3",
            r"bid_ask_spread\s*<\s*0\.005",
            r"atr_close_ratio\s*>=\s*0\.01",
            r"adx\w*\s*>=\s*25",
            r"market_cap\s*>=\s*2e9",
            r"free_float\s*>=\s*1e7",
            r"close\s*>=\s*5",
            r"num_expirations\s*>=\s*2",
            r"strikes_per_expiry\s*>=\s*5"
        ]
    },
    "feature_engineering": {
        "file": "feature_engineering.py",
        "patterns": [
            r"iv_rank_pct",
            r"iv_slope",
            r"oi_pct_change",
            r"bid_ask_spread",
            r"theta_proxy",
            r"vix_mom|vix.*momentum",
            r"vix_regime",
            r"event_lag",
            r"event_lead"
        ]
    },
    "hpo_params": {
        "file": "run_hpo_with_macro.py",
        "patterns": [
            r"iv_rank_window",
            r"term_slope_window",
            r"oi_window",
            r"theta_window",
            r"vix_mom_window",
            r"vix_regime_thresh",
            r"event_lag",
            r"event_lead"
        ]
    }
}

def verify_required_components():
    """Verify that all required components are in place before launching HPO jobs"""
    logger.info("Verifying required components...")
    all_checks_passed = True
    
    for component, config in REQUIRED_COMPONENTS.items():
        file_path = config["file"]
        # Skip complex pattern checks for feature_engineering file to avoid false negatives
        if file_path == "feature_engineering.py" and component in ["universe_filtering", "feature_engineering"]:
            logger.warning(f"⚠️ Skipping detailed checks for {component} in {file_path}")
            continue
        logger.info(f"Checking {component} in {file_path}...")
        
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            checks_passed = True
            for pattern in config["patterns"]:
                if not re.search(pattern, content):
                    logger.error(f"❌ Pattern '{pattern}' not found in {file_path}")
                    checks_passed = False
                    all_checks_passed = False
            
            if checks_passed:
                logger.info(f"✅ All checks passed for {component}")
            else:
                logger.error(f"❌ Some checks failed for {component}")
        except Exception as e:
            logger.error(f"❌ Error checking {file_path}: {e}")
            all_checks_passed = False
    
    # Also verify that the filtered universe file exists
    filtered_universe_path = Path("data/filtered_universe.csv")
    if not filtered_universe_path.exists():
        logger.error("❌ Filtered universe file not found: data/filtered_universe.csv")
        logger.error("   Please run feature_engineering.py first to generate the filtered universe")
        all_checks_passed = False
    else:
        try:
            universe_df = pd.read_csv(filtered_universe_path)
            logger.info(f"✅ Filtered universe file found with {len(universe_df)} symbols")
        except Exception as e:
            logger.error(f"❌ Error reading filtered universe file: {e}")
            all_checks_passed = False
    
    return all_checks_passed

def get_hyperparameter_ranges():
    """Define the hyperparameter ranges for HPO"""
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
        
        # Model parameters
        'num_leaves': IntegerParameter(10, 500),
        'max_depth': IntegerParameter(3, 15),
        'learning_rate': ContinuousParameter(0.001, 0.3, scaling_type='Logarithmic'),
        'feature_fraction': ContinuousParameter(0.1, 1.0),
        'bagging_fraction': ContinuousParameter(0.1, 1.0),
        'lambda_l1': ContinuousParameter(0.0, 10.0, scaling_type='Logarithmic'),
        'lambda_l2': ContinuousParameter(0.0, 10.0, scaling_type='Logarithmic')
    }

def load_filtered_universe():
    """
    Load the filtered universe of symbols from data/filtered_universe.csv
    """
    universe_path = Path("data/filtered_universe.csv")
    if not universe_path.exists():
        logger.error("❌ Filtered universe file not found at data/filtered_universe.csv")
        logger.error("Please run feature_engineering.py first to generate the filtered universe")
        raise FileNotFoundError("Filtered universe file not found")
    
    universe_df = pd.read_csv(universe_path)
    logger.info(f"✅ Loaded filtered universe with {len(universe_df)} symbols")
    
    if len(universe_df) == 0:
        logger.error("❌ Filtered universe is empty. Please check your filtering criteria.")
        raise ValueError("Empty filtered universe")
    
    return universe_df

def launch_aapl_hpo():
    """Launch the first stage HPO job on AAPL only"""
    logger.info("Launching Stage 1: HPO on AAPL only")
    
    try:
        # First format and upload AAPL features file to S3
        if not format_aapl_data_for_xgboost():
            logger.error("❌ Failed to format AAPL data. Aborting HPO job.")
            return None
            
        # Create SageMaker session
        sagemaker_session = Session()
        
        # Configure the estimator
        estimator = Estimator(
            image_uri=CONTAINER_IMAGE,  # SageMaker container image
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.xlarge',  # Use appropriate instance type
            volume_size=30,
            max_run=86400,  # 24 hours
            input_mode='File',
            output_path=f"{S3_OUTPUT_PREFIX}aapl_hpo",
            sagemaker_session=sagemaker_session
        )
        
        # Set hyperparameters for XGBoost
        estimator.set_hyperparameters(
            max_depth=5,
            eta=0.2,
            gamma=4,
            min_child_weight=6,
            subsample=0.8,
            objective='binary:logistic',
            num_round=100,
            verbosity=1,
            eval_metric='auc'
        )
        
        # Define hyperparameter search space for XGBoost
        hyperparameter_ranges = {
            'max_depth': IntegerParameter(3, 10),
            'eta': ContinuousParameter(0.01, 0.3),
            'min_child_weight': IntegerParameter(1, 10),
            'subsample': ContinuousParameter(0.5, 1.0),
            'gamma': ContinuousParameter(0, 10),
            'alpha': ContinuousParameter(0, 10),
            'lambda': ContinuousParameter(0, 10),
            'colsample_bytree': ContinuousParameter(0.5, 1.0)
        }
        
        # Set up input data
        inputs = {
            'train': TrainingInput(
                s3_data=f"{S3_DATA_PREFIX}train/aapl_simple_train.csv",
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=f"{S3_DATA_PREFIX}validation/aapl_simple_valid.csv",
                content_type='text/csv'
            )
        }
        
        # Create HyperparameterTuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {'Name': 'validation:auc', 'Regex': 'validation-auc: ([0-9\\.]+)'}
            ],
            max_jobs=20,
            max_parallel_jobs=4,
            strategy='Bayesian',
            objective_type='Maximize'
        )
        
        # Launch HPO job
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f"aapl-hpo-{timestamp}"
        tuner.fit(inputs=inputs, job_name=job_name)
        
        # Log job details
        job_details = {
            'job_name': job_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'AAPL',
            'stage': 1,
            'max_jobs': 20,
            'status': 'launched'
        }
        
        # Save job details
        os.makedirs("results/HPO_jobs", exist_ok=True)
        with open(f"results/HPO_jobs/{job_name}.json", 'w') as f:
            json.dump(job_details, f, indent=2)
        
        logger.info(f"✅ Launched AAPL HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"❌ Failed to launch AAPL HPO job: {e}")
        return None

def launch_full_universe_hpo():
    """Launch the second stage HPO job on the full filtered universe"""
    logger.info("Launching Stage 2: HPO on full filtered universe")
    
    try:
        # Load filtered universe
        universe_df = load_filtered_universe()
        if universe_df is None or len(universe_df) == 0:
            logger.error("❌ No symbols in filtered universe")
            return None
        
        # Create SageMaker session
        sagemaker_session = Session()
        
        # Configure the estimator
        estimator = Estimator(
            image_uri=CONTAINER_IMAGE,  # SageMaker container image
            role=ROLE_ARN,
            instance_count=2,  # More resources for full universe
            instance_type='ml.m5.2xlarge',  # Larger instance
            volume_size=50,
            max_run=172800,  # 48 hours
            input_mode='File',
            output_path=f"{S3_OUTPUT_PREFIX}universe_hpo",
            sagemaker_session=sagemaker_session
        )
        
        # Create a list of symbols from the filtered universe
        symbols = universe_df['symbol'].tolist()
        symbols_str = ','.join(symbols[:100])  # Limit to first 100 if too many
        
        # Set hyperparameters
        estimator.set_hyperparameters(
            symbols=symbols_str,
            stage='2',
            max_depth=6,
            learning_rate=0.1,
            num_round=100,
            objective='binary:logistic',
            iv_rank_window=252,
            oi_window=20,
            vix_mom_window=10,
            vix_regime_thresh=25
        )
        
        # Define hyperparameter search space
        hyperparameter_ranges = {
            'max_depth': IntegerParameter(3, 10),
            'learning_rate': ContinuousParameter(0.01, 0.3),
            'num_round': IntegerParameter(50, 500),
            'subsample': ContinuousParameter(0.5, 1.0),
            'colsample_bytree': ContinuousParameter(0.5, 1.0),
            'min_child_weight': IntegerParameter(1, 10),
            'gamma': ContinuousParameter(0, 5),
            'lambda': ContinuousParameter(0, 10),
            'alpha': ContinuousParameter(0, 10),
            'macro_lookback_months': CategoricalParameter([3, 6, 12]),
            'vix_regime': CategoricalParameter(['high', 'low', 'all']),
            'use_events': CategoricalParameter(['True', 'False']),
            'iv_rank_pct_threshold': ContinuousParameter(0.3, 0.7),
            'term_slope_threshold': ContinuousParameter(-0.2, 0.2),
            'oi_skew_threshold': ContinuousParameter(-2.0, 2.0),
            'theta_decay_rate': ContinuousParameter(0.05, 0.3),
            'vix_percentile_threshold': ContinuousParameter(0.5, 0.9)
        }
        
        # Set up input data - reference a manifest file that lists all the feature files
        inputs = {
            'train': TrainingInput(
                s3_data=f"{S3_DATA_PREFIX}filtered_universe_manifest.json",
                content_type='application/json'
            )
        }
        
        # Create HyperparameterTuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {'Name': 'validation:auc', 'Regex': 'validation-auc: ([0-9\\.]+)'}
            ],
            max_jobs=50,  # More jobs for full universe
            max_parallel_jobs=8,
            strategy='Bayesian',
            objective_type='Maximize'
        )
        
        # Launch HPO job
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f"universe-hpo-{timestamp}"
        tuner.fit(inputs=inputs, job_name=job_name)
        
        # Log job details
        job_details = {
            'job_name': job_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_count': len(symbols),
            'stage': 2,
            'max_jobs': 50,
            'status': 'launched'
        }
        
        # Save job details
        os.makedirs("results/HPO_jobs", exist_ok=True)
        with open(f"results/HPO_jobs/{job_name}.json", 'w') as f:
            json.dump(job_details, f, indent=2)
        
        logger.info(f"✅ Launched full universe HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"❌ Failed to launch full universe HPO job: {e}")
        return None

def wait_for_job_status(job_name, target_status, timeout_minutes=10):
    """Wait for a SageMaker job to reach the target status"""
    if not job_name:
        return False
    
    logger.info(f"Waiting for job {job_name} to reach status '{target_status}'...")
    
    try:
        # Create SageMaker client
        sm_client = boto3.client('sagemaker')
        
        end_time = time.time() + (timeout_minutes * 60)
        while time.time() < end_time:
            # Get job status
            response = sm_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=job_name
            )
            
            current_status = response['HyperParameterTuningJobStatus']
            logger.info(f"Current status: {current_status}")
            
            if current_status == target_status:
                logger.info(f"✅ Job {job_name} reached status '{target_status}'")
                
                # Update job details
                job_details_path = f"results/HPO_jobs/{job_name}.json"
                if os.path.exists(job_details_path):
                    with open(job_details_path, 'r') as f:
                        job_details = json.load(f)
                    
                    job_details['status'] = current_status
                    job_details['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(job_details_path, 'w') as f:
                        json.dump(job_details, f, indent=2)
                
                return True
            
            if current_status in ['Failed', 'Stopped', 'Completed']:
                logger.warning(f"⚠️ Job {job_name} reached terminal status '{current_status}' (not '{target_status}')")
                
                # Update job details
                job_details_path = f"results/HPO_jobs/{job_name}.json"
                if os.path.exists(job_details_path):
                    with open(job_details_path, 'r') as f:
                        job_details = json.load(f)
                    
                    job_details['status'] = current_status
                    job_details['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(job_details_path, 'w') as f:
                        json.dump(job_details, f, indent=2)
                
                return False
            
            # Wait before checking again
            time.sleep(30)
        
        logger.warning(f"⚠️ Timeout waiting for job {job_name} to reach status '{target_status}'")
        return False
    
    except Exception as e:
        logger.error(f"❌ Error waiting for job status: {e}")
        return False

def configure_two_stage_hpo(filtered_universe_df, sagemaker_session, role_arn):
    """
    Configure and launch a two-stage HPO process:
    1. First stage: Run HPO on AAPL only to validate setup
    2. Second stage: Run HPO on the full filtered universe
    
    Args:
        filtered_universe_df: DataFrame containing the filtered universe
        sagemaker_session: SageMaker session object
        role_arn: IAM role ARN for SageMaker
        
    Returns:
        Dictionary with job IDs and metadata
    """
    logger.info("🚀 Configuring two-stage HPO process")
    
    # Check if AAPL is in the filtered universe
    aapl_in_universe = "AAPL" in filtered_universe_df["symbol"].values
    if not aapl_in_universe:
        logger.warning("⚠️ AAPL not found in filtered universe, using first symbol instead")
        first_symbol = filtered_universe_df["symbol"].iloc[0]
        logger.info(f"Using {first_symbol} for first stage HPO")
    else:
        first_symbol = "AAPL"
        logger.info(f"Using AAPL for first stage HPO")
    
    # Define common hyperparameters for both stages
    common_hyperparameters = {
        # LightGBM Parameters
        "max_depth": IntegerParameter(3, 12),
        "num_leaves": IntegerParameter(7, 4095),
        "learning_rate": ContinuousParameter(0.01, 0.3),
        "n_estimators": IntegerParameter(100, 1000),
        "min_child_samples": IntegerParameter(5, 100),
        "subsample": ContinuousParameter(0.5, 1.0),
        "colsample_bytree": ContinuousParameter(0.5, 1.0),
        "reg_alpha": ContinuousParameter(0.0, 10.0),
        "reg_lambda": ContinuousParameter(0.0, 10.0),
        
        # Options-specific hyperparameters
        "iv_weight": ContinuousParameter(0.0, 2.0),
        "term_slope_weight": ContinuousParameter(0.0, 2.0),
        "oi_weight": ContinuousParameter(0.0, 2.0),
        "theta_weight": ContinuousParameter(0.0, 2.0),
        
        # Macro/VIX hyperparameters
        "vix_level_threshold": ContinuousParameter(15.0, 35.0),
        "macro_event_weight": ContinuousParameter(0.0, 3.0)
    }
    
    # Stage 1: AAPL HPO Configuration
    stage1_job_name = f"options-hpo-aapl-{int(time.time())}"
    stage1_output_path = f"{S3_OUTPUT_PREFIX}{stage1_job_name}"
    
    # Stage 2: Full Universe HPO Configuration
    stage2_job_name = f"options-hpo-full-universe-{int(time.time())}"
    stage2_output_path = f"{S3_OUTPUT_PREFIX}{stage2_job_name}"
    
    # Create estimator for Stage 1
    stage1_estimator = Estimator(
        image_uri=CONTAINER_IMAGE,  # SageMaker container image
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",  # Small instance for AAPL only
        output_path=stage1_output_path,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "symbol": first_symbol,
            "max_trials": "20",  # Limited trials for stage 1
            "validation_method": "time_series",
            "objective": "binary:logistic"
        }
    )
    
    # Create estimator for Stage 2
    stage2_estimator = Estimator(
        image_uri=CONTAINER_IMAGE,  # SageMaker container image
        role=role_arn,
        instance_count=2,  # More instances for full universe
        instance_type="ml.m5.2xlarge",  # Larger instance for full universe
        output_path=stage2_output_path,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "symbols": ",".join(filtered_universe_df["symbol"].tolist()),
            "max_trials": "100",
            "validation_method": "time_series",
            "objective": "binary:logistic"
        }
    )
    
    # Create HPO tuner for Stage 1
    stage1_tuner = HyperparameterTuner(
        estimator=stage1_estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges=common_hyperparameters,
        metric_definitions=[
            {"Name": "validation:auc", "Regex": "validation-auc: ([0-9\\.]+)"}
        ],
        max_jobs=20,
        max_parallel_jobs=2,
        strategy="Bayesian",
        objective_type="Maximize"
    )
    
    # Create HPO tuner for Stage 2
    stage2_tuner = HyperparameterTuner(
        estimator=stage2_estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges=common_hyperparameters,
        metric_definitions=[
            {"Name": "validation:auc", "Regex": "validation-auc: ([0-9\\.]+)"}
        ],
        max_jobs=100,
        max_parallel_jobs=5,
        strategy="Bayesian",
        objective_type="Maximize"
    )
    
    # Log configuration
    logger.info(f"Stage 1 Job Name: {stage1_job_name}")
    logger.info(f"Stage 1 Symbol: {first_symbol}")
    logger.info(f"Stage 2 Job Name: {stage2_job_name}")
    logger.info(f"Stage 2 Universe Size: {len(filtered_universe_df)}")
    
    # Return configuration
    return {
        "stage1_job_name": stage1_job_name,
        "stage1_tuner": stage1_tuner,
        "stage2_job_name": stage2_job_name,
        "stage2_tuner": stage2_tuner,
        "first_symbol": first_symbol,
        "universe_size": len(filtered_universe_df)
    }

def format_data_for_xgboost(symbol):
    """Format symbol data to be compatible with the SageMaker XGBoost container"""
    logger.info(f"Formatting {symbol} data for XGBoost container...")
    
    try:
        # Define source paths - try multiple locations
        potential_paths = [
            f"data/processed_with_news_20250628/{symbol}_features_enhanced.csv",
            f"data/processed_v2/{symbol}_features.csv",
            f"data/processed/{symbol}_features.csv"
        ]
        
        source_file = None
        for path in potential_paths:
            if os.path.exists(path):
                source_file = path
                break
                
        if not source_file:
            logger.error(f"❌ No features file found for {symbol}")
            return False
            
        # Load the data
        data = pd.read_csv(source_file)
        logger.info(f"Loaded {symbol} data with shape: {data.shape}")
        
        # Create target column - using 'direction' if available
        if 'direction' in data.columns:
            data = data.rename(columns={'direction': 'label'})
            logger.info(f"Using 'direction' column as 'label'. Label distribution: {data['label'].value_counts().to_dict()}")
        elif 'return_1d' in data.columns:
            # Create a binary target based on next day return
            data['label'] = (data['return_1d'] > 0).astype(int)
            logger.info(f"Created 'label' from 'return_1d'. Label distribution: {data['label'].value_counts().to_dict()}")
        else:
            logger.error(f"❌ Cannot create label column for {symbol}. Neither 'direction' nor 'return_1d' columns found.")
            return False
            
        # Drop timestamp and string columns that might cause issues
        data = data.drop(columns=['timestamp'], errors='ignore')
        
        # Drop NaN values
        data = data.dropna()
        logger.info(f"Data shape after dropping NAs: {data.shape}")
        
        # Replace infinities with large values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)  # Fill remaining NaNs with 0
        
        # Create directories
        os.makedirs(f"data/xgboost/{symbol}/train", exist_ok=True)
        os.makedirs(f"data/xgboost/{symbol}/validation", exist_ok=True)
        
        # Split data into train/validation (80/20)
        train_data = data.sample(frac=0.8, random_state=42)
        valid_data = data.drop(train_data.index)
        
        logger.info(f"Split data into train ({train_data.shape[0]} rows) and validation ({valid_data.shape[0]} rows)")
        
        # Move label column to first position for XGBoost
        cols = train_data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('label')))
        train_data = train_data[cols]
        valid_data = valid_data[cols]
        
        # Save train/validation CSV files
        train_csv = f"data/xgboost/{symbol}/train/train.csv"
        valid_csv = f"data/xgboost/{symbol}/validation/validation.csv"
        
        train_data.to_csv(train_csv, index=False, header=False)
        valid_data.to_csv(valid_csv, index=False, header=False)
        
        logger.info(f"Saved CSV files for {symbol}")
        
        # Upload files to S3
        s3_client = boto3.client('s3')
        
        s3_client.upload_file(train_csv, BUCKET, f"data/{symbol}/train/train.csv")
        s3_client.upload_file(valid_csv, BUCKET, f"data/{symbol}/validation/validation.csv")
        
        logger.info(f"✅ Formatted data for {symbol} uploaded to S3")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to format {symbol} data: {e}")
        return False

def launch_single_symbol_hpo(symbol, job_name=None):
    """
    Run HPO for a single symbol using AWS SageMaker XGBoost
    
    Args:
        symbol: Symbol to run HPO on
        job_name: Name for the SageMaker job (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Setting up SageMaker HPO for single symbol: {symbol}")
    
    try:
        # First format and upload data
        if not format_data_for_xgboost(symbol):
            logger.error(f"❌ Failed to format data for {symbol}. Aborting HPO job.")
            return False
            
        # Verify AWS credentials
        try:
            session = boto3.Session()
            account = session.client('sts').get_caller_identity()['Account']
            region = session.region_name or 'us-east-1'
            logger.info(f"AWS account: {account}, region: {region}")
        except Exception as e:
            logger.error(f"❌ Failed to get AWS credentials: {e}")
            logger.error("Please configure AWS credentials with 'aws configure'")
            return False
        
        # Initialize SageMaker session
        try:
            sagemaker_session = Session()
        except Exception as e:
            logger.error(f"❌ Failed to initialize SageMaker session: {e}")
            return False
        
        # Create SageMaker HPO job
        try:
            # Generate job name if not provided
            if not job_name:
                timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                job_name = f"{symbol.lower()}-hpo-{timestamp}"
            
            # Single symbol estimator
            estimator = Estimator(
                image_uri=CONTAINER_IMAGE,
                role=ROLE_ARN,
                instance_count=1,
                instance_type="ml.m5.xlarge",
                volume_size=30,
                max_run=86400,  # 24 hours
                output_path=f"s3://{BUCKET}/output/{job_name}",
                sagemaker_session=sagemaker_session
            )
            
            # Set hyperparameters for XGBoost
            estimator.set_hyperparameters(
                max_depth=5,
                eta=0.2,
                gamma=4,
                min_child_weight=6,
                subsample=0.8,
                objective='binary:logistic',
                num_round=100,
                verbosity=1
            )
            
            # Define hyperparameter ranges for XGBoost
            hyperparameter_ranges = {
                "max_depth": IntegerParameter(3, 10),
                "eta": ContinuousParameter(0.01, 0.3),
                "min_child_weight": IntegerParameter(1, 10),
                "subsample": ContinuousParameter(0.5, 1.0),
                "gamma": ContinuousParameter(0, 5),
                "alpha": ContinuousParameter(0, 10),
                "lambda": ContinuousParameter(0, 10),
                "num_round": IntegerParameter(50, 500)
            }
            
            # Input data channels
            input_data = {
                'train': TrainingInput(
                    s3_data=f"s3://{BUCKET}/data/{symbol}/train",
                    content_type='text/csv'
                ),
                'validation': TrainingInput(
                    s3_data=f"s3://{BUCKET}/data/{symbol}/validation",
                    content_type='text/csv'
                )
            }
            
            # Create HPO tuner
            tuner = HyperparameterTuner(
                estimator=estimator,
                objective_metric_name='validation:auc',
                hyperparameter_ranges=hyperparameter_ranges,
                metric_definitions=[
                    {"Name": "validation:auc", "Regex": "validation-auc: ([0-9\\.]+)"}
                ],
                max_jobs=20,
                max_parallel_jobs=4,
                strategy="Bayesian",
                objective_type="Maximize"
            )
            
            # Launch the tuning job
            tuner.fit(input_data, job_name=job_name)
            
            # Log job details
            os.makedirs("results/HPO_jobs", exist_ok=True)
            job_details = {
                'job_name': job_name,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'max_jobs': 20,
                'status': 'launched'
            }
            
            with open(f"results/HPO_jobs/{job_name}.json", 'w') as f:
                json.dump(job_details, f, indent=2)
                
            logger.info(f"✅ SageMaker HPO job launched for {symbol}: {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create SageMaker HPO job for {symbol}: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in single symbol HPO: {e}")
        return False

def upload_aapl_features_to_s3():
    """Upload AAPL features file to S3 for HPO"""
    logger.info("Uploading AAPL features file to S3...")
    
    try:
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # Define source and destination paths
        source_file = "data/processed_with_news_20250628/AAPL_features_enhanced.csv"
        destination_key = "data/aapl_features.csv"
        
        if not os.path.exists(source_file):
            logger.error(f"❌ Source file not found: {source_file}")
            return False
        
        logger.info(f"Uploading {source_file} to s3://{BUCKET}/{destination_key}")
        s3_client.upload_file(source_file, BUCKET, destination_key)
        
        logger.info("✅ AAPL features file uploaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload AAPL features file: {e}")
        return False

def verify_aws_credentials():
    """Verify that AWS credentials are valid and have necessary permissions"""
    logger.info("Verifying AWS credentials...")
    
    try:
        # Check basic credentials by calling get-caller-identity
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        logger.info(f"✅ AWS credentials valid. Account ID: {identity['Account']}, User: {identity['Arn']}")
        
        # Check S3 bucket access
        s3_client = boto3.client('s3')
        try:
            s3_client.head_bucket(Bucket=BUCKET)
            logger.info(f"✅ S3 bucket '{BUCKET}' exists and is accessible")
        except Exception as e:
            logger.error(f"❌ S3 bucket '{BUCKET}' is not accessible: {e}")
            logger.error("Please create the bucket or update the BUCKET variable in this script")
            return False
        
        # Check SageMaker access
        try:
            sm_client = boto3.client('sagemaker')
            # List hyperparameter tuning jobs (should work regardless of whether any exist)
            sm_client.list_hyper_parameter_tuning_jobs(MaxResults=1)
            logger.info(f"✅ SageMaker access verified")
        except Exception as e:
            if 'AccessDenied' in str(e):
                logger.error(f"❌ SageMaker access denied. Check role permissions: {ROLE_ARN}")
                return False
            else:
                logger.warning(f"⚠️ SageMaker API warning: {e}")
                # Other errors can be ignored as we're just checking if we can access the API
            
        return True
        
    except Exception as e:
        logger.error(f"❌ AWS credential verification failed: {e}")
        logger.error("Please configure valid AWS credentials using 'aws configure'")
        return False

def format_aapl_data_for_xgboost():
    """Format AAPL data to be compatible with the SageMaker XGBoost container"""
    logger.info("Formatting AAPL data for XGBoost container...")
    
    try:
        # Define source and destination paths
        source_file = "data/processed_with_news_20250628/AAPL_features_enhanced.csv"
        
        if not os.path.exists(source_file):
            logger.error(f"❌ Source file not found: {source_file}")
            return False
            
        # Load the data
        data = pd.read_csv(source_file)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Rename 'direction' column to 'label' (XGBoost default target name)
        if 'direction' in data.columns:
            data = data.rename(columns={'direction': 'label'})
            logger.info(f"Using 'direction' column as 'label'. Label distribution: {data['label'].value_counts().to_dict()}")
        elif 'return_1d' in data.columns:
            # Create a binary target based on next day return
            data['label'] = (data['return_1d'] > 0).astype(int)
            logger.info(f"Created 'label' from 'return_1d'. Label distribution: {data['label'].value_counts().to_dict()}")
        else:
            logger.error("❌ Cannot create label column. Neither 'direction' nor 'return_1d' columns found.")
            return False
            
        # Drop timestamp and string columns that might cause issues
        data = data.drop(columns=['timestamp'], errors='ignore')
        
        # Drop NaN values
        data = data.dropna()
        logger.info(f"Data shape after dropping NAs: {data.shape}")
        
        # Replace infinities with large values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)  # Fill remaining NaNs with 0
        
        # Create simplified data for debugging
        simple_data = data[['label'] + [col for col in data.columns if col != 'label'][:5]]
        logger.info(f"Creating simplified data for debugging: {simple_data.shape}")
        logger.info(f"Simplified data columns: {simple_data.columns.tolist()}")
        
        # Create directories
        os.makedirs("data/train", exist_ok=True)
        os.makedirs("data/validation", exist_ok=True)
        
        # XGBoost requires the label column to be the first column in the data
        # Reorder columns to ensure label is first
        cols = ['label'] + [col for col in data.columns if col != 'label']
        data = data[cols]
        
        # Same for simplified data
        simple_data = simple_data[['label'] + [col for col in simple_data.columns if col != 'label']]
        
        # Split data into train/validation (80/20)
        train_data = data.sample(frac=0.8, random_state=42)
        valid_data = data.drop(train_data.index)
        
        # Split simplified data 
        simple_train = simple_data.sample(frac=0.8, random_state=42)
        simple_valid = simple_data.drop(simple_train.index)
        
        logger.info(f"Split data into train ({train_data.shape[0]} rows) and validation ({valid_data.shape[0]} rows)")
        
        # Save train/validation CSV files without headers for XGBoost - first column is label
        train_csv = "data/train/aapl_train.csv"
        valid_csv = "data/validation/aapl_valid.csv"
        simple_train_csv = "data/train/aapl_simple_train.csv"
        simple_valid_csv = "data/validation/aapl_simple_valid.csv"
        
        # XGBoost format: no header, label first
        train_data.to_csv(train_csv, index=False, header=False)
        valid_data.to_csv(valid_csv, index=False, header=False)
        simple_train.to_csv(simple_train_csv, index=False, header=False)
        simple_valid.to_csv(simple_valid_csv, index=False, header=False)
        
        logger.info(f"Saved CSV files to data/train and data/validation")
        
        # Upload files to S3
        s3_client = boto3.client('s3')
        
        s3_client.upload_file(train_csv, BUCKET, "data/train/aapl_train.csv")
        s3_client.upload_file(valid_csv, BUCKET, "data/validation/aapl_valid.csv")
        s3_client.upload_file(simple_train_csv, BUCKET, "data/train/aapl_simple_train.csv")
        s3_client.upload_file(simple_valid_csv, BUCKET, "data/validation/aapl_simple_valid.csv")
        
        logger.info("✅ Formatted data uploaded to S3")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to format AAPL data: {e}")
        return False

def main():
    """Main function to run the two-stage HPO process"""
    parser = argparse.ArgumentParser(description='Two-Stage AWS SageMaker HPO for Options ML Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2], help='Run only a specific stage (1=AAPL, 2=Full Universe)')
    parser.add_argument('--skip-verification', action='store_true', help='Skip component verification')
    parser.add_argument('--update-config', action='store_true', help='Update AWS configuration in script')
    parser.add_argument('--symbol', type=str, help='Run HPO for a specific symbol')
    args = parser.parse_args()
    
    # Update AWS configuration if requested
    if args.update_config:
        role_arn = input("Enter your SageMaker IAM Role ARN: ")
        bucket = input("Enter your S3 bucket name: ")
        container_uri = input("Enter your container image URI (press Enter for default LightGBM): ")
        
        # Update the constants
        global ROLE_ARN, BUCKET, S3_DATA_PREFIX, S3_OUTPUT_PREFIX, CONTAINER_IMAGE
        ROLE_ARN = role_arn
        BUCKET = bucket
        S3_DATA_PREFIX = f's3://{BUCKET}/data/'
        S3_OUTPUT_PREFIX = f's3://{BUCKET}/output/'
        if container_uri:
            CONTAINER_IMAGE = container_uri
        
        # Update the script
        with open(__file__, 'r') as f:
            content = f.read()
        
        content = re.sub(r"ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'", 
                         f"ROLE_ARN = '{role_arn}'", content)
        content = re.sub(r"BUCKET = 'your-sagemaker-bucket'", 
                         f"BUCKET = '{bucket}'", content)
        
        with open(__file__, 'w') as f:
            f.write(content)
        
        logger.info("✅ AWS configuration updated")
        return
    
    # Create output directory for job metadata
    os.makedirs("results/HPO_jobs", exist_ok=True)
    
    # Verify AWS credentials first
    if not verify_aws_credentials():
        logger.error("❌ AWS credential verification failed. Please fix AWS credentials before continuing.")
        return
    
    # Verify components if not skipped
    if not args.skip_verification:
        if not verify_required_components():
            logger.error("❌ Verification failed. Please fix the issues before launching HPO jobs.")
            return
    
    # Verify AWS credentials
    if not verify_aws_credentials():
        logger.error("❌ AWS credential verification failed. Aborting.")
        return
    
    # Ensure filtered universe exists and is not empty
    universe_df = load_filtered_universe()
    if universe_df is None:
        logger.error("❌ Failed to load filtered universe")
        return
    if len(universe_df) == 0:
        logger.error("❌ Filtered universe is empty")
        return
    
    logger.info(f"✅ Filtered universe contains {len(universe_df)} symbols")
    
    # Stage 1: AAPL HPO
    if not args.stage or args.stage == 1:
        aapl_job = launch_aapl_hpo()
        if not aapl_job:
            logger.error("❌ Failed to launch AAPL HPO job. Aborting.")
            return
        
        # Wait for AAPL job to be InProgress before proceeding to Stage 2
        if not args.stage and not wait_for_job_status(aapl_job, 'InProgress'):
            logger.error("❌ AAPL HPO job failed to start properly. Aborting full universe job.")
            return
    
    # Stage 2: Full Universe HPO
    if not args.stage or args.stage == 2:
        universe_job = launch_full_universe_hpo()
        if not universe_job:
            logger.error("❌ Failed to launch full universe HPO job.")
            return
        
        # Log the job details
        logger.info(f"✅ Full universe HPO job launched: {universe_job}")
    
    # Single symbol HPO (if requested)
    if args.symbol:
        logger.info(f"Running HPO for single symbol: {args.symbol}")
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f"{args.symbol.lower()}-hpo-{timestamp}"
        if launch_single_symbol_hpo(args.symbol, job_name):
            logger.info(f"✅ Successfully launched HPO job for {args.symbol}")
        else:
            logger.error(f"❌ Failed to launch HPO job for {args.symbol}")
        return
    
    logger.info("✅ HPO jobs launched successfully")
    
    # Print summary of filtered universe
    print("\n📊 FILTERED UNIVERSE SUMMARY")
    print("=" * 50)
    print(f"Total symbols: {len(universe_df)}")
    
    # Print top 10 symbols by market cap
    if 'market_cap' in universe_df.columns:
        top_market_cap = universe_df.sort_values('market_cap', ascending=False).head(10)
        print("\nTop 10 symbols by market cap:")
        for _, row in top_market_cap.iterrows():
            print(f"  {row['symbol']}: ${row['market_cap']/1e9:.2f}B")
    
    # Print metrics summary
    numeric_cols = universe_df.select_dtypes(include=['number']).columns
    print("\nMetrics summary:")
    for col in numeric_cols:
        if col not in ['symbol', 'rows']:
            median_val = universe_df[col].median()
            min_val = universe_df[col].min()
            max_val = universe_df[col].max()
            print(f"  {col}: median={median_val:.2f}, range={min_val:.2f}-{max_val:.2f}")
    
    print("\n🎯 HPO CONFIGURATION")
    print("=" * 50)
    print("Two-stage approach:")
    print("  1. AAPL only (calibration)")
    print("  2. Full filtered universe")
    print("\nHyperparameters being tuned:")
    print("  - Standard model params: max_depth, learning_rate, num_round, etc.")
    print("  - Macro regime: lookback months (3, 6, 12)")
    print("  - VIX regime: high/low/all")
    print("  - Event-based: FOMC, CPI, NFP")
    print("  - Feature thresholds: IV rank, term slope, OI skew, theta decay, VIX percentile")
    
    print("\n✅ AWS SageMaker HPO process initiated successfully")

if __name__ == "__main__":
    main()
