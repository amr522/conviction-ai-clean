#!/usr/bin/env python3
"""
AWS CatBoost HPO Launch Script for 46-Stock Universe

This script launches CatBoost HPO jobs on AWS SageMaker using the same
infrastructure as the successful XGBoost HPO pipeline.
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
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aws_catboost_hpo_launch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

ROLE_ARN = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'
BUCKET = 'hpo-bucket-773934887314'
S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/catboost/'

def get_input_data_s3(cli_arg=None):
    """Get input data S3 URI with proper precedence order"""
    if cli_arg:
        logger.info(f"🔗 Using CLI argument: {cli_arg}")
        return cli_arg
    
    pinned_data = os.environ.get('PINNED_DATA_S3')
    if pinned_data:
        logger.info(f"🔗 Using PINNED_DATA_S3 from environment: {pinned_data}")
        return pinned_data
    
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

def get_catboost_hyperparameter_ranges():
    """Define CatBoost hyperparameter ranges for tuning"""
    return {
        'iterations': IntegerParameter(100, 500),
        'learning_rate': ContinuousParameter(0.01, 0.3),
        'depth': IntegerParameter(4, 10),
        'l2_leaf_reg': ContinuousParameter(1, 10),
        'border_count': IntegerParameter(32, 255),
        'bagging_temperature': ContinuousParameter(0, 1),
        'random_strength': ContinuousParameter(0, 10),
    }

def create_catboost_training_script():
    """Create CatBoost training script for SageMaker"""
    script_content = '''
import argparse
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--l2_leaf_reg', type=float, default=3.0)
    parser.add_argument('--border_count', type=int, default=128)
    parser.add_argument('--bagging_temperature', type=float, default=1.0)
    parser.add_argument('--random_strength', type=float, default=1.0)
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_file = os.path.join(args.train, 'train.csv')
    df = pd.read_csv(train_file)
    
    target_col = 'direction' if 'direction' in df.columns else 'target_next_day'
    feature_cols = [col for col in df.columns if col not in ['date', 'symbol', target_col]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        border_count=args.border_count,
        bagging_temperature=args.bagging_temperature,
        random_strength=args.random_strength,
        random_seed=42,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    
    print(f'validation-auc:{auc}')
    
    model_path = os.path.join(args.model_dir, 'catboost-model')
    model.save_model(model_path)
    
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
'''
    
    with open('catboost_train.py', 'w') as f:
        f.write(script_content)
    
    logger.info("✅ Created catboost_train.py")

def launch_catboost_hpo(input_data_s3=None, dry_run=False):
    """Launch AWS SageMaker CatBoost HPO job for 46-stock universe"""
    try:
        logger.info("Launching AWS SageMaker CatBoost HPO job for 46-stock universe")
        
        training_data = get_input_data_s3(input_data_s3)
        
        if dry_run:
            logger.info("🧪 DRY RUN MODE - No SageMaker calls will be made")
            job_name = f"catboost-hpo-46-{int(time.time())}-dry-run"
            logger.info(f"✅ DRY RUN: Would launch CatBoost HPO job: {job_name}")
            logger.info(f"✅ DRY RUN: Would use training data: {training_data}")
            return job_name
        
        create_catboost_training_script()
        
        session = sagemaker.Session()
        
        estimator = SKLearn(
            entry_point='catboost_train.py',
            role=ROLE_ARN,
            instance_count=1,
            instance_type='ml.m5.4xlarge',
            framework_version='1.0-1',
            py_version='py3',
            sagemaker_session=session
        )
        
        hyperparameter_ranges = get_catboost_hyperparameter_ranges()
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='validation-auc',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'validation-auc', 'Regex': 'validation-auc:([0-9\\.]+)'}],
            max_jobs=50,
            max_parallel_jobs=3
        )
        
        job_name = f"catboost-hpo-46-{int(time.time())}"
        tuner.fit({
            'train': TrainingInput(
                s3_data=training_data,
                content_type='text/csv',
                input_mode='File'
            )
        }, job_name=job_name)
        
        logger.info(f"Successfully launched CatBoost HPO job: {job_name}")
        return job_name
    
    except Exception as e:
        logger.error(f"Failed to launch CatBoost HPO job: {e}")
        return None

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description='Launch AWS SageMaker CatBoost HPO job')
    parser.add_argument('--input-data-s3', type=str, help='S3 URI for training data')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    
    args = parser.parse_args()
    
    logger.info("Starting CatBoost HPO pipeline...")
    
    job_name = launch_catboost_hpo(args.input_data_s3, args.dry_run)
    if job_name:
        logger.info(f"CatBoost HPO job launched: {job_name}")
    else:
        logger.error("Failed to launch CatBoost HPO job")
        if not args.dry_run:
            sys.exit(1)
    
    logger.info("CatBoost HPO launch completed!")

if __name__ == "__main__":
    main()
