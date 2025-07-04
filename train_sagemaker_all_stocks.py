#!/usr/bin/env python3
"""
Standard SageMaker Training Pipeline for All Stocks
"""
import argparse
import boto3
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_all_stocks(deploy=False, models_file=None):
    """Train models for all stocks using standard pipeline"""
    logger.info("📊 Starting standard SageMaker training pipeline")
    
    if models_file:
        logger.info(f"Using models file: {models_file}")
        with open(models_file, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        logger.info(f"Training {len(models)} models")
    else:
        logger.info("Training all available models")
    
    sagemaker = boto3.client('sagemaker')
    
    timestamp = int(datetime.now().timestamp())
    job_name = f"standard-training-{timestamp}"
    
    training_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:latest"
    role_arn = "arn:aws:iam::773934887314:role/SageMakerExecutionRole"
    
    hyperparameters = {
        'max_depth': '6',
        'eta': '0.1',
        'gamma': '4',
        'min_child_weight': '6',
        'subsample': '0.8',
        'objective': 'binary:logistic',
        'num_round': '100'
    }
    
    input_data_config = [
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None'
        }
    ]
    
    output_data_config = {
        'S3OutputPath': f's3://hpo-bucket-773934887314/standard-training-{timestamp}/'
    }
    
    resource_config = {
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30
    }
    
    stopping_condition = {
        'MaxRuntimeInSeconds': 3600
    }
    
    try:
        logger.info(f"🚀 Launching standard training job: {job_name}")
        
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage': training_image,
                'TrainingInputMode': 'File'
            },
            RoleArn=role_arn,
            InputDataConfig=input_data_config,
            OutputDataConfig=output_data_config,
            ResourceConfig=resource_config,
            StoppingCondition=stopping_condition,
            HyperParameters=hyperparameters
        )
        
        logger.info(f"✅ Successfully launched training job: {job_name}")
        logger.info(f"🔗 Job ARN: {response['TrainingJobArn']}")
        
        if deploy:
            logger.info("🚀 Deployment will be handled after training completion")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to launch training job: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Standard SageMaker Training Pipeline')
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy the model after training')
    parser.add_argument('--models-file', type=str,
                        help='File containing list of models to train')
    
    args = parser.parse_args()
    
    success = train_all_stocks(args.deploy, args.models_file)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
