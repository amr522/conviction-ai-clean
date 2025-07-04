#!/usr/bin/env python3
"""
Launch LightGBM HPO job on SageMaker
"""
import boto3
import json
import time
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def launch_lightgbm_hpo(input_data_s3: str, dry_run: bool = False) -> str | None:
    """Launch LightGBM hyperparameter optimization job"""
    
    timestamp = int(time.time())
    job_name = f"lgbm-hpo-{timestamp}"
    
    if dry_run:
        logger.info(f"🧪 DRY RUN: Would launch LightGBM HPO job: {job_name}")
        return job_name
    
    sagemaker = boto3.client('sagemaker')
    
    hyperparameter_ranges = {
        'num_leaves': {
            'Name': 'num_leaves',
            'Type': 'Integer',
            'MinValue': '10',
            'MaxValue': '300'
        },
        'learning_rate': {
            'Name': 'learning_rate',
            'Type': 'Continuous',
            'MinValue': '0.01',
            'MaxValue': '0.3'
        },
        'feature_fraction': {
            'Name': 'feature_fraction',
            'Type': 'Continuous',
            'MinValue': '0.4',
            'MaxValue': '1.0'
        },
        'bagging_fraction': {
            'Name': 'bagging_fraction',
            'Type': 'Continuous',
            'MinValue': '0.4',
            'MaxValue': '1.0'
        },
        'bagging_freq': {
            'Name': 'bagging_freq',
            'Type': 'Integer',
            'MinValue': '1',
            'MaxValue': '7'
        },
        'min_child_samples': {
            'Name': 'min_child_samples',
            'Type': 'Integer',
            'MinValue': '5',
            'MaxValue': '100'
        },
        'reg_alpha': {
            'Name': 'reg_alpha',
            'Type': 'Continuous',
            'MinValue': '0',
            'MaxValue': '10'
        },
        'reg_lambda': {
            'Name': 'reg_lambda',
            'Type': 'Continuous',
            'MinValue': '0',
            'MaxValue': '10'
        }
    }
    
    static_hyperparameters = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_boost_round': '1000',
        'early_stopping_rounds': '50',
        'verbose': '-1'
    }
    
    training_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-lightgbm:latest"
    
    role_arn = "arn:aws:iam::773934887314:role/SageMakerExecutionRole"
    
    output_path = f"s3://hpo-bucket-773934887314/lightgbm-hpo-{timestamp}/"
    
    hpo_config = {
        'HyperParameterTuningJobName': job_name,
        'HyperParameterTuningJobConfig': {
            'Strategy': 'Bayesian',
            'HyperParameterTuningJobObjective': {
                'Type': 'Maximize',
                'MetricName': 'validation:auc'
            },
            'ResourceLimits': {
                'MaxNumberOfTrainingJobs': 50,
                'MaxParallelTrainingJobs': 5
            },
            'ParameterRanges': {
                'IntegerParameterRanges': [
                    hyperparameter_ranges['num_leaves'],
                    hyperparameter_ranges['bagging_freq'],
                    hyperparameter_ranges['min_child_samples']
                ],
                'ContinuousParameterRanges': [
                    hyperparameter_ranges['learning_rate'],
                    hyperparameter_ranges['feature_fraction'],
                    hyperparameter_ranges['bagging_fraction'],
                    hyperparameter_ranges['reg_alpha'],
                    hyperparameter_ranges['reg_lambda']
                ]
            }
        },
        'TrainingJobDefinition': {
            'AlgorithmSpecification': {
                'TrainingImage': training_image,
                'TrainingInputMode': 'File'
            },
            'RoleArn': role_arn,
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_data_s3,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': output_path
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            },
            'StaticHyperParameters': static_hyperparameters
        }
    }
    
    try:
        logger.info(f"🚀 Launching LightGBM HPO job: {job_name}")
        logger.info(f"📊 Input data: {input_data_s3}")
        logger.info(f"📁 Output path: {output_path}")
        
        response = sagemaker.create_hyper_parameter_tuning_job(**hpo_config)
        
        logger.info(f"✅ Successfully launched LightGBM HPO job: {job_name}")
        logger.info(f"🔗 Job ARN: {response['HyperParameterTuningJobArn']}")
        
        return job_name
        
    except Exception as e:
        logger.error(f"❌ Failed to launch LightGBM HPO job: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch LightGBM HPO job on SageMaker')
    parser.add_argument('--input-data-s3', type=str, required=True,
                        help='S3 URI for training data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without launching actual job')
    
    args = parser.parse_args()
    
    job_name = launch_lightgbm_hpo(args.input_data_s3, args.dry_run)
    
    if job_name:
        print(f"✅ LightGBM HPO job launched: {job_name}")
    else:
        print("❌ Failed to launch LightGBM HPO job")
        exit(1)

if __name__ == "__main__":
    main()
