#!/usr/bin/env python3
"""
Extract best CatBoost hyperparameters from completed HPO job
"""
import json
import boto3
import os

def extract_best_hyperparams():
    sagemaker = boto3.client('sagemaker')
    
    try:
        response = sagemaker.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName='catboost-hpo-46-1751615821'
        )
        
        best_job = response['BestTrainingJob']
        hyperparams = best_job['TunedHyperParameters']
        objective_value = best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']
        
        config = {
            'algorithm': 'catboost',
            'job_name': 'catboost-hpo-46-1751615821',
            'best_training_job': best_job['TrainingJobName'],
            'objective_value': float(objective_value),
            'hyperparameters': hyperparams,
            'timestamp': '1751615821'
        }
        
        os.makedirs('configs/hpo', exist_ok=True)
        
        with open('configs/hpo/best_full_catboost_hyperparams.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print('✅ Created configs/hpo/best_full_catboost_hyperparams.json')
        print(f'📊 Best AUC: {objective_value}')
        print(f'🏆 Best job: {best_job["TrainingJobName"]}')
        
        return config
        
    except Exception as e:
        print(f'❌ Failed to extract hyperparameters: {e}')
        return None

if __name__ == '__main__':
    extract_best_hyperparams()
