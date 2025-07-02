#!/usr/bin/env python3
"""
Updated AWS HPO Hyperparameter Setup

This script contains all the hyperparameter ranges for the AWS SageMaker HPO job.
It includes advanced options-specific hyperparameters.
"""

import os
import json
import boto3
import sagemaker
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# === CONFIGURATION ===
SYMBOL_COUNT = 241
TRIALS_PER_SYMBOL = 25
PARALLEL_JOBS = 20
INSTANCE_TYPE = 'ml.c5.xlarge'  # CPU instance; consider ml.g4dn.4xlarge for GPU
ROLE_ARN = 'arn:aws:iam::123456789012:role/YourSageMakerExecutionRole'
BUCKET = 'your-hpo-bucket'
S3_DATA_PREFIX = f's3://{BUCKET}/data/'
S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/'

def get_hpo_params():
    """
    Define the full hyperparameter tuning space for the options-oriented ML pipeline
    Includes all required advanced parameters
    """
    hyperparameter_ranges = {
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
    
    return hyperparameter_ranges
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()

    # Load feature data for this symbol
    data_path = os.path.join(os.environ['SM_CHANNEL_TRAINING'], f"{args.symbol}_features.csv")
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ['timestamp','direction','next_ret']]
    X = df[feature_cols].fillna(0)
    y = df['direction']

    # Initialize and cross-validate model
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
    auc = float(np.mean(scores))

    # Write metrics in JSON for SageMaker
    metrics = {{'auc': auc}}
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(f"AUC: {{auc}}")

if __name__ == '__main__':
    train()
'''
    with open('sagemaker_train.py', 'w') as f:
        f.write(script)
    os.chmod('sagemaker_train.py', 0o755)
    print("✅ Created 'sagemaker_train.py'")


def create_hpo_launcher():
    """Generate the SageMaker HPO launcher script 'launch_hpo.py'."""
    symbols = [f"SYM{{i}}" for i in range(1, SYMBOL_COUNT+1)]  # replace with real symbols list or load from file
    launcher = f'''#!/usr/bin/env python3
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# Initialize SageMaker session and role
session = sagemaker.Session()
role = '{ROLE_ARN}'

# Define estimator
estimator = SKLearn(
    entry_point='sagemaker_train.py',
    role=role,
    instance_type='{INSTANCE_TYPE}',
    instance_count=1,
    framework_version='0.23-1',
    py_version='py3',
    output_path='{S3_OUTPUT_PREFIX}'
)

# Hyperparameter ranges
hyperparameter_ranges = {{
    'n_estimators': IntegerParameter(100, 1000),
    'max_depth': IntegerParameter(5, 20),
    'min_samples_split': IntegerParameter(2, 10)
}}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs={PARALLEL_JOBS * len(symbols)},
    max_parallel_jobs={PARALLEL_JOBS}
)

# Launch HPO for each symbol
''' + '
'.join([f"tuner.fit({{'training': TrainingInput('{S3_DATA_PREFIX}{sym}_features.csv', content_type='text/csv')}}, wait=False)
print('Started HPO for {sym}')" for sym in symbols]) + "
"  
    with open('launch_hpo.py', 'w') as f:
        f.write(launcher)
    os.chmod('launch_hpo.py', 0o755)
    print("✅ Created 'launch_hpo.py'")


if __name__ == '__main__':
    setup_aws_hpo()
    create_training_script()
    create_hpo_launcher()
    print("\nAll scripts generated. Run 'launch_hpo.py' to start your HPO jobs.")
'''
    
    # Note: adjust real symbol list instead of placeholder SYM1..SYM242

def main():
    setup_aws_hpo()
    create_training_script()
    create_hpo_launcher()

if __name__ == '__main__':
    main()
