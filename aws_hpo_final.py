#!/usr/bin/env python3

import os
import json
import pandas as pd
from pathlib import Path

# Get actual symbols
def get_symbols():
    data_dir = Path("data/processed_with_news_20250628")
    symbols = [f.stem.replace('_features', '') for f in data_dir.glob("*_features.csv")]
    return sorted(symbols)

SYMBOLS = get_symbols()
SYMBOL_COUNT = len(SYMBOLS)
TRIALS_PER_SYMBOL = 25
PARALLEL_JOBS = 20
INSTANCE_TYPE = 'ml.c5.xlarge'
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
BUCKET = 'your-hpo-bucket'
S3_DATA_PREFIX = f's3://{BUCKET}/data/'
S3_OUTPUT_PREFIX = f's3://{BUCKET}/models/'

def create_training_script():
    """Create SageMaker training script with all 38 features"""
    
    script = '''#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import argparse

def objective_with_news(trial, X, y, model_type):
    # News processing
    Xp = X.copy()
    lookback = trial.suggest_int('lookback_window', 1, 5)
    Xp['news_sent_lb'] = Xp['news_sent'].rolling(lookback).mean().fillna(0)
    
    # Filter by threshold
    thr = trial.suggest_float('news_threshold', 0.01, 0.1)
    mask = Xp['news_sent_lb'].abs() > thr
    if mask.sum() > 100:
        Xf = Xp.loc[mask]
        yf = y.loc[mask]
    else:
        Xf, yf = Xp, y
    
    # Model params
    if model_type == 'extra_trees':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        model = ExtraTreesClassifier(**params, n_jobs=1, random_state=42)
    elif model_type == 'lightgbm':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        }
        model = lgb.LGBMClassifier(**params, n_jobs=1, random_state=42)
    
    # Cross-validate
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for ti, vi in cv.split(Xf, yf):
        model.fit(Xf.iloc[ti], yf.iloc[ti])
        pred = model.predict_proba(Xf.iloc[vi])[:,1]
        scores.append(roc_auc_score(yf.iloc[vi], pred))
    
    return float(np.mean(scores))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='extra_trees')
    args = parser.parse_args()

    # Load data
    data_path = os.path.join(os.environ['SM_CHANNEL_TRAINING'], f"{args.symbol}_features.csv")
    df = pd.read_csv(data_path)
    
    # All 38 feature columns (excluding timestamp, direction, next_ret)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'direction', 'next_ret']]
    X = df[feature_cols].fillna(0)
    y = df['direction']
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_with_news(trial, X, y, args.model_type), n_trials=25)
    
    auc = study.best_value
    
    # Save results
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'auc': float(auc),
        'best_params': study.best_params,
        'symbol': args.symbol,
        'model_type': args.model_type
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    
    print(f"AUC: {auc}")

if __name__ == '__main__':
    train()
'''
    
    with open('sagemaker_train_final.py', 'w') as f:
        f.write(script)
    print("✅ Created sagemaker_train_final.py")

def create_launcher():
    """Create launcher for all 241 symbols"""
    
    launcher = f'''#!/usr/bin/env python3
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

session = sagemaker.Session()
role = '{ROLE_ARN}'

symbols = {SYMBOLS}

for symbol in symbols:
    estimator = SKLearn(
        entry_point='sagemaker_train_final.py',
        role=role,
        instance_type='{INSTANCE_TYPE}',
        instance_count=1,
        framework_version='0.23-1',
        py_version='py3',
        output_path='{S3_OUTPUT_PREFIX}',
        hyperparameters={{'symbol': symbol, 'model_type': 'extra_trees'}}
    )
    
    # Launch training job
    estimator.fit({{'training': TrainingInput(f'{S3_DATA_PREFIX}{{symbol}}_features.csv', content_type='text/csv')}}, 
                  job_name=f'hpo-{{symbol}}-{{int(time.time())}}', wait=False)
    print(f'Started HPO for {{symbol}}')
    
    # Rate limiting
    time.sleep(2)
'''
    
    with open('launch_hpo_final.py', 'w') as f:
        f.write(launcher)
    print("✅ Created launch_hpo_final.py")

def main():
    print(f"🚀 AWS HPO FINAL SETUP")
    print("=" * 25)
    print(f"Symbols: {SYMBOL_COUNT}")
    print(f"Features: 38 per symbol")
    print(f"Total trials: {SYMBOL_COUNT * TRIALS_PER_SYMBOL}")
    
    create_training_script()
    create_launcher()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Update ROLE_ARN and BUCKET in script")
    print(f"2. aws s3 sync data/processed_with_news_20250628/ s3://your-bucket/data/")
    print(f"3. python launch_hpo_final.py")

if __name__ == '__main__':
    main()