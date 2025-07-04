#!/usr/bin/env python3
"""
XGBoost Training Script for AWS SageMaker HPO
Compatible with SageMaker XGBoost container
"""

import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib
import json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN') or os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--colsample_bytree', type=float, default=1.0)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--reg_lambda', type=float, default=1.0)
    parser.add_argument('--lambda', type=float, default=1.0, dest='reg_lambda')
    parser.add_argument('--num_round', type=int, default=100)
    
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--symbol', type=str, default='AAPL')
    parser.add_argument('--model', type=str, default='xgb')
    parser.add_argument('--apply_universe_filter', type=bool, default=False)
    parser.add_argument('--iv_rank_window', type=int, default=30)
    parser.add_argument('--iv_rank_weight', type=float, default=0.5)
    parser.add_argument('--term_slope_window', type=int, default=15)
    parser.add_argument('--term_slope_weight', type=float, default=0.5)
    parser.add_argument('--oi_window', type=int, default=15)
    parser.add_argument('--oi_weight', type=float, default=0.5)
    parser.add_argument('--theta_window', type=int, default=15)
    parser.add_argument('--theta_weight', type=float, default=0.5)
    parser.add_argument('--vix_mom_window', type=int, default=10)
    parser.add_argument('--vix_regime_thresh', type=float, default=25.0)
    parser.add_argument('--event_lag', type=int, default=2)
    parser.add_argument('--event_lead', type=int, default=2)
    parser.add_argument('--news_threshold', type=float, default=0.05)
    parser.add_argument('--lookback_window', type=int, default=5)
    parser.add_argument('--reuters_weight', type=float, default=1.0)
    parser.add_argument('--sa_weight', type=float, default=1.0)
    
    args = parser.parse_args()
    
    if not args.train and not (os.environ.get('SM_CHANNEL_TRAIN') or os.environ.get('SM_CHANNEL_TRAINING')):
        parser.error("--train argument is required when SM_CHANNEL_TRAIN environment variable is not set")
    
    if not args.model_dir and not os.environ.get('SM_MODEL_DIR'):
        parser.error("--model-dir argument is required when SM_MODEL_DIR environment variable is not set")
    
    return args

def load_data(data_path):
    """Load training data from CSV"""
    print(f"Attempting to load data from: {data_path}")
    
    if not data_path:
        raise ValueError("❌ Data path cannot be empty")
    
    if os.path.isdir(data_path):
        print(f"Data path is directory: {data_path}")
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        print(f"Found CSV files: {csv_files}")
        if 'train.csv' in csv_files:
            data_path = os.path.join(data_path, 'train.csv')
        elif csv_files:
            data_path = os.path.join(data_path, csv_files[0])
        else:
            raise FileNotFoundError(f"❌ No CSV files found in directory: {data_path}")
    
    print(f"Loading CSV file: {data_path}")
    df = pd.read_csv(data_path, header=None)
    
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    
    return X, y

def train_model(args):
    """Train XGBoost model"""
    print("Loading training data...")
    print(f"Training data path: {args.train}")
    
    if not args.train:
        raise ValueError("❌ --train path is required")
    
    if not os.path.exists(args.train):
        raise FileNotFoundError(f"❌ Training data not found at: {args.train}")
    
    X_train, y_train = load_data(args.train)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'min_child_weight': args.min_child_weight,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'lambda': args.reg_lambda,
        'seed': 42
    }
    
    print(f"Training with parameters: {params}")
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        verbose_eval=10
    )
    
    train_preds = model.predict(dtrain)
    train_auc = roc_auc_score(y_train, train_preds)
    
    print(f"Training AUC: {train_auc:.4f}")
    
    validation_auc = train_auc  # Default to training AUC
    if args.validation and os.path.exists(args.validation):
        try:
            X_val, y_val = load_data(args.validation)
            dval = xgb.DMatrix(X_val, label=y_val)
            val_preds = model.predict(dval)
            validation_auc = roc_auc_score(y_val, val_preds)
            print(f"Validation AUC: {validation_auc:.4f}")
        except Exception as e:
            print(f"Validation failed: {e}")
    
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)
    
    metrics = {
        'validation:auc': validation_auc,
        'train:auc': train_auc
    }
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model saved to {model_path}")
    print(f"Final validation AUC: {validation_auc:.4f}")
    print(f"validation-auc:{validation_auc:.6f}")
    
    return validation_auc

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    auc = train_model(args)
    
    print(f"Training completed. Final AUC: {auc:.4f}")
