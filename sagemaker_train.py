#!/usr/bin/env python3
"""
SageMaker-compatible training script for Combined Features HPO
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--feature_fraction', type=float, default=1.0)
    parser.add_argument('--bagging_fraction', type=float, default=1.0)
    
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--min_child_samples', type=int, default=20)
    parser.add_argument('--lambda_l1', type=float, default=0.0)
    parser.add_argument('--lambda_l2', type=float, default=0.0)
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        logger.info(f"Ignoring unknown hyperparameters: {unknown}")
    
    return args

def load_data(data_path):
    """Load training data from CSV"""
    train_file = os.path.join(data_path, 'train.csv')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    logger.info(f"Loading training data from {train_file}")
    df = pd.read_csv(train_file)
    
    if 'target_next_day' in df.columns:
        target_col = 'target_next_day'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]
    
    feature_cols = [col for col in df.columns if col not in [target_col, 'date', 'symbol', 'Date', 'Symbol']]
    
    X = df[feature_cols].fillna(0)
    y_raw = df[target_col]
    
    y = (y_raw > 0).astype(int)
    
    logger.info(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_model(X_train, y_train, args):
    """Train RandomForest model with given hyperparameters"""
    logger.info("Training RandomForest model...")
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth if args.max_depth > 0 else None,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    
    logger.info(f"Training AUC: {train_auc:.4f}")
    
    return model, train_auc

def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation data"""
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    logger.info(f"Validation AUC: {val_auc:.4f}")
    
    print(f"validation:auc={val_auc:.6f}")
    
    return val_auc

def save_model(model, model_dir, feature_cols):
    """Save model and metadata"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    feature_path = os.path.join(model_dir, 'feature_names.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_cols, f)
    
    logger.info(f"Model saved to {model_path}")

def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting SageMaker training...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        X_train, y_train, feature_cols = load_data(args.train)
        
        model, train_auc = train_model(X_train, y_train, args)
        
        if args.validation and os.path.exists(os.path.join(args.validation, 'validation.csv')):
            val_file = os.path.join(args.validation, 'validation.csv')
            logger.info(f"Loading validation data from {val_file}")
            df_val = pd.read_csv(val_file)
            
            if 'target_next_day' in df_val.columns:
                target_col = 'target_next_day'
            elif 'target' in df_val.columns:
                target_col = 'target'
            else:
                target_col = df_val.columns[-1]
            
            X_val = df_val[feature_cols].fillna(0)
            y_val_raw = df_val[target_col]
            
            y_val = (y_val_raw > 0).astype(int)
            
            val_auc = evaluate_model(model, X_val, y_val)
        else:
            logger.info("No separate validation data found, using training data")
            val_auc = evaluate_model(model, X_train, y_train)
        
        save_model(model, args.model_dir, feature_cols)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
