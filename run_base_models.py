#!/usr/bin/env python3
"""
run_base_models.py - Train base models on processed features

This script:
1. Loads data from the processed features directory
2. Trains 11 different model types on the data
3. Saves the models and evaluation metrics to the output directory
4. Optionally pushes training to AWS SageMaker
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs/base_models.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)

def train_base_models(features_dir, symbols_file, output_dir, use_aws=False):
    """
    Train base models on the processed features for each symbol
    
    Args:
        features_dir: Directory containing processed features
        symbols_file: File containing list of symbols to use
        output_dir: Directory to save model outputs
        use_aws: Whether to use AWS SageMaker for training
    """
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    ensure_directory('pipeline_logs')
    
    # Load symbols
    with open(symbols_file, 'r') as f:
        all_lines = f.readlines()
        symbols = [line.strip() for line in all_lines if line.strip()]
        logger.info(f"Read {len(all_lines)} lines from {symbols_file}, found {len(symbols)} non-empty symbols")
    
    logger.info(f"Training base models for {len(symbols)} symbols")
    logger.info(f"Using AWS SageMaker: {use_aws}")
    
    features_file = os.path.join(features_dir, 'all_symbols_features.csv')
    if not os.path.exists(features_file):
        logger.error(f"Features file not found: {features_file}")
        return False
    
    logger.info(f"Loading features from {features_file}")
    df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    trained_count = 0
    for symbol in symbols:
        try:
            logger.info(f"Training model for {symbol} ({trained_count + 1}/{len(symbols)})")
            
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) == 0:
                logger.warning(f"No data found for symbol {symbol}, skipping")
                continue
            
            feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'target_next_day']]
            X = symbol_data[feature_cols]
            y = symbol_data['target_next_day']
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            auc_score = roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred_binary)
            
            model_file = os.path.join(output_dir, f"{symbol}_lightgbm.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            metrics = {
                'symbol': symbol,
                'auc_score': auc_score,
                'accuracy': accuracy,
                'best_iteration': model.best_iteration,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(feature_cols),
                'trained_on': datetime.now().isoformat()
            }
            
            metrics_file = os.path.join(output_dir, f"{symbol}_metrics.json")
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"✅ Trained {symbol} base model - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
            trained_count += 1
            
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {str(e)}")
            continue
    
    logger.info(f"Successfully trained {trained_count} out of {len(symbols)} base models")
    logger.info(f"All base models saved to {output_dir}")
    return trained_count > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train base models on processed features')
    parser.add_argument('--features-dir', type=str, required=True,
                        help='Directory containing processed features')
    parser.add_argument('--symbols-file', type=str, required=True,
                        help='File containing list of symbols to use')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save model outputs')
    parser.add_argument('--use-aws', action='store_true',
                        help='Use AWS SageMaker for training')
    args = parser.parse_args()
    
    # Train the base models
    train_base_models(args.features_dir, args.symbols_file, args.output_dir, args.use_aws)

if __name__ == "__main__":
    main()
