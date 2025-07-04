
import subprocess
import sys
import os

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])

import argparse
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
