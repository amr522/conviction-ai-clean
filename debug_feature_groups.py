#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from datetime import datetime

def load_symbol_data(symbol):
    """Load individual symbol data from processed files"""
    data_dir = Path("data/processed_with_news_20250628")
    file_path = data_dir / f"{symbol}_features.csv"
    
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    future_columns = ['target_1d', 'target_3d', 'target_5d', 'target_10d']
    df = df.drop(columns=[col for col in future_columns if col in df.columns])
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df

def debug_lagged_features():
    """Debug lagged features creation for AAPL"""
    print("=== Debugging Lagged Features for AAPL ===")
    
    df = load_symbol_data('AAPL')
    if df is None:
        print("Failed to load AAPL data")
        return
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    enhanced_df = df.copy()
    
    enhanced_df['ret_1d_lag1'] = enhanced_df['close'].pct_change(1).shift(1)
    enhanced_df['ret_3d_lag1'] = enhanced_df['close'].pct_change(3).shift(1)
    enhanced_df['ret_5d_lag1'] = enhanced_df['close'].pct_change(5).shift(1)
    
    enhanced_df['vol_5d_lag1'] = enhanced_df['close'].pct_change().rolling(5).std().shift(1)
    enhanced_df['vol_10d_lag1'] = enhanced_df['close'].pct_change().rolling(10).std().shift(1)
    
    enhanced_df['price_mom_5d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(5)).shift(1)
    enhanced_df['price_mom_10d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(10)).shift(1)
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    print(f"Enhanced data shape before dropna: {enhanced_df.shape}")
    
    enhanced_df = enhanced_df.dropna()
    print(f"Enhanced data shape after dropna: {enhanced_df.shape}")
    
    basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target_next_day', 'symbol']
    feature_cols = [col for col in enhanced_df.columns if col not in basic_cols]
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    if len(feature_cols) == 0:
        print("❌ No feature columns found!")
        return
    
    X = enhanced_df[feature_cols]
    y = enhanced_df['target_next_day']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Missing values: {missing_counts[missing_counts > 0].to_dict()}")
    
    X_filled = X.fillna(0)
    
    inf_counts = np.isinf(X_filled).sum()
    if inf_counts.sum() > 0:
        print(f"Infinite values: {inf_counts[inf_counts > 0].to_dict()}")
        X_filled = X_filled.replace([np.inf, -np.inf], 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    
    print(f"Scaled features shape: {X_scaled.shape}")
    print(f"Scaled features stats: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
    
    print("\n=== Testing Simple LightGBM Training ===")
    
    try:
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=10  # Reduce for faster testing
        )
        
        split_idx = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        lgb_model.fit(X_train, y_train)
        
        train_score = lgb_model.score(X_train, y_train)
        test_score = lgb_model.score(X_test, y_test)
        
        print(f"✅ Simple training successful!")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        print("\n=== Testing 2-Fold Cross-Validation ===")
        
        cv_scores = cross_val_score(
            lgb_model, X_scaled, y, 
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=1  # Use single job to avoid parallelization issues
        )
        
        print(f"✅ 2-fold CV successful!")
        print(f"CV AUC scores: {cv_scores}")
        print(f"Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
    except Exception as e:
        print(f"❌ LightGBM training failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Debug the feature group evaluation issues"""
    debug_lagged_features()

if __name__ == "__main__":
    main()
