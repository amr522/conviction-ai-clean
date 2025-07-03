#!/usr/bin/env python3
"""
XGBoost model training with proper early stopping
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

class XGBoostTrainer:
    def __init__(self, data_path='data/enhanced_features/enhanced_features.csv'):
        self.data_path = data_path
        self.random_state = 42
        
    def load_and_prepare_data(self):
        """Load and prepare data with feature selection"""
        print("📊 Loading enhanced features dataset...")
        df = pd.read_csv(self.data_path)
        
        y = df['target_next_day'].values
        feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'target_next_day']]
        X = df[feature_cols].values
        
        var_selector = VarianceThreshold(threshold=0.001)
        X_var = var_selector.fit_transform(X)
        
        lasso_selector = SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000), threshold='median')
        X_selected = lasso_selector.fit_transform(X_var, y)
        
        print(f"✅ Data prepared: {X_selected.shape[0]} samples, {X_selected.shape[1]} features")
        return X_selected, y, (var_selector, lasso_selector)
    
    def train_xgboost_models(self, X, y, selectors):
        """Train XGBoost models with proper early stopping"""
        print("🚀 Training XGBoost models...")
        
        os.makedirs('models/xgboost', exist_ok=True)
        
        model_configs = [
            {
                'name': 'xgb_hpo_1',
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbosity': 0
                }
            },
            {
                'name': 'xgb_hpo_2',
                'params': {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 43,
                    'verbosity': 0
                }
            },
            {
                'name': 'xgb_hpo_3',
                'params': {
                    'n_estimators': 250,
                    'learning_rate': 0.08,
                    'max_depth': 7,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 44,
                    'verbosity': 0
                }
            }
        ]
        
        trained_models = []
        model_metrics = {}
        
        for config in model_configs:
            print(f"📈 Training {config['name']}...")
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            try:
                model = xgb.XGBClassifier(**config['params'])
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                val_pred = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, val_pred)
                
                model_path = f"models/xgboost/{config['name']}.pkl"
                joblib.dump(model, model_path)
                
                selectors_path = f"models/xgboost/{config['name']}_selectors.pkl"
                joblib.dump(selectors, selectors_path)
                
                model_metrics[config['name']] = {
                    'auc_score': auc_score,
                    'best_iteration': getattr(model, 'best_iteration', None),
                    'model_path': model_path,
                    'selectors_path': selectors_path
                }
                
                trained_models.append(config['name'])
                print(f"✅ {config['name']} - AUC: {auc_score:.6f}")
                
            except Exception as e:
                print(f"❌ {config['name']} failed: {e}")
                continue
        
        report = {
            'trained_models': trained_models,
            'model_metrics': model_metrics,
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('models/xgboost/xgboost_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ XGBoost training completed: {len(trained_models)} models")
        return report

def main():
    trainer = XGBoostTrainer()
    X, y, selectors = trainer.load_and_prepare_data()
    report = trainer.train_xgboost_models(X, y, selectors)
    return report

if __name__ == "__main__":
    main()
