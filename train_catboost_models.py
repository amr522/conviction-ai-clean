#!/usr/bin/env python3
"""
CatBoost model training with proper early stopping and memory management
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

class CatBoostTrainer:
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
    
    def train_catboost_models(self, X, y, selectors):
        """Train multiple CatBoost models with proper configuration"""
        print("🚀 Training CatBoost models...")
        
        os.makedirs('models/catboost', exist_ok=True)
        
        model_configs = [
            {
                'name': 'catboost_hpo_1',
                'params': {
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 4,
                    'random_seed': 42,
                    'verbose': False,
                    'early_stopping_rounds': 20,
                    'task_type': 'CPU',
                    'thread_count': 2,
                    'max_ctr_complexity': 1
                }
            },
            {
                'name': 'catboost_hpo_2', 
                'params': {
                    'iterations': 150,
                    'learning_rate': 0.05,
                    'depth': 5,
                    'random_seed': 43,
                    'verbose': False,
                    'early_stopping_rounds': 25,
                    'task_type': 'CPU',
                    'thread_count': 2,
                    'max_ctr_complexity': 2
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
                model = CatBoostClassifier(**config['params'])
                
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=False,
                    plot=False
                )
                
                val_pred = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, val_pred)
                
                model_path = f"models/catboost/{config['name']}.pkl"
                joblib.dump(model, model_path)
                
                selectors_path = f"models/catboost/{config['name']}_selectors.pkl"
                joblib.dump(selectors, selectors_path)
                
                model_metrics[config['name']] = {
                    'auc_score': auc_score,
                    'best_iteration': model.get_best_iteration(),
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
        
        with open('models/catboost/catboost_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ CatBoost training completed: {len(trained_models)} models")
        return report

def main():
    trainer = CatBoostTrainer()
    X, y, selectors = trainer.load_and_prepare_data()
    report = trainer.train_catboost_models(X, y, selectors)
    return report

if __name__ == "__main__":
    main()
