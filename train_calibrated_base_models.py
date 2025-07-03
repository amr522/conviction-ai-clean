#!/usr/bin/env python3
"""
Calibrated base model training using CalibratedClassifierCV
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

class CalibratedBaseTrainer:
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
    
    def train_calibrated_models(self, X, y, selectors):
        """Train calibrated base models"""
        print("🚀 Training calibrated base models...")
        
        os.makedirs('models/calibrated_base', exist_ok=True)
        
        base_configs = [
            {
                'name': 'lightgbm_base_01',
                'model': lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
            },
            {
                'name': 'lightgbm_base_02',
                'model': lgb.LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=43,
                    verbose=-1
                )
            },
            {
                'name': 'lightgbm_base_03',
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.08,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=44,
                    verbose=-1
                )
            }
        ]
        
        trained_models = []
        model_metrics = {}
        
        for config in base_configs:
            print(f"📈 Training {config['name']}...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            try:
                base_model = config['model']
                base_model.fit(X_train, y_train)
                
                calibrated_model = CalibratedClassifierCV(
                    base_model, 
                    method='sigmoid',
                    cv=3
                )
                
                calibrated_model.fit(X_train, y_train)
                
                test_pred = calibrated_model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, test_pred)
                
                model_path = f"models/calibrated_base/{config['name']}_calibrated.pkl"
                joblib.dump(calibrated_model, model_path)
                
                selectors_path = f"models/calibrated_base/{config['name']}_selectors.pkl"
                joblib.dump(selectors, selectors_path)
                
                model_metrics[config['name']] = {
                    'auc_score': auc_score,
                    'model_path': model_path,
                    'selectors_path': selectors_path,
                    'calibration_method': 'sigmoid'
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
        
        with open('models/calibrated_base/calibrated_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Calibrated base training completed: {len(trained_models)} models")
        return report

def main():
    trainer = CalibratedBaseTrainer()
    X, y, selectors = trainer.load_and_prepare_data()
    report = trainer.train_calibrated_models(X, y, selectors)
    return report

if __name__ == "__main__":
    main()
