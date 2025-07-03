#!/usr/bin/env python3
"""
TabNet model training integration
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    print("⚠️ pytorch-tabnet not available, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pytorch-tabnet'])
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True

class TabNetTrainer:
    def __init__(self, data_path='data/enhanced_features/enhanced_features.csv'):
        self.data_path = data_path
        self.random_state = 42
        
    def load_and_prepare_data(self):
        """Load and prepare data with feature selection and scaling"""
        print("📊 Loading enhanced features dataset...")
        df = pd.read_csv(self.data_path)
        
        y = df['target_next_day'].values
        feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'target_next_day']]
        X = df[feature_cols].values
        
        var_selector = VarianceThreshold(threshold=0.001)
        X_var = var_selector.fit_transform(X)
        
        lasso_selector = SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000), threshold='median')
        X_selected = lasso_selector.fit_transform(X_var, y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        print(f"✅ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y, (var_selector, lasso_selector, scaler)
    
    def train_tabnet_models(self, X, y, selectors):
        """Train TabNet models"""
        print("🚀 Training TabNet models...")
        
        os.makedirs('models/tabnet', exist_ok=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        try:
            tabnet_params = {
                'n_d': 32,
                'n_a': 32,
                'n_steps': 3,
                'gamma': 1.3,
                'lambda_sparse': 1e-3,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': dict(lr=2e-2),
                'mask_type': 'entmax',
                'scheduler_params': {"step_size": 10, "gamma": 0.9},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'verbose': 1,
                'seed': 42
            }
            
            model = TabNetClassifier(**tabnet_params)
            
            model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                eval_name=['val'],
                eval_metric=['auc'],
                max_epochs=100,
                patience=20,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            
            test_pred = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, test_pred)
            
            model_path = 'models/tabnet/tabnet_model.zip'
            model.save_model(model_path)
            
            selectors_path = 'models/tabnet/tabnet_selectors.pkl'
            scaler_path = 'models/tabnet/tabnet_scaler.pkl'
            
            var_selector, lasso_selector, scaler = selectors
            joblib.dump((var_selector, lasso_selector), selectors_path)
            joblib.dump(scaler, scaler_path)
            
            feature_importances = model.feature_importances_
            
            model_metrics = {
                'auc_score': auc_score,
                'best_epoch': model.best_epoch,
                'model_path': model_path,
                'selectors_path': selectors_path,
                'scaler_path': scaler_path,
                'feature_importances': feature_importances.tolist()
            }
            
            print(f"✅ TabNet training completed - AUC: {auc_score:.6f}")
            
            report = {
                'model_metrics': model_metrics,
                'training_timestamp': pd.Timestamp.now().isoformat(),
                'tabnet_params': tabnet_params
            }
            
            with open('models/tabnet/tabnet_training_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return report
            
        except Exception as e:
            print(f"❌ TabNet training failed: {e}")
            return {
                'model_metrics': {},
                'training_timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e)
            }

def main():
    if not TABNET_AVAILABLE:
        print("❌ TabNet not available")
        return None
        
    trainer = TabNetTrainer()
    X, y, selectors = trainer.load_and_prepare_data()
    report = trainer.train_tabnet_models(X, y, selectors)
    return report

if __name__ == "__main__":
    main()
