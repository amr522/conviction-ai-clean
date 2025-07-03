#!/usr/bin/env python3
"""
Out-of-Fold (OOF) Generation with k-fold cross-validation
Handles early stopping correctly for all model types
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")

class OOFGenerator:
    def __init__(self, data_path='data/enhanced_features/enhanced_features.csv', n_folds=5):
        self.data_path = data_path
        self.n_folds = n_folds
        self.random_state = 42
        
    def load_data(self):
        """Load enhanced features dataset"""
        print("📊 Loading enhanced features dataset...")
        
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            
            if 'target_next_day' in df.columns:
                y = df['target_next_day'].values
                feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'target_next_day']]
                X = df[feature_cols].values
                symbols = df['symbol'].values if 'symbol' in df.columns else None
                print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
                return X, y, feature_cols, symbols
        
        raise FileNotFoundError(f"Enhanced features dataset not found at {self.data_path}")
    
    def apply_feature_selection(self, X, y, threshold=0.001):
        """Apply feature selection with safeguards"""
        print("🔍 Applying feature selection...")
        
        var_selector = VarianceThreshold(threshold=threshold)
        X_var_selected = var_selector.fit_transform(X)
        
        print(f"📊 Variance threshold: {X.shape[1]} -> {X_var_selected.shape[1]} features")
        
        if X_var_selected.shape[1] == 0:
            print("⚠️ Variance threshold too aggressive, using original features")
            X_var_selected = X
            var_selector = None
        
        lasso_selector = SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000), threshold='median')
        X_selected = lasso_selector.fit_transform(X_var_selected, y)
        
        print(f"📊 L1 selection: {X_var_selected.shape[1]} -> {X_selected.shape[1]} features")
        
        if X_selected.shape[1] < 10:  # Ensure minimum features
            print("⚠️ L1 selection too aggressive, using 50% threshold")
            lasso_selector = SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000), threshold='0.5*median')
            X_selected = lasso_selector.fit_transform(X_var_selected, y)
            print(f"📊 Adjusted L1 selection: {X_selected.shape[1]} features")
        
        return X_selected, var_selector, lasso_selector
    
    def create_model_configs(self):
        """Create model configurations for OOF generation"""
        configs = {
            'lightgbm_models': [
                {
                    'name': f'lgb_oof_{i}',
                    'model_class': lgb.LGBMClassifier,
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'random_state': 42 + i,
                        'verbose': -1
                    },
                    'use_calibration': True
                } for i in range(1, 6)
            ],
            'xgboost_models': [
                {
                    'name': f'xgb_oof_{i}',
                    'model_class': xgb.XGBClassifier,
                    'params': {
                        'n_estimators': 200,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42 + i,
                        'verbosity': 0
                    },
                    'use_early_stopping': True,
                    'early_stopping_rounds': 50
                } for i in range(1, 4)
            ],
            'catboost_models': [
                {
                    'name': f'cat_oof_{i}',
                    'model_class': CatBoostClassifier,
                    'params': {
                        'iterations': 200,
                        'learning_rate': 0.1,
                        'depth': 6,
                        'random_seed': 42 + i,
                        'verbose': False
                    },
                    'use_early_stopping': True,
                    'early_stopping_rounds': 50
                } for i in range(1, 4)
            ]
        }
        return configs
    
    def train_model_with_early_stopping(self, model_config, X_train, y_train, X_val, y_val):
        """Train model with proper early stopping handling"""
        model_class = model_config['model_class']
        params = model_config['params'].copy()
        
        if model_config.get('use_early_stopping', False):
            if model_class == xgb.XGBClassifier:
                model = model_class(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif model_class == CatBoostClassifier:
                params['early_stopping_rounds'] = model_config['early_stopping_rounds']
                model = model_class(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
            else:
                model = model_class(**params)
                model.fit(X_train, y_train)
        else:
            model = model_class(**params)
            model.fit(X_train, y_train)
        
        if model_config.get('use_calibration', False):
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train, y_train)
            return calibrated_model
        
        return model
    
    def generate_oof_predictions(self, X, y, model_configs):
        """Generate out-of-fold predictions for all models"""
        print(f"🔄 Generating OOF predictions with {self.n_folds}-fold CV...")
        
        n_samples = len(y)
        all_model_names = []
        all_oof_predictions = []
        
        all_configs = []
        for model_type, configs in model_configs.items():
            all_configs.extend(configs)
            all_model_names.extend([config['name'] for config in configs])
        
        oof_matrix = np.zeros((n_samples, len(all_configs)))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        model_metrics = {}
        
        for model_idx, model_config in enumerate(all_configs):
            model_name = model_config['name']
            print(f"📈 Training {model_name}...")
            
            fold_predictions = np.zeros(n_samples)
            fold_aucs = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                if model_config.get('use_early_stopping', False):
                    X_train_split, X_early_val, y_train_split, y_early_val = train_test_split(
                        X_train_fold, y_train_fold, test_size=0.2, random_state=42, stratify=y_train_fold
                    )
                    
                    model = self.train_model_with_early_stopping(
                        model_config, X_train_split, y_train_split, X_early_val, y_early_val
                    )
                else:
                    model = self.train_model_with_early_stopping(
                        model_config, X_train_fold, y_train_fold, None, None
                    )
                
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(X_val_fold)[:, 1]
                else:
                    val_pred = model.predict(X_val_fold)
                
                fold_predictions[val_idx] = val_pred
                fold_auc = roc_auc_score(y_val_fold, val_pred)
                fold_aucs.append(fold_auc)
                
                print(f"    Fold {fold_idx + 1}: AUC = {fold_auc:.6f}")
            
            oof_matrix[:, model_idx] = fold_predictions
            
            oof_auc = roc_auc_score(y, fold_predictions)
            avg_fold_auc = np.mean(fold_aucs)
            
            model_metrics[model_name] = {
                'oof_auc': oof_auc,
                'avg_fold_auc': avg_fold_auc,
                'fold_aucs': fold_aucs,
                'model_config': model_config
            }
            
            print(f"✅ {model_name} - OOF AUC: {oof_auc:.6f}, Avg Fold AUC: {avg_fold_auc:.6f}")
        
        return oof_matrix, all_model_names, model_metrics
    
    def save_oof_results(self, oof_matrix, model_names, model_metrics, feature_selectors, X_original):
        """Save OOF predictions and metadata with discoverable structure"""
        print("💾 Saving OOF results...")
        
        base_dir = 'data/oof_predictions'
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        oof_df = pd.DataFrame(oof_matrix, columns=model_names)
        predictions_file = f'{base_dir}/oof_predictions_{timestamp}.csv'
        oof_df.to_csv(predictions_file, index=False)
        
        selectors = {
            'var_selector': feature_selectors[0],
            'l1_selector': feature_selectors[1]
        }
        selectors_file = f'{base_dir}/feature_selectors_{timestamp}.pkl'
        joblib.dump(selectors, selectors_file)
        
        metadata = {
            'n_samples': oof_matrix.shape[0],
            'n_models': oof_matrix.shape[1],
            'model_names': model_names,
            'original_features': X_original.shape[1],
            'selected_features': oof_matrix.shape[1] if len(feature_selectors) > 0 else X_original.shape[1],
            'model_metrics': model_metrics,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'predictions_file': predictions_file,
            'selectors_file': selectors_file
        }
        
        metadata_file = f'{base_dir}/oof_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        latest_metadata_file = f'{base_dir}/latest_oof_metadata.json'
        with open(latest_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ OOF results saved:")
        print(f"   Predictions: {predictions_file}")
        print(f"   Selectors: {selectors_file}")
        print(f"   Metadata: {metadata_file}")
        print(f"   Latest: {latest_metadata_file}")
        
        return metadata
    
    def run_oof_generation(self):
        """Complete OOF generation pipeline"""
        print("🚀 Starting OOF generation pipeline...")
        
        # Load data
        X, y, feature_cols, symbols = self.load_data()
        
        X_selected, var_selector, lasso_selector = self.apply_feature_selection(X, y)
        
        model_configs = self.create_model_configs()
        
        oof_matrix, model_names, model_metrics = self.generate_oof_predictions(X_selected, y, model_configs)
        
        # Save results
        metadata = self.save_oof_results(
            oof_matrix, model_names, model_metrics, 
            (var_selector, lasso_selector), X
        )
        
        print("🎉 OOF generation completed successfully!")
        
        print("\n📊 OOF Generation Summary:")
        print(f"   Total samples: {len(y)}")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Total models: {len(model_names)}")
        print(f"   Average OOF AUC: {np.mean([m['oof_auc'] for m in model_metrics.values()]):.6f}")
        
        return metadata

def main():
    """CLI entry point for OOF generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate out-of-fold predictions')
    parser.add_argument('--data-path', default='data/enhanced_features/enhanced_features.csv',
                       help='Path to input features dataset')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', default='data/oof_predictions',
                       help='Output directory for OOF results')
    
    args = parser.parse_args()
    
    generator = OOFGenerator(data_path=args.data_path, n_folds=args.n_folds)
    
    try:
        metadata = generator.run_oof_generation()
        print("✅ OOF generation completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ OOF generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
