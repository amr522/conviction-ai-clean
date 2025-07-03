#!/usr/bin/env python3
"""
Stacking Meta-Learner Training
Trains second-level models on OOF predictions
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("⚠️ XGBoost not available for meta-learning")
    xgb = None

class StackingMetaLearner:
    def __init__(self, oof_dir='data/oof_predictions'):
        self.oof_dir = oof_dir
        self.random_state = 42
        
    def discover_latest_oof_files(self):
        """Dynamically discover the latest OOF files"""
        print("🔍 Discovering OOF files...")
        
        latest_metadata_path = os.path.join(self.oof_dir, 'latest_oof_metadata.json')
        if os.path.exists(latest_metadata_path):
            with open(latest_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            predictions_file = metadata.get('predictions_file')
            if predictions_file and os.path.exists(predictions_file):
                print(f"✅ Found latest OOF files via metadata")
                return predictions_file, latest_metadata_path
        
        print("🔍 Scanning for timestamped OOF files...")
        oof_files = []
        metadata_files = []
        
        for file in os.listdir(self.oof_dir):
            if file.startswith('oof_predictions_') and file.endswith('.csv'):
                oof_files.append(os.path.join(self.oof_dir, file))
            elif file.startswith('oof_metadata_') and file.endswith('.json'):
                metadata_files.append(os.path.join(self.oof_dir, file))
        
        if not oof_files:
            oof_files = [f for f in os.listdir(self.oof_dir) if f.endswith('predictions.csv')]
            metadata_files = [f for f in os.listdir(self.oof_dir) if f.endswith('metadata.json')]
            
            if oof_files:
                oof_files = [os.path.join(self.oof_dir, f) for f in oof_files]
                metadata_files = [os.path.join(self.oof_dir, f) for f in metadata_files]
        
        if not oof_files:
            raise FileNotFoundError(f"No OOF prediction files found in {self.oof_dir}")
        
        latest_oof = max(oof_files, key=os.path.getmtime)
        latest_metadata = max(metadata_files, key=os.path.getmtime) if metadata_files else None
        
        print(f"✅ Discovered latest OOF files: {latest_oof}")
        return latest_oof, latest_metadata

    def load_oof_data(self):
        """Load OOF predictions and metadata"""
        print("📊 Loading OOF predictions...")
        
        oof_path, metadata_path = self.discover_latest_oof_files()
        
        oof_df = pd.read_csv(oof_path)
        X_meta = oof_df.values
        
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        enhanced_path = 'data/enhanced_features/enhanced_features.csv'
        if os.path.exists(enhanced_path):
            df = pd.read_csv(enhanced_path)
            y = df['target_next_day'].values
        else:
            raise FileNotFoundError("Cannot find original target values")
        
        print(f"✅ Loaded OOF data: {X_meta.shape[0]} samples, {X_meta.shape[1]} meta-features")
        
        return X_meta, y, metadata
    
    def create_meta_learner_configs(self):
        """Create meta-learner configurations"""
        configs = {
            'logistic_regression': {
                'name': 'LogisticRegression',
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='liblinear'
                ),
                'params': {}
            },
            'random_forest': {
                'name': 'RandomForest',
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {}
            }
        }
        
        if xgb is not None:
            configs['xgboost_meta'] = {
                'name': 'XGBoost_Meta',
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=self.random_state,
                    verbosity=0
                ),
                'params': {}
            }
        
        return configs
    
    def evaluate_meta_learner(self, model, X_meta, y, cv_folds=5):
        """Evaluate meta-learner with cross-validation"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(model, X_meta, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
    
    def train_meta_learners(self, X_meta, y):
        """Train all meta-learner configurations"""
        print("🧠 Training meta-learners...")
        
        configs = self.create_meta_learner_configs()
        results = {}
        trained_models = {}
        
        for config_name, config in configs.items():
            print(f"📈 Training {config['name']}...")
            
            model = config['model']
            
            cv_results = self.evaluate_meta_learner(model, X_meta, y)
            
            model.fit(X_meta, y)
            
            predictions = model.predict_proba(X_meta)[:, 1]
            train_auc = roc_auc_score(y, predictions)
            
            results[config_name] = {
                'model_name': config['name'],
                'cv_auc_mean': cv_results['cv_auc_mean'],
                'cv_auc_std': cv_results['cv_auc_std'],
                'train_auc': train_auc,
                'cv_scores': cv_results['cv_scores']
            }
            
            trained_models[config_name] = model
            
            print(f"✅ {config['name']} - CV AUC: {cv_results['cv_auc_mean']:.6f}±{cv_results['cv_auc_std']:.6f}")
        
        return trained_models, results
    
    def create_ensemble_blend(self, trained_models, X_meta, y):
        """Create weighted ensemble of meta-learners"""
        print("🎯 Creating ensemble blend...")
        
        meta_predictions = {}
        for name, model in trained_models.items():
            pred = model.predict_proba(X_meta)[:, 1]
            meta_predictions[name] = pred
        
        ensemble_pred = np.mean(list(meta_predictions.values()), axis=0)
        ensemble_auc = roc_auc_score(y, ensemble_pred)
        
        individual_aucs = {}
        for name, pred in meta_predictions.items():
            individual_aucs[name] = roc_auc_score(y, pred)
        
        best_model_name = max(individual_aucs, key=individual_aucs.get)
        best_model = trained_models[best_model_name]
        
        print(f"✅ Ensemble AUC: {ensemble_auc:.6f}")
        print(f"✅ Best individual model: {best_model_name} (AUC: {individual_aucs[best_model_name]:.6f})")
        
        return {
            'ensemble_auc': ensemble_auc,
            'individual_aucs': individual_aucs,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'ensemble_predictions': ensemble_pred,
            'meta_predictions': meta_predictions
        }
    
    def save_meta_learner_results(self, trained_models, results, ensemble_results):
        """Save trained meta-learners and results"""
        print("💾 Saving meta-learner results...")
        
        os.makedirs('models/stacking_meta', exist_ok=True)
        
        best_model_path = 'models/stacking_meta/best_meta_learner.pkl'
        joblib.dump(ensemble_results['best_model'], best_model_path)
        
        for name, model in trained_models.items():
            model_path = f'models/stacking_meta/{name}_meta_learner.pkl'
            joblib.dump(model, model_path)
        
        ensemble_data = {
            'best_model': ensemble_results['best_model'],
            'ensemble_weights': {name: 1.0/len(trained_models) for name in trained_models.keys()},
            'meta_learners': trained_models,
            'ensemble_auc': ensemble_results['ensemble_auc'],
            'individual_aucs': ensemble_results['individual_aucs']
        }
        
        ensemble_path = 'models/stacking_meta/ensemble_blend.pkl'
        joblib.dump(ensemble_data, ensemble_path)
        
        report = {
            'training_summary': {
                'n_meta_learners': len(trained_models),
                'best_model': ensemble_results['best_model_name'],
                'best_auc': ensemble_results['individual_aucs'][ensemble_results['best_model_name']],
                'ensemble_auc': ensemble_results['ensemble_auc'],
                'improvement': ensemble_results['ensemble_auc'] - max(ensemble_results['individual_aucs'].values())
            },
            'individual_results': results,
            'ensemble_results': {
                'ensemble_auc': ensemble_results['ensemble_auc'],
                'individual_aucs': ensemble_results['individual_aucs'],
                'best_model_name': ensemble_results['best_model_name']
            }
        }
        
        report_path = 'models/stacking_meta/stacking_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Meta-learner results saved:")
        print(f"   Best model: {best_model_path}")
        print(f"   Ensemble: {ensemble_path}")
        print(f"   Report: {report_path}")
        
        return report
    
    def run_stacking_training(self):
        """Complete stacking meta-learner training pipeline"""
        print("🚀 Starting stacking meta-learner training...")
        
        X_meta, y, oof_metadata = self.load_oof_data()
        
        trained_models, results = self.train_meta_learners(X_meta, y)
        
        ensemble_results = self.create_ensemble_blend(trained_models, X_meta, y)
        
        report = self.save_meta_learner_results(trained_models, results, ensemble_results)
        
        print("🎉 Stacking meta-learner training completed successfully!")
        
        print("\n📊 Stacking Training Summary:")
        print(f"   Meta-features: {X_meta.shape[1]}")
        print(f"   Meta-learners trained: {len(trained_models)}")
        print(f"   Best individual AUC: {max(ensemble_results['individual_aucs'].values()):.6f}")
        print(f"   Ensemble AUC: {ensemble_results['ensemble_auc']:.6f}")
        
        return report

def main():
    """CLI entry point for stacking training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stacking meta-learners')
    parser.add_argument('--oof-dir', default='data/oof_predictions',
                       help='Directory containing OOF predictions')
    parser.add_argument('--output-dir', default='models/stacking_meta',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    trainer = StackingMetaLearner(oof_dir=args.oof_dir)
    
    try:
        report = trainer.run_stacking_training()
        print("✅ Stacking training completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Stacking training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
