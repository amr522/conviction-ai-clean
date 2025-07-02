#!/usr/bin/env python3
"""
Simplified ensemble training using linear blending of best HPO models
"""
import os
import numpy as np
import pandas as pd
import joblib
import tarfile
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

class SimplifiedEnsemble:
    def __init__(self, models_dir, validation_file):
        self.models_dir = models_dir
        self.validation_file = validation_file
        self.models = []
        self.ensemble_model = None
        
    def load_validation_data(self):
        """Load validation data for ensemble training"""
        print("📊 Loading validation data...")
        
        df = pd.read_csv(self.validation_file, header=None)
        
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        
        print(f"✅ Loaded validation data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def extract_and_load_models(self):
        """Extract and load XGBoost models from tar.gz files"""
        print("🔍 Discovering and loading models...")
        
        model_predictions = []
        model_names = []
        
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file == 'model.tar.gz':
                    model_path = os.path.join(root, file)
                    trial_name = os.path.basename(root.replace('/output', ''))
                    
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            with tarfile.open(model_path, 'r:gz') as tar:
                                tar.extractall(temp_dir)
                            
                            xgb_model_path = None
                            for extracted_file in os.listdir(temp_dir):
                                if 'xgboost' in extracted_file.lower() or extracted_file.endswith('.model'):
                                    xgb_model_path = os.path.join(temp_dir, extracted_file)
                                    break
                            
                            if xgb_model_path and os.path.exists(xgb_model_path):
                                model = xgb.Booster()
                                model.load_model(xgb_model_path)
                                
                                model_predictions.append(model)
                                model_names.append(trial_name)
                                print(f"✅ Loaded model: {trial_name}")
                            else:
                                print(f"⚠️ No XGBoost model found in {trial_name}")
                                
                    except Exception as e:
                        print(f"❌ Failed to load model {trial_name}: {e}")
        
        print(f"📊 Successfully loaded {len(model_predictions)} models")
        return model_predictions, model_names
    
    def generate_predictions(self, models, X):
        """Generate predictions from all models"""
        print("🔮 Generating predictions from all models...")
        
        predictions = []
        
        for i, model in enumerate(models):
            try:
                dmatrix = xgb.DMatrix(X)
                pred = model.predict(dmatrix)
                predictions.append(pred)
                
            except Exception as e:
                print(f"❌ Failed to predict with model {i}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions generated")
        
        predictions_array = np.column_stack(predictions)
        print(f"✅ Generated predictions: {predictions_array.shape}")
        
        return predictions_array
    
    def train_ensemble(self):
        """Train simplified ensemble using linear blending"""
        print("🚀 Training simplified ensemble...")
        
        X_val, y_val = self.load_validation_data()
        
        models, model_names = self.extract_and_load_models()
        
        if len(models) == 0:
            raise ValueError("No models loaded for ensemble training")
        
        predictions = self.generate_predictions(models, X_val)
        
        print("🔗 Training linear blending model...")
        self.ensemble_model = LogisticRegression(random_state=42)
        self.ensemble_model.fit(predictions, y_val)
        
        ensemble_pred = self.ensemble_model.predict_proba(predictions)[:, 1]
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        
        individual_aucs = []
        for i in range(predictions.shape[1]):
            try:
                auc = roc_auc_score(y_val, predictions[:, i])
                individual_aucs.append(auc)
            except:
                individual_aucs.append(0.0)
        
        best_individual_auc = max(individual_aucs)
        avg_individual_auc = np.mean(individual_aucs)
        
        print(f"📊 Ensemble Results:")
        print(f"   Ensemble AUC: {ensemble_auc:.6f}")
        print(f"   Best Individual AUC: {best_individual_auc:.6f}")
        print(f"   Average Individual AUC: {avg_individual_auc:.6f}")
        print(f"   Improvement: {ensemble_auc - best_individual_auc:+.6f}")
        
        self.models = models
        self.model_names = model_names
        
        return {
            'ensemble_auc': ensemble_auc,
            'best_individual_auc': best_individual_auc,
            'avg_individual_auc': avg_individual_auc,
            'num_models': len(models)
        }
    
    def save_ensemble(self, output_path):
        """Save ensemble model"""
        print(f"💾 Saving ensemble model to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        ensemble_data = {
            'ensemble_model': self.ensemble_model,
            'model_names': self.model_names,
            'num_base_models': len(self.models)
        }
        
        joblib.dump(ensemble_data, output_path)
        print(f"✅ Ensemble saved to {output_path}")

def main():
    models_dir = 'models/hpo_best/46_models_hpo'
    validation_file = 'data/sagemaker/validation.csv'
    output_file = 'models/regression_ensemble/ensemble_blend.pkl'
    
    print("🎯 Starting simplified ensemble training...")
    
    ensemble = SimplifiedEnsemble(models_dir, validation_file)
    
    try:
        results = ensemble.train_ensemble()
        ensemble.save_ensemble(output_file)
        
        print("🎉 Ensemble training completed successfully!")
        print(f"📊 Final Results: {results}")
        
    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
