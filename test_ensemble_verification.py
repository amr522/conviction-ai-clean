#!/usr/bin/env python3
"""
Test script to verify ensemble model creation
"""
import joblib
import os

def test_ensemble_model():
    ensemble_path = 'models/regression_ensemble/ensemble_blend.pkl'
    
    if os.path.exists(ensemble_path):
        try:
            ensemble = joblib.load(ensemble_path)
            print(f'✅ Ensemble loaded: {ensemble["num_base_models"]} base models')
            return True
        except Exception as e:
            print(f'❌ Failed to load ensemble: {e}')
            return False
    else:
        print('❌ Ensemble model not found')
        return False

if __name__ == "__main__":
    test_ensemble_model()
