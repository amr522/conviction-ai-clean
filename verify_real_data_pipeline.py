#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def verify_real_vs_synthetic():
    """Step 1: Confirm real vs synthetic data"""
    print("=== Step 1: Real vs Synthetic Data Verification ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    train_df = pd.read_csv(data_dir / "train.csv", nrows=1000)
    val_df = pd.read_csv(data_dir / "validation.csv", nrows=500)
    
    train_target_dist = train_df.iloc[:, 0].value_counts(normalize=True).sort_index()
    val_target_dist = val_df.iloc[:, 0].value_counts(normalize=True).sort_index()
    
    print(f"Train target distribution: {train_target_dist.to_dict()}")
    print(f"Validation target distribution: {val_target_dist.to_dict()}")
    
    feature_cols = train_df.iloc[:, 1:]
    feature_means = feature_cols.mean().abs().mean()
    feature_stds = feature_cols.std().mean()
    
    print(f"Average absolute feature mean: {feature_means:.4f}")
    print(f"Average feature std: {feature_stds:.4f}")
    
    feature_min = feature_cols.min().min()
    feature_max = feature_cols.max().max()
    print(f"Feature value range: [{feature_min:.2f}, {feature_max:.2f}]")
    
    with open(data_dir / "feature_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Feature count: {len(metadata['feature_columns'])}")
    print(f"Target column: {metadata['target_column']}")
    
    is_real = (
        abs(train_target_dist[0] - 0.522) < 0.05 and  # Close to realistic distribution
        abs(feature_means) < 0.2 and  # Properly standardized
        0.8 < feature_stds < 1.2 and  # Proper standardization
        len(metadata['feature_columns']) >= 60  # Comprehensive feature set
    )
    
    return {
        'is_real': is_real,
        'train_target_dist': train_target_dist.to_dict(),
        'val_target_dist': val_target_dist.to_dict(),
        'feature_count': len(metadata['feature_columns']),
        'standardized': abs(feature_means) < 0.2 and 0.8 < feature_stds < 1.2
    }

def smoke_test_aapl_training():
    """Step 2: Smoke-test model training on AAPL"""
    print("\n=== Step 2: AAPL Model Training Smoke Test ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_val = val_df.iloc[:, 1:].values
    y_val = val_df.iloc[:, 0].values
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_pred_proba)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'features': X_train.shape[1]
    }

def check_timestamp_alignment():
    """Step 3: Verify timestamp alignment (basic checks)"""
    print("\n=== Step 3: Timestamp Alignment Check ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    train_rows = sum(1 for _ in open(data_dir / "train.csv")) - 1
    val_rows = sum(1 for _ in open(data_dir / "validation.csv")) - 1
    test_rows = sum(1 for _ in open(data_dir / "test.csv")) - 1
    
    print(f"Train rows: {train_rows}")
    print(f"Validation rows: {val_rows}")
    print(f"Test rows: {test_rows}")
    
    train_sample = pd.read_csv(data_dir / "train.csv", nrows=1)
    val_sample = pd.read_csv(data_dir / "validation.csv", nrows=1)
    test_sample = pd.read_csv(data_dir / "test.csv", nrows=1)
    
    train_cols = train_sample.shape[1]
    val_cols = val_sample.shape[1]
    test_cols = test_sample.shape[1]
    
    print(f"Column counts - Train: {train_cols}, Val: {val_cols}, Test: {test_cols}")
    
    alignment_ok = train_cols == val_cols == test_cols
    
    print(f"Column alignment: {'✅' if alignment_ok else '❌'}")
    
    return {
        'alignment_ok': alignment_ok,
        'train_rows': train_rows,
        'val_rows': val_rows,
        'test_rows': test_rows,
        'consistent_columns': train_cols == val_cols == test_cols
    }

def check_metadata_alignment():
    """Step 4: Verify metadata alignment"""
    print("\n=== Step 4: Metadata Alignment Check ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    with open(data_dir / "feature_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    scaler = joblib.load(data_dir / "scaler.joblib")
    
    train_sample = pd.read_csv(data_dir / "train.csv", nrows=1)
    
    expected_features = len(metadata['feature_columns'])
    actual_features = train_sample.shape[1] - 1  # Subtract target column
    
    print(f"Expected features (metadata): {expected_features}")
    print(f"Actual features (data): {actual_features}")
    
    scaler_features = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else len(scaler.scale_)
    print(f"Scaler features: {scaler_features}")
    
    metadata_match = expected_features == actual_features == scaler_features
    
    print(f"Metadata alignment: {'✅' if metadata_match else '❌'}")
    
    return {
        'metadata_match': metadata_match,
        'expected_features': expected_features,
        'actual_features': actual_features,
        'scaler_features': scaler_features
    }

def generate_integration_recommendations(results):
    """Step 5: Generate integration recommendations"""
    print("\n=== Step 5: Integration Recommendations ===")
    
    val_auc = results['smoke_test']['val_auc']
    
    if val_auc >= 0.55:
        print("✅ AUC threshold met - recommend integration")
        print("\nRecommended changes:")
        print("1. Update config.yaml to point to real data directory")
        print("2. Modify training scripts to use real data by default")
        print("3. Add --use-synthetic flag for testing purposes")
        print("4. Update data loading paths in run_base_models.py")
        
        integration_plan = {
            'recommend_integration': True,
            'config_changes': [
                'Update data_dir in config.yaml',
                'Set use_real_data: true by default',
                'Add synthetic_data_dir for fallback'
            ],
            'script_changes': [
                'Modify run_base_models.py data loading',
                'Update prepare_sagemaker_data.py paths',
                'Add command line flag for synthetic data'
            ]
        }
    else:
        print("❌ AUC threshold not met - investigate further")
        integration_plan = {
            'recommend_integration': False,
            'reason': f'Validation AUC {val_auc:.4f} < 0.55 threshold'
        }
    
    return integration_plan

def main():
    """Run complete verification pipeline"""
    print("Real Data Pipeline Verification")
    print("=" * 50)
    
    results = {}
    
    try:
        results['real_vs_synthetic'] = verify_real_vs_synthetic()
    except Exception as e:
        print(f"Error in Step 1: {e}")
        results['real_vs_synthetic'] = {'is_real': False, 'error': str(e)}
    
    try:
        results['smoke_test'] = smoke_test_aapl_training()
    except Exception as e:
        print(f"Error in Step 2: {e}")
        results['smoke_test'] = {'train_auc': 0.0, 'val_auc': 0.0, 'error': str(e)}
    
    try:
        results['timestamp_alignment'] = check_timestamp_alignment()
    except Exception as e:
        print(f"Error in Step 3: {e}")
        results['timestamp_alignment'] = {'alignment_ok': False, 'error': str(e)}
    
    try:
        results['metadata_alignment'] = check_metadata_alignment()
    except Exception as e:
        print(f"Error in Step 4: {e}")
        results['metadata_alignment'] = {'metadata_match': False, 'error': str(e)}
    
    try:
        results['integration'] = generate_integration_recommendations(results)
    except Exception as e:
        print(f"Error in Step 5: {e}")
        results['integration'] = {'recommend_integration': False, 'error': str(e)}
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY TABLE")
    print("=" * 50)
    
    print("| Check | Result | Notes |")
    print("|-------|--------|-------|")
    
    real_result = "✅" if results['real_vs_synthetic'].get('is_real', False) else "❌"
    real_notes = f"Target dist: {results['real_vs_synthetic'].get('train_target_dist', {}).get(0, 'N/A'):.3f}/0.522 expected"
    print(f"| Real vs. Synthetic | {real_result} | {real_notes} |")
    
    train_auc = results['smoke_test'].get('train_auc', 0.0)
    val_auc = results['smoke_test'].get('val_auc', 0.0)
    print(f"| AAPL AUC | {val_auc:.4f} | train: {train_auc:.4f}, val: {val_auc:.4f} |")
    
    align_result = "✅" if results['timestamp_alignment'].get('alignment_ok', False) else "❌"
    align_notes = f"Consistent columns across splits"
    print(f"| Timestamp Alignment | {align_result} | {align_notes} |")
    
    meta_result = "✅" if results['metadata_alignment'].get('metadata_match', False) else "❌"
    meta_notes = f"Features: {results['metadata_alignment'].get('actual_features', 'N/A')}"
    print(f"| Metadata Match | {meta_result} | {meta_notes} |")
    
    print("\n" + "=" * 50)
    
    if results['integration'].get('recommend_integration', False):
        print("🎯 RECOMMENDATION: Integrate real data pipeline")
        print(f"   Validation AUC {val_auc:.4f} meets ≥0.55 threshold")
    else:
        print("⚠️  RECOMMENDATION: Further investigation needed")
        print(f"   Validation AUC {val_auc:.4f} below 0.55 threshold")
    
    with open('verification_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to verification_results.json")
    
    return results

if __name__ == "__main__":
    main()
