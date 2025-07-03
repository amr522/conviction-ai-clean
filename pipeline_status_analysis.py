#!/usr/bin/env python3
"""
Comprehensive pipeline status analysis for accuracy boost cycle
"""
import json
import glob
import os
import statistics
from datetime import datetime

def analyze_base_models():
    """Analyze base model performance"""
    print("=== BASE MODEL ANALYSIS ===")
    
    metrics_files = glob.glob('data/base_model_outputs/46_models/*_metrics.json')
    if not metrics_files:
        print("❌ No base model metrics found")
        return
    
    aucs = []
    symbols = []
    
    for file in metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            aucs.append(data['auc_score'])
            symbols.append(data['symbol'])
    
    print(f"✅ Total models trained: {len(aucs)}")
    print(f"📊 Average AUC: {statistics.mean(aucs):.4f}")
    print(f"📊 Median AUC: {statistics.median(aucs):.4f}")
    print(f"📊 Min AUC: {min(aucs):.4f} ({symbols[aucs.index(min(aucs))]})")
    print(f"📊 Max AUC: {max(aucs):.4f} ({symbols[aucs.index(max(aucs))]})")
    print(f"🎯 Models above 0.55: {sum(1 for auc in aucs if auc > 0.55)}")
    print(f"🎯 Models above 0.595 (target): {sum(1 for auc in aucs if auc > 0.595)}")
    print()
    
    return aucs

def check_pipeline_steps():
    """Check completion status of 7-step accuracy boost pipeline"""
    print("=== ACCURACY BOOST PIPELINE STATUS ===")
    
    steps = [
        ("Step 1 - Calibrated Base Models", "models/calibrated_base"),
        ("Step 2 - XGBoost Models", "models/xgboost"), 
        ("Step 3 - CatBoost Models", "models/catboost"),
        ("Step 4 - OOF Generation", "data/oof_predictions/oof_meta_features.csv"),
        ("Step 5 - Stacking Meta-Learners", "models/stacking_meta_enhanced"),
        ("Step 6 - TabNet Integration", "models/tabnet"),
        ("Step 7 - Holdout Validation", "data/holdout_test")
    ]
    
    completed_steps = 0
    
    for step_name, path in steps:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                status = f"✅ ({len(files)} files)" if files else "⚠️ (empty)"
            else:
                size = os.path.getsize(path)
                status = f"✅ ({size:,} bytes)"
            completed_steps += 1
        else:
            status = "❌ (missing)"
        
        print(f"{step_name}: {status}")
    
    print(f"\n📈 Pipeline Progress: {completed_steps}/7 steps completed")
    print()
    
    return completed_steps

def check_infrastructure():
    """Check supporting infrastructure status"""
    print("=== INFRASTRUCTURE STATUS ===")
    
    infrastructure = [
        ("Enhanced Features", "data/enhanced_features/enhanced_features.csv"),
        ("OOF Predictions", "data/oof_predictions/oof_meta_features.csv"),
        ("Artifact Management", "enhanced_artifact_inventory.py"),
        ("S3 Sync Utility", "s3_artifact_sync.py"),
        ("OOF Generation Script", "oof_generation.py"),
        ("Stacking Meta-Learner", "stacking_meta_learner.py")
    ]
    
    for name, path in infrastructure:
        if os.path.exists(path):
            if path.endswith('.csv'):
                size = os.path.getsize(path)
                status = f"✅ ({size:,} bytes)"
            else:
                status = "✅"
        else:
            status = "❌"
        
        print(f"{name}: {status}")
    
    print()

def analyze_performance_issues():
    """Analyze why models are performing poorly"""
    print("=== PERFORMANCE ANALYSIS ===")
    
    enhanced_features_path = "data/enhanced_features/enhanced_features.csv"
    if os.path.exists(enhanced_features_path):
        size = os.path.getsize(enhanced_features_path)
        print(f"✅ Enhanced features dataset: {size:,} bytes")
    else:
        print("❌ Enhanced features dataset missing")
    
    oof_path = "data/oof_predictions/oof_meta_features.csv"
    if os.path.exists(oof_path):
        with open(oof_path, 'r') as f:
            line_count = sum(1 for line in f)
        print(f"✅ OOF predictions: {line_count:,} samples")
    else:
        print("❌ OOF predictions missing")
    
    stacking_path = "models/stacking_meta_enhanced"
    if os.path.exists(stacking_path):
        files = os.listdir(stacking_path)
        total_size = sum(os.path.getsize(os.path.join(stacking_path, f)) for f in files)
        print(f"✅ Stacking models: {len(files)} files, {total_size:,} bytes total")
    else:
        print("❌ Stacking models missing")
    
    print()

def main():
    print(f"PIPELINE STATUS ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print()
    
    aucs = analyze_base_models()
    
    completed_steps = check_pipeline_steps()
    
    check_infrastructure()
    
    analyze_performance_issues()
    
    print("=== SUMMARY & RECOMMENDATIONS ===")
    
    if aucs:
        avg_auc = statistics.mean(aucs)
        if avg_auc < 0.52:
            print("🚨 CRITICAL: Models performing at random chance level")
            print("   Likely issues: data quality, feature engineering, target construction")
        elif avg_auc < 0.595:
            print("⚠️ WARNING: Models below target performance (0.595+ AUC)")
            print("   Need: ensemble optimization, feature selection, hyperparameter tuning")
        else:
            print("✅ SUCCESS: Models meeting target performance")
    
    if completed_steps < 7:
        print(f"📋 TODO: Complete remaining {7 - completed_steps} pipeline steps")
        
        missing_steps = []
        if not os.path.exists("models/calibrated_base"):
            missing_steps.append("Calibrated base models")
        if not os.path.exists("models/xgboost"):
            missing_steps.append("XGBoost models")
        if not os.path.exists("models/catboost"):
            missing_steps.append("CatBoost models")
        if not os.path.exists("models/tabnet"):
            missing_steps.append("TabNet integration")
        if not os.path.exists("data/holdout_test"):
            missing_steps.append("Holdout validation")
            
        if missing_steps:
            print(f"   Missing: {', '.join(missing_steps)}")
    
    print()
    print("=== NEXT ACTIONS ===")
    print("1. Investigate data quality and feature engineering")
    print("2. Complete missing pipeline steps")
    print("3. Implement advanced ensemble methods")
    print("4. Run comprehensive holdout validation")
    print("5. Optimize for 0.595+ AUC target")

if __name__ == "__main__":
    main()
