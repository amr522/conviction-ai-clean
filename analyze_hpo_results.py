#!/usr/bin/env python3

import os
import json
import pandas as pd
from datetime import datetime

def analyze_hpo_results():
    """Analyze completed HPO results"""
    
    print("🎉 HPO COMPLETION ANALYSIS")
    print("=" * 50)
    
    # Check results directory
    results_dir = "models/hpo_20250629"
    
    if not os.path.exists(results_dir):
        print("❌ HPO results directory not found")
        return
    
    print(f"📁 Results directory: {results_dir}")
    
    # Count files by model type
    model_counts = {}
    total_files = 0
    
    for file in os.listdir(results_dir):
        if file.endswith(('.json', '.pkl', '.joblib')):
            total_files += 1
            
            # Extract model type from filename
            for model in ['extra_trees', 'lightgbm', 'xgboost', 'catboost', 'random_forest']:
                if model in file.lower():
                    model_counts[model] = model_counts.get(model, 0) + 1
                    break
    
    print(f"📊 COMPLETION SUMMARY:")
    print(f"  Total files: {total_files}")
    
    for model, count in model_counts.items():
        completion_pct = (count / 242) * 100
        status = "✅" if count >= 242 else "⚠️"
        print(f"  {status} {model:12} | {count:3d}/242 ({completion_pct:5.1f}%)")
    
    # Calculate total model combinations
    total_expected = 242 * 5  # 242 symbols × 5 models
    total_completed = sum(model_counts.values())
    overall_pct = (total_completed / total_expected) * 100
    
    print(f"\n🎯 OVERALL PROGRESS:")
    print(f"  Completed: {total_completed:,}/{total_expected:,} ({overall_pct:.1f}%)")
    
    if overall_pct >= 100:
        print("  🎉 HPO FULLY COMPLETE!")
    elif overall_pct >= 95:
        print("  ✅ HPO NEARLY COMPLETE!")
    else:
        print("  ⚠️ HPO PARTIALLY COMPLETE")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    
    if overall_pct >= 95:
        print("  1. Generate ensemble predictions:")
        print("     python generate_oof.py --symbols all --base_models top3 --cv 5")
        print()
        print("  2. Train stacking meta-learner:")
        print("     python train_stacking_meta.py --oof_dir oof/ --meta_model logistic")
        print()
        print("  3. Run backtesting:")
        print("     python backtest_pipeline.py --pred_dir predictions/stacked --cost_per_share 0.0005")
        print()
        print("  4. Portfolio optimization:")
        print("     python portfolio_optimizer.py")
        print()
        print("  5. Start live API:")
        print("     python live_prediction_api.py")
    else:
        missing_models = []
        for model in ['extra_trees', 'lightgbm', 'xgboost', 'catboost', 'random_forest']:
            if model_counts.get(model, 0) < 242:
                missing = 242 - model_counts.get(model, 0)
                missing_models.append(f"{model} ({missing} symbols)")
        
        if missing_models:
            print("  Complete remaining HPO:")
            for model_info in missing_models:
                model_name = model_info.split(' ')[0]
                print(f"     python run_hpo.py --symbols remaining --models {model_name} --n_trials 30")

def check_model_performance():
    """Check performance of completed models"""
    
    print(f"\n📈 PERFORMANCE CHECK:")
    
    # Look for performance files
    perf_files = []
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(keyword in file.lower() for keyword in ['performance', 'results', 'scores', 'auc']):
                if file.endswith(('.csv', '.json')):
                    perf_files.append(os.path.join(root, file))
    
    if perf_files:
        print(f"  Found {len(perf_files)} performance files:")
        for file in perf_files[:5]:  # Show first 5
            print(f"    • {file}")
    else:
        print("  No performance files found yet")
        print("  Run model evaluation to generate performance metrics")

def update_readme_status():
    """Update README with completion status"""
    
    print(f"\n📝 UPDATING README STATUS...")
    
    # This would update the README with new completion status
    # For now, just show what should be updated
    
    print("  Status to update:")
    print("  🎯 Status: HPO COMPLETE → ENSEMBLE READY")
    print("  📊 Next phase: Stacking & Portfolio Optimization")
    
    # Update timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Last updated: {current_time}")

if __name__ == "__main__":
    analyze_hpo_results()
    check_model_performance()
    update_readme_status()