#!/usr/bin/env python3
"""
Master script to run all accuracy boost training steps
"""
import os
import sys
import json
from datetime import datetime

def run_training_step(script_name, step_name):
    """Run a training step and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {step_name}")
    print(f"{'='*60}")
    
    try:
        if script_name == 'train_calibrated_base_models':
            from train_calibrated_base_models import main
        elif script_name == 'train_xgboost_models':
            from train_xgboost_models import main
        elif script_name == 'train_catboost_models':
            from train_catboost_models import main
        elif script_name == 'train_tabnet_models':
            from train_tabnet_models import main
        else:
            raise ValueError(f"Unknown script: {script_name}")
        
        result = main()
        print(f"✅ {step_name} completed successfully")
        return True, result
        
    except Exception as e:
        print(f"❌ {step_name} failed: {e}")
        return False, str(e)

def main():
    """Run complete accuracy boost training pipeline"""
    print("🎯 Starting Accuracy Boost Training Pipeline")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    training_steps = [
        ('train_calibrated_base_models', 'Step 1 - Calibrated Base Models'),
        ('train_xgboost_models', 'Step 2 - XGBoost Models'),
        ('train_catboost_models', 'Step 3 - CatBoost Models'),
        ('train_tabnet_models', 'Step 6 - TabNet Integration')
    ]
    
    results = {}
    failed_steps = []
    
    for script_name, step_name in training_steps:
        success, result = run_training_step(script_name, step_name)
        results[step_name] = {
            'success': success,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        if not success:
            failed_steps.append(step_name)
            print(f"⚠️ Continuing despite {step_name} failure...")
    
    print(f"\n{'='*60}")
    print("📊 ACCURACY BOOST TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful_steps = len([r for r in results.values() if r['success']])
    total_steps = len(training_steps)
    
    print(f"✅ Successful steps: {successful_steps}/{total_steps}")
    
    if failed_steps:
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
    
    final_report = {
        'pipeline_summary': {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'completion_timestamp': datetime.now().isoformat()
        },
        'step_results': results
    }
    
    with open('accuracy_boost_training_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"📋 Full report saved to: accuracy_boost_training_report.json")
    
    if successful_steps == total_steps:
        print("🎉 All training steps completed successfully!")
        return 0
    else:
        print(f"⚠️ {len(failed_steps)} steps failed - check individual reports")
        return 1

if __name__ == "__main__":
    exit(main())
