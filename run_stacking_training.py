#!/usr/bin/env python3
"""
CLI script for stacking meta-learner training
Usage: python run_stacking_training.py --oof-dir data/oof_predictions
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stacking_meta_learner import StackingMetaLearner

def main():
    parser = argparse.ArgumentParser(description='Train stacking meta-learners')
    parser.add_argument('--oof-dir', default='data/oof_predictions',
                       help='Directory containing OOF predictions')
    parser.add_argument('--output-dir', default='models/stacking_meta',
                       help='Output directory for trained meta-learners')
    
    args = parser.parse_args()
    
    print("🚀 Starting Stacking Meta-Learner Training")
    print(f"📁 OOF directory: {args.oof_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    trainer = StackingMetaLearner(oof_dir=args.oof_dir)
    
    try:
        report = trainer.run_stacking_training()
        print(f"\n🎉 Stacking training completed successfully!")
        print(f"📊 Best ensemble AUC: {report['ensemble_results']['ensemble_auc']:.6f}")
        print(f"📊 Meta-learners trained: {report['training_summary']['n_meta_learners']}")
        print(f"📊 Best individual model: {report['training_summary']['best_model']}")
        print(f"📁 Models saved to: {args.output_dir}")
        return 0
    except Exception as e:
        print(f"❌ Stacking training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
