#!/usr/bin/env python3
"""
CLI script for OOF generation
Usage: python run_oof_generation.py --data-path data/enhanced_features/enhanced_features.csv --n-folds 5
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from oof_generation import OOFGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate out-of-fold predictions')
    parser.add_argument('--data-path', default='data/enhanced_features/enhanced_features.csv',
                       help='Path to input features dataset')
    parser.add_argument('--symbols-file', help='Path to symbols list file (optional)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', default='data/oof_predictions',
                       help='Output directory for OOF results')
    
    args = parser.parse_args()
    
    print("🚀 Starting OOF Generation Pipeline")
    print(f"📊 Data path: {args.data_path}")
    print(f"🔄 CV folds: {args.n_folds}")
    print(f"📁 Output dir: {args.output_dir}")
    
    generator = OOFGenerator(
        data_path=args.data_path,
        n_folds=args.n_folds
    )
    
    try:
        metadata = generator.run_oof_generation()
        print(f"\n🎉 OOF generation completed successfully!")
        print(f"📊 Generated predictions for {metadata['n_models']} models")
        print(f"📊 Total samples: {metadata['n_samples']}")
        print(f"📊 Selected features: {metadata['selected_features']}")
        print(f"📁 Results saved to: {args.output_dir}")
        return 0
    except Exception as e:
        print(f"❌ OOF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
