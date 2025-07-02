#!/usr/bin/env python3

import pickle
from pathlib import Path
from datetime import datetime

def compare_hpo_progress():
    """Compare current HPO progress to previous check"""
    
    # Previous status (from earlier check)
    previous = {
        'completed': 13,
        'total': 285,
        'progress': 4.6,
        'best_symbol': 'MP',
        'best_auc': 0.5831
    }
    
    # Current status
    hpo_dir = Path("models/debug_hpo")
    current_completed = 0
    current_best_auc = 0
    current_best_symbol = ""
    
    if hpo_dir.exists():
        pkl_files = list(hpo_dir.glob("*.pkl"))
        current_completed = len(pkl_files)
        
        # Find current best
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                auc = data.get('best_auc', 0)
                if auc > current_best_auc:
                    current_best_auc = auc
                    current_best_symbol = data.get('symbol', pkl_file.stem.split('_')[0])
            except:
                pass
    
    # Calculate changes
    progress_change = (current_completed / 285 * 100) - previous['progress']
    new_completions = current_completed - previous['completed']
    
    print("📊 HPO PROGRESS COMPARISON")
    print("=" * 30)
    print(f"Previous: {previous['completed']}/285 ({previous['progress']:.1f}%)")
    print(f"Current:  {current_completed}/285 ({current_completed/285*100:.1f}%)")
    print(f"Change:   +{new_completions} symbols ({progress_change:+.1f}%)")
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"Previous: {previous['best_symbol']} ({previous['best_auc']:.4f})")
    print(f"Current:  {current_best_symbol} ({current_best_auc:.4f})")
    
    if new_completions > 0:
        print(f"\n✅ Progress made: {new_completions} new completions")
    elif new_completions == 0:
        print(f"\n⏸️ No new progress since last check")
    
    return {
        'previous_completed': previous['completed'],
        'current_completed': current_completed,
        'new_completions': new_completions,
        'progress_change': progress_change
    }

if __name__ == "__main__":
    compare_hpo_progress()