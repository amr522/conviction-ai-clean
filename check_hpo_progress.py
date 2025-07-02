#!/usr/bin/env python3

import os
import json
import pandas as pd
from datetime import datetime

def check_hpo_progress():
    """Check hyperparameter optimization progress"""
    
    print("🔍 HPO PROGRESS CHECK")
    print("=" * 50)
    
    # Check for HPO results directory
    hpo_dirs = ["hpo_results", "optuna_studies", "hyperopt_results"]
    hpo_dir = None
    
    for d in hpo_dirs:
        if os.path.exists(d):
            hpo_dir = d
            break
    
    if not hpo_dir:
        print("❌ No HPO results directory found")
        print("💡 Run: python run_hpo.py --symbols all --models lightgbm,xgboost,extra_trees --n_trials 30")
        return
    
    print(f"📁 HPO Directory: {hpo_dir}")
    
    # Check study files
    study_files = [f for f in os.listdir(hpo_dir) if f.endswith(('.json', '.db', '.pkl'))]
    
    if not study_files:
        print("❌ No study files found")
        return
    
    print(f"📊 Found {len(study_files)} study files")
    
    # Analyze progress by model type
    progress = {}
    
    for file in study_files:
        try:
            if file.endswith('.json'):
                with open(f"{hpo_dir}/{file}", 'r') as f:
                    data = json.load(f)
                
                # Extract model and symbol from filename
                parts = file.replace('.json', '').split('_')
                if len(parts) >= 2:
                    model = parts[0]
                    symbol = parts[1] if len(parts) > 1 else 'unknown'
                    
                    if model not in progress:
                        progress[model] = {'completed': 0, 'trials': 0, 'best_score': None}
                    
                    progress[model]['completed'] += 1
                    
                    # Get trial count and best score
                    if 'trials' in data:
                        progress[model]['trials'] += len(data['trials'])
                        
                        # Find best score
                        scores = [t.get('value', 0) for t in data['trials'] if 'value' in t]
                        if scores:
                            best = max(scores)
                            if progress[model]['best_score'] is None or best > progress[model]['best_score']:
                                progress[model]['best_score'] = best
        except:
            continue
    
    # Display progress
    if progress:
        print("\n📈 HPO PROGRESS BY MODEL:")
        print("-" * 60)
        
        for model, stats in progress.items():
            best_score = f"{stats['best_score']:.4f}" if stats['best_score'] else "N/A"
            print(f"  {model:15} | {stats['completed']:3d} symbols | {stats['trials']:4d} trials | Best: {best_score}")
        
        # Overall stats
        total_symbols = sum(s['completed'] for s in progress.values())
        total_trials = sum(s['trials'] for s in progress.values())
        
        print("-" * 60)
        print(f"  {'TOTAL':15} | {total_symbols:3d} symbols | {total_trials:4d} trials")
        
        # Progress percentage (assuming 242 symbols target)
        target_symbols = 242
        completion_pct = (total_symbols / (target_symbols * len(progress))) * 100 if progress else 0
        
        print(f"\n🎯 Overall Progress: {completion_pct:.1f}% complete")
        
        if completion_pct < 100:
            remaining = target_symbols * len(progress) - total_symbols
            print(f"⏳ Remaining: {remaining} symbol-model combinations")
    
    # Check for running processes
    check_running_hpo()
    
    # Show recent activity
    show_recent_activity(hpo_dir)

def check_running_hpo():
    """Check if HPO is currently running"""
    
    import subprocess
    
    try:
        # Check for running Python processes with HPO keywords
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        hpo_processes = []
        for line in result.stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['hpo', 'optuna', 'hyperopt', 'run_hpo']):
                if 'python' in line.lower():
                    hpo_processes.append(line.strip())
        
        if hpo_processes:
            print(f"\n🔄 RUNNING HPO PROCESSES ({len(hpo_processes)}):")
            for proc in hpo_processes:
                # Extract relevant parts
                parts = proc.split()
                if len(parts) > 10:
                    print(f"  PID: {parts[1]} | {' '.join(parts[10:])}")
        else:
            print("\n💤 No HPO processes currently running")
            
    except:
        print("\n⚠️ Could not check running processes")

def show_recent_activity(hpo_dir):
    """Show recent HPO activity"""
    
    print(f"\n📅 RECENT ACTIVITY:")
    
    # Get file modification times
    files_with_times = []
    
    for file in os.listdir(hpo_dir):
        file_path = f"{hpo_dir}/{file}"
        if os.path.isfile(file_path):
            mtime = os.path.getmtime(file_path)
            files_with_times.append((file, mtime))
    
    # Sort by modification time (most recent first)
    files_with_times.sort(key=lambda x: x[1], reverse=True)
    
    # Show last 5 files
    for file, mtime in files_with_times[:5]:
        mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {mod_time} | {file}")
    
    if not files_with_times:
        print("  No recent activity found")

def show_hpo_commands():
    """Show available HPO commands"""
    
    print(f"\n🛠️ HPO COMMANDS:")
    print("# Start HPO for specific models")
    print("python run_hpo.py --symbols all --models lightgbm,xgboost,extra_trees --n_trials 30")
    print()
    print("# Resume HPO (if supported)")
    print("python run_hpo.py --resume --n_trials 50")
    print()
    print("# HPO for specific symbols")
    print("python run_hpo.py --symbols AAPL,MSFT,GOOGL --models lightgbm --n_trials 100")
    print()
    print("# Check this progress")
    print("python check_hpo_progress.py")

if __name__ == "__main__":
    check_hpo_progress()
    show_hpo_commands()