#!/usr/bin/env python3

import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_hpo_monitoring():
    """Run HPO monitoring and generate reports"""
    
    print(f"🔄 HPO Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run dashboard
        result = subprocess.run(['python', 'hpo_dashboard.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dashboard generated successfully")
            
            # Check for new plots
            plots = list(Path("outputs").glob("hpo_*.png"))
            print(f"📊 Generated {len(plots)} monitoring plots")
            
        else:
            print(f"❌ Dashboard failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def continuous_monitoring(interval_minutes=30):
    """Run continuous HPO monitoring"""
    
    print(f"🚀 Starting continuous HPO monitoring (every {interval_minutes} min)")
    
    while True:
        run_hpo_monitoring()
        print(f"⏰ Next check in {interval_minutes} minutes...\n")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HPO Monitoring Automation')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in minutes')
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitoring(args.interval)
    else:
        run_hpo_monitoring()