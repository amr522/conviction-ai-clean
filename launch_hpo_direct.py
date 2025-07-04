#!/usr/bin/env python3
"""
Direct HPO launch script - bypasses GitHub Actions
"""
import os
import sys

os.environ['PINNED_DATA_S3'] = 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv'

from aws_hpo_launch import launch_aapl_hpo

def main():
    print("🚀 Launching AAPL HPO job directly...")
    print(f"Using pinned dataset: {os.environ.get('PINNED_DATA_S3')}")
    
    try:
        print("🧪 Testing with dry-run mode first...")
        dry_run_job = launch_aapl_hpo(dry_run=True)
        if dry_run_job:
            print(f"✅ Dry-run successful: {dry_run_job}")
            
            print("🚀 Launching actual HPO job...")
            job_name = launch_aapl_hpo(dry_run=False)
            if job_name:
                print(f"✅ AAPL HPO job launched successfully: {job_name}")
                return job_name
            else:
                print("❌ Failed to launch AAPL HPO job")
                sys.exit(1)
        else:
            print("❌ Dry-run failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching HPO job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
