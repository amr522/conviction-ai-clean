#!/usr/bin/env python3
"""
Check the status of all SageMaker HPO jobs
"""
import boto3
import argparse
from datetime import datetime
import time
import sys

def format_duration(seconds):
    """Format seconds into a human-readable duration"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def check_hpo_jobs(symbol=None, status_filter=None):
    """Check status of HPO jobs
    
    Args:
        symbol: Optional symbol to filter by (lowercase)
        status_filter: Optional status to filter by
    """
    sm = boto3.client("sagemaker")
    
    # Get all HPO jobs
    print("Fetching HPO jobs...")
    response = sm.list_hyper_parameter_tuning_jobs()
    jobs = response["HyperParameterTuningJobSummaries"]
    
    # Filter by symbol if provided
    if symbol:
        symbol = symbol.lower()
        jobs = [job for job in jobs if symbol in job["HyperParameterTuningJobName"].lower()]
    
    # Filter by status if provided
    if status_filter:
        jobs = [job for job in jobs if job["HyperParameterTuningJobStatus"] == status_filter]
    
    # Sort by creation time (newest first)
    jobs = sorted(jobs, key=lambda x: x["CreationTime"], reverse=True)
    
    if not jobs:
        print("No matching HPO jobs found.")
        return
    
    print(f"Found {len(jobs)} HPO jobs:")
    print(f"{'Status':<12} {'Name':<30} {'Best Obj':<12} {'Duration':<10} {'Created'}")
    print("-" * 75)
    
    for job in jobs:
        name = job["HyperParameterTuningJobName"]
        status = job["HyperParameterTuningJobStatus"]
        
        # Calculate duration
        now = datetime.now().replace(tzinfo=job["CreationTime"].tzinfo)
        duration = (now - job["CreationTime"]).total_seconds()
        duration_str = format_duration(duration)
        
        # Format creation time
        created = job["CreationTime"].strftime("%Y-%m-%d %H:%M")
        
        # Get best objective if available
        best_obj = "N/A"
        if "BestTrainingJob" in job:
            best_obj = f"{job['BestTrainingJob']['FinalHyperParameterTuningJobObjectiveMetric']['Value']:.4f}"
        
        status_color = "\033[32m" if status == "Completed" else "\033[33m" if status == "InProgress" else "\033[31m"
        print(f"{status_color}{status}\033[0m{'':<6} {name:<30} {best_obj:<12} {duration_str:<10} {created}")
    
    # More detailed info about completed jobs
    completed_jobs = [job for job in jobs if job["HyperParameterTuningJobStatus"] == "Completed"]
    if completed_jobs:
        print("\nCompleted jobs details:")
        for job in completed_jobs:
            print(f"\n{job['HyperParameterTuningJobName']}:")
            
            # Get job details
            details = sm.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=job["HyperParameterTuningJobName"]
            )
            
            # Print training job counts
            training_job_statuses = details.get("TrainingJobStatusCounters", {})
            print(f"  Training jobs: {training_job_statuses.get('Completed', 0)} completed, "
                  f"{training_job_statuses.get('InProgress', 0)} in progress, "
                  f"{training_job_statuses.get('Failed', 0)} failed")
            
            # Print best training job info if available
            if "BestTrainingJob" in details:
                best_job = details["BestTrainingJob"]
                print(f"  Best job: {best_job['TrainingJobName']}")
                print(f"  Objective: {best_job['FinalHyperParameterTuningJobObjectiveMetric']['Name']} = "
                      f"{best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']:.4f}")
                
                # Print hyperparameters
                print("  Best hyperparameters:")
                for param, value in best_job["TunedHyperParameters"].items():
                    print(f"    {param}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check SageMaker HPO job status")
    parser.add_argument("--symbol", "-s", help="Filter by symbol (e.g., msft)")
    parser.add_argument("--status", "-t", help="Filter by status (e.g., Completed, InProgress, Failed)")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode - update every 60 seconds")
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("Watch mode enabled. Press Ctrl+C to exit.")
            while True:
                check_hpo_jobs(args.symbol, args.status)
                print("\nRefreshing in 60 seconds... (Ctrl+C to exit)")
                time.sleep(60)
                print("\033c", end="")  # Clear screen
        else:
            check_hpo_jobs(args.symbol, args.status)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
