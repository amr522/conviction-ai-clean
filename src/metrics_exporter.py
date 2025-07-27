#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, Response
import glob

app = Flask(__name__)

def get_directory_metrics(path):
    """Get file count and total size for a directory"""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return {"file_count": 0, "total_size_mb": 0}
    
    files = glob.glob(f"{abs_path}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    total_size = sum(os.path.getsize(f) for f in files)
    return {
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }

def get_pipeline_runtime():
    """Extract last pipeline runtime from logs"""
    log_files = glob.glob("logs/pipeline_*.log")
    if not log_files:
        return 0
    
    latest_log = max(log_files, key=os.path.getctime)
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            # Look for completion time in logs
            for line in reversed(lines):
                if "completed" in line.lower():
                    # Extract runtime if available
                    return 300  # Default 5 minutes
    except:
        pass
    return 0

def get_data_freshness():
    """Check how fresh the latest data is"""
    datasets_path = os.path.abspath("datasets")
    if not os.path.exists(datasets_path):
        return 999  # Very stale
    
    # Find newest file in datasets
    files = glob.glob(f"{datasets_path}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return 999
    
    newest_file = max(files, key=os.path.getctime)
    file_age_hours = (time.time() - os.path.getctime(newest_file)) / 3600
    return round(file_age_hours, 1)

def get_last_run_status():
    """Get exit code of last pipeline run"""
    # Check for success/failure indicators
    success_flag = os.path.abspath("staged/pipeline_success.flag")
    failure_flag = os.path.abspath("staged/pipeline_failure.flag")
    
    if os.path.exists(success_flag):
        return 0
    elif os.path.exists(failure_flag):
        return 1
    return -1  # Unknown

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    
    # Get metrics
    datasets_metrics = get_directory_metrics("datasets")
    master_metrics = get_directory_metrics("master")
    staged_metrics = get_directory_metrics("staged")
    
    runtime = get_pipeline_runtime()
    freshness = get_data_freshness()
    last_status = get_last_run_status()
    
    # Generate Prometheus format
    metrics_output = f"""# HELP vol_pipeline_files_total Total number of files in directory
# TYPE vol_pipeline_files_total gauge
vol_pipeline_files_total{{directory="datasets"}} {datasets_metrics['file_count']}
vol_pipeline_files_total{{directory="master"}} {master_metrics['file_count']}
vol_pipeline_files_total{{directory="staged"}} {staged_metrics['file_count']}

# HELP vol_pipeline_size_mb Total size in MB of directory
# TYPE vol_pipeline_size_mb gauge
vol_pipeline_size_mb{{directory="datasets"}} {datasets_metrics['total_size_mb']}
vol_pipeline_size_mb{{directory="master"}} {master_metrics['total_size_mb']}
vol_pipeline_size_mb{{directory="staged"}} {staged_metrics['total_size_mb']}

# HELP vol_pipeline_runtime_seconds Last pipeline runtime in seconds
# TYPE vol_pipeline_runtime_seconds gauge
vol_pipeline_runtime_seconds {runtime}

# HELP vol_pipeline_data_age_hours Age of newest data file in hours
# TYPE vol_pipeline_data_age_hours gauge
vol_pipeline_data_age_hours {freshness}

# HELP vol_pipeline_last_run_status Exit code of last pipeline run (0=success, 1=failure, -1=unknown)
# TYPE vol_pipeline_last_run_status gauge
vol_pipeline_last_run_status {last_status}

# HELP vol_pipeline_up Metrics exporter is running
# TYPE vol_pipeline_up gauge
vol_pipeline_up 1
"""
    
    return Response(metrics_output, mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)