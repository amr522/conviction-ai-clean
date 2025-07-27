#!/usr/bin/env python3

import pytest
import requests
import threading
import time
import os
import json
from pathlib import Path
import tempfile
import sys
import socket

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def find_free_port():
    """Find a free port for testing"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class MetricsExporterTestServer:
    """Test server wrapper for metrics exporter"""
    
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.server_thread = None
        self.port = find_free_port()
        self.original_cwd = None
        
    def start(self):
        """Start the Flask server in a background thread"""
        # Import here to avoid issues
        from metrics_exporter import app
        
        # Change working directory to test directory
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        def run_server():
            app.run(host='127.0.0.1', port=self.port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        for _ in range(50):  # 5 second timeout
            try:
                response = requests.get(f'http://127.0.0.1:{self.port}/health', timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(0.1)
        else:
            if self.original_cwd:
                os.chdir(self.original_cwd)
            raise RuntimeError("Server failed to start")
    
    def stop(self):
        """Stop the server and restore working directory"""
        if self.original_cwd:
            os.chdir(self.original_cwd)

@pytest.fixture
def test_data_dir(tmp_path):
    """Create test directory structure with mock data"""
    # Create directory structure
    (tmp_path / "datasets").mkdir()
    (tmp_path / "master").mkdir()
    (tmp_path / "staged").mkdir()
    (tmp_path / "logs").mkdir()
    
    return tmp_path

@pytest.fixture
def metrics_server(test_data_dir):
    """Start metrics exporter server for testing"""
    server = MetricsExporterTestServer(test_data_dir)
    server.start()
    yield server
    server.stop()

def test_health_endpoint(metrics_server):
    """Test the /health endpoint returns correct response"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/health')
    
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_metrics_endpoint_format(metrics_server):
    """Test the /metrics endpoint returns Prometheus format"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    
    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/plain; charset=utf-8'
    
    content = response.text
    assert '# HELP' in content
    assert '# TYPE' in content

def test_metrics_endpoint_required_metrics(metrics_server):
    """Test that all required metrics are present"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Check for required metrics
    required_metrics = [
        'vol_pipeline_up',
        'vol_pipeline_last_run_status',
        'vol_pipeline_files_total',
        'vol_pipeline_data_age_hours',
        'vol_pipeline_runtime_seconds',
        'vol_pipeline_size_mb'
    ]
    
    for metric in required_metrics:
        assert metric in content, f"Missing metric: {metric}"

def test_metrics_up_status(metrics_server):
    """Test that vol_pipeline_up is always 1"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Find the vol_pipeline_up metric
    lines = content.split('\n')
    up_line = [line for line in lines if line.startswith('vol_pipeline_up ')]
    
    assert len(up_line) == 1
    assert up_line[0] == 'vol_pipeline_up 1'

def test_empty_directories_metrics(metrics_server, test_data_dir):
    """Test metrics with empty directories"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Check file counts are 0 for empty directories
    lines = content.split('\n')
    file_count_lines = [line for line in lines if 'vol_pipeline_files_total{' in line]
    
    for line in file_count_lines:
        if 'directory="datasets"' in line or 'directory="master"' in line or 'directory="staged"' in line:
            assert line.endswith(' 0'), f"Expected 0 files, got: {line}"

def test_missing_output_simulation(metrics_server, test_data_dir):
    """Test pipeline status when master directory is empty"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Find last run status - should be -1 (unknown) when no flags present
    lines = content.split('\n')
    status_line = [line for line in lines if line.startswith('vol_pipeline_last_run_status ')]
    
    assert len(status_line) == 1
    assert status_line[0] == 'vol_pipeline_last_run_status -1'

def test_pipeline_failure_flag(metrics_server, test_data_dir):
    """Test pipeline status when failure flag is present"""
    # Create failure flag
    (test_data_dir / "staged" / "pipeline_failure.flag").touch()
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Find last run status - should be 1 (failure)
    lines = content.split('\n')
    status_line = [line for line in lines if line.startswith('vol_pipeline_last_run_status ')]
    
    assert len(status_line) == 1
    assert status_line[0] == 'vol_pipeline_last_run_status 1'

def test_pipeline_success_flag(metrics_server, test_data_dir):
    """Test pipeline status when success flag is present"""
    # Create success flag
    (test_data_dir / "staged" / "pipeline_success.flag").touch()
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Find last run status - should be 0 (success)
    lines = content.split('\n')
    status_line = [line for line in lines if line.startswith('vol_pipeline_last_run_status ')]
    
    assert len(status_line) == 1
    assert status_line[0] == 'vol_pipeline_last_run_status 0'

def test_fresh_data_simulation(metrics_server, test_data_dir):
    """Test data freshness with newly created files"""
    # Create fresh parquet files
    datasets_dir = test_data_dir / "datasets"
    (datasets_dir / "fresh_data.parquet").touch()
    
    master_dir = test_data_dir / "master"
    (master_dir / "master_data.parquet").touch()
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    # Find data age - should be very low (< 1 hour)
    lines = content.split('\n')
    age_line = [line for line in lines if line.startswith('vol_pipeline_data_age_hours ')]
    
    assert len(age_line) == 1
    age_value = float(age_line[0].split(' ')[1])
    assert age_value < 1.0, f"Expected fresh data (<1 hour), got {age_value} hours"

def test_file_count_with_data(metrics_server, test_data_dir):
    """Test file counts when directories contain files"""
    # Create test files
    (test_data_dir / "datasets" / "file1.parquet").touch()
    (test_data_dir / "datasets" / "file2.parquet").touch()
    (test_data_dir / "master" / "master.parquet").touch()
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    lines = content.split('\n')
    
    # Check datasets has 2 files
    datasets_line = [line for line in lines if 'vol_pipeline_files_total{directory="datasets"}' in line]
    assert len(datasets_line) == 1
    assert datasets_line[0].endswith(' 2')
    
    # Check master has 1 file
    master_line = [line for line in lines if 'vol_pipeline_files_total{directory="master"}' in line]
    assert len(master_line) == 1
    assert master_line[0].endswith(' 1')

def test_size_metrics_with_data(metrics_server, test_data_dir):
    """Test size metrics when directories contain files"""
    # Create test file with some content
    test_file = test_data_dir / "datasets" / "test.parquet"
    test_file.write_text("x" * 10240)  # 10KB file to ensure non-zero MB value
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    lines = content.split('\n')
    size_line = [line for line in lines if 'vol_pipeline_size_mb{directory="datasets"}' in line]
    
    assert len(size_line) == 1
    size_value = float(size_line[0].split(' ')[1])
    assert size_value >= 0.01, f"Expected size >= 0.01 MB, got {size_value} MB"

def test_runtime_metric_default(metrics_server):
    """Test runtime metric returns default value when no logs"""
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    lines = content.split('\n')
    runtime_line = [line for line in lines if line.startswith('vol_pipeline_runtime_seconds ')]
    
    assert len(runtime_line) == 1
    runtime_value = float(runtime_line[0].split(' ')[1])
    assert runtime_value == 0  # Default when no logs

def test_stale_data_simulation(metrics_server, test_data_dir):
    """Test data age when no files exist (should be very stale)"""
    # Don't create any files - datasets directory is empty
    
    response = requests.get(f'http://127.0.0.1:{metrics_server.port}/metrics')
    content = response.text
    
    lines = content.split('\n')
    age_line = [line for line in lines if line.startswith('vol_pipeline_data_age_hours ')]
    
    assert len(age_line) == 1
    age_value = float(age_line[0].split(' ')[1])
    assert age_value == 999  # Default for no files