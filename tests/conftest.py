"""
Test configuration and fixtures for the conviction-ai test suite.
"""
import sys
from unittest.mock import MagicMock

# Mock external modules that may not be available in test environment
MOCK_MODULES = [
    'feast',
    'feast.FeatureStore',
    'openlineage.client',
    'openlineage.client.dataset',
    'openlineage.client.facet',
    'openlineage.client.run',
    'evidently',
    'evidently.metric_preset',
    'evidently.metrics',
    'evidently.report',
    'great_expectations',
    'delta',
    'delta.configure_spark_with_delta_pip',
    'pyspark',
    'pyspark.sql',
    'aws_xray_sdk.ext.fastapi',
    'prefect.tasks.shell_run_command',
]

for module in MOCK_MODULES:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Set environment variables for testing
import os
os.environ['AWS_XRAY_TRACING_DISABLED'] = 'true'
os.environ['TESTING'] = 'true'