#!/usr/bin/env python3
"""
Unit tests for HPO orchestration automation
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.orchestrate_hpo_pipeline import HPOOrchestrator

class TestHPOOrchestration(unittest.TestCase):
    
    def setUp(self):
        self.orchestrator = HPOOrchestrator()
    
    @patch('boto3.client')
    def test_job_name_length_compliance(self, mock_boto):
        """Test that job names comply with AWS 32-character limit"""
        job_name = self.orchestrator.launch_catboost_hpo("s3://test/data", dry_run=True)
        self.assertLessEqual(len(job_name), 32, "Job name exceeds AWS limit")
        self.assertIn("cb-hpo-", job_name, "Job name should use short prefix")
    
    @patch('boto3.client')
    def test_endpoint_monitoring_timeout(self, mock_boto):
        """Test endpoint monitoring with timeout"""
        mock_sagemaker = MagicMock()
        mock_boto.return_value = mock_sagemaker
        mock_sagemaker.describe_endpoint.return_value = {'EndpointStatus': 'Creating'}
        
        result = self.orchestrator.monitor_endpoint_health("test-endpoint", timeout_minutes=1, dry_run=False)
        self.assertFalse(result, "Should timeout for stuck endpoint")
    
    def test_notification_setup_dry_run(self):
        """Test notification setup in dry-run mode"""
        topic_arn = self.orchestrator.setup_notifications("test-endpoint", dry_run=True)
        self.assertIsNotNone(topic_arn, "Should return mock topic ARN in dry-run")
        self.assertIn("arn:aws:sns", topic_arn, "Should return valid ARN format")
    
    @patch('boto3.client')
    def test_multiple_endpoint_monitoring(self, mock_boto):
        """Test monitoring multiple endpoints"""
        mock_sagemaker = MagicMock()
        mock_boto.return_value = mock_sagemaker
        mock_sagemaker.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
        
        endpoints = ["endpoint-1", "endpoint-2"]
        results = self.orchestrator.monitor_and_fix_endpoints(endpoints, timeout_minutes=1, dry_run=True)
        
        self.assertEqual(len(results), 2, "Should monitor all endpoints")
        for endpoint in endpoints:
            self.assertIn(endpoint, results, f"Should have result for {endpoint}")
    
    def test_set_and_forget_dry_run(self):
        """Test set-and-forget mode in dry-run"""
        from scripts.orchestrate_hpo_pipeline import run_full_automation
        
        try:
            run_full_automation("s3://test/data", dry_run=True)
        except Exception as e:
            self.fail(f"Set-and-forget dry-run should not fail: {e}")

if __name__ == '__main__':
    unittest.main()
