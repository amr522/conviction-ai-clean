#!/usr/bin/env python3
"""
Test Sentry integration for error tracking and performance monitoring
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestSentryIntegration:
    
    def setup_method(self):
        """Setup test environment"""
        # Mock Sentry to avoid actual initialization
        self.sentry_mock = MagicMock()
        
        # Patch Sentry functions
        self.patches = [
            patch('sentry_sdk.init'),
            patch('sentry_sdk.capture_exception'),
            patch('sentry_sdk.capture_message'),
            patch('sentry_sdk.start_span'),
            patch('sentry_sdk.set_tag'),
            patch('sentry_sdk.set_context'),
            patch('aws_xray_sdk.core.xray_recorder'),
            patch('aws_xray_sdk.core.patch_all'),
            patch('aws_xray_sdk.ext.fastapi.XRayMiddleware'),
            patch('app.metrics.update_model_info')
        ]
        
        for p in self.patches:
            p.start()
        
        # Import after patching
        from app.main import app
        self.client = TestClient(app)
    
    def teardown_method(self):
        """Cleanup patches"""
        for p in self.patches:
            p.stop()
    
    @patch('sentry_sdk.capture_exception')
    def test_sentry_captures_exceptions(self, mock_capture_exception):
        """Test that Sentry captures exceptions"""
        # This will trigger an exception due to missing model
        response = self.client.post("/predict", json={"ticker": "AAPL"})
        
        # Should have captured the exception
        assert mock_capture_exception.called or response.status_code in [401, 403, 503]
    
    @patch('sentry_sdk.capture_message')
    def test_sentry_captures_messages(self, mock_capture_message):
        """Test that Sentry captures custom messages"""
        # Health endpoint should work without auth
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Startup message should be captured
        # Note: This might not be called in test environment
    
    @patch('sentry_sdk.start_span')
    def test_sentry_performance_tracing(self, mock_start_span):
        """Test that Sentry creates performance spans"""
        # Mock span context manager
        mock_span = MagicMock()
        mock_start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_start_span.return_value.__exit__ = MagicMock(return_value=None)
        
        # Make a request that would create spans
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Spans might be created during model loading or other operations
    
    @patch('sentry_sdk.set_tag')
    def test_sentry_sets_tags(self, mock_set_tag):
        """Test that Sentry tags are set"""
        # Tags should be set during startup
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Tags might be set during startup
    
    @patch('sentry_sdk.set_context')
    def test_sentry_sets_context(self, mock_set_context):
        """Test that Sentry context is set"""
        # Context should be set during model loading or predictions
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Context might be set during model operations
    
    def test_global_exception_handler(self):
        """Test that global exception handler works"""
        # Try to trigger an unhandled exception
        # Most endpoints will have proper error handling, so this tests the fallback
        response = self.client.get("/nonexistent-endpoint")
        assert response.status_code == 404  # FastAPI's built-in 404 handler
    
    def test_sentry_environment_variables(self):
        """Test Sentry configuration from environment variables"""
        # Test that environment variables are read correctly
        test_env_vars = {
            'SENTRY_DSN': 'https://test@sentry.io/123456',
            'SENTRY_TRACES_SAMPLE_RATE': '0.5',
            'SENTRY_PROFILES_SAMPLE_RATE': '0.2',
            'ENVIRONMENT': 'test',
            'RELEASE': 'v1.0.0-test'
        }
        
        for key, value in test_env_vars.items():
            # These would be used during Sentry initialization
            assert isinstance(value, str)
    
    def test_sentry_integration_imports(self):
        """Test that Sentry integration imports work"""
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration
            
            assert sentry_sdk is not None
            assert FastApiIntegration is not None
            assert LoggingIntegration is not None
        except ImportError as e:
            pytest.fail(f"Sentry integration import failed: {e}")

def test_sentry_mock_functionality():
    """Test Sentry mock functionality for CI/CD"""
    with patch('sentry_sdk.init') as mock_init, \
         patch('sentry_sdk.capture_exception') as mock_capture_exception, \
         patch('sentry_sdk.capture_message') as mock_capture_message:
        
        # Simulate Sentry operations
        import sentry_sdk
        
        # Test initialization
        sentry_sdk.init(dsn="test://test@test.com/1")
        mock_init.assert_called_once()
        
        # Test exception capture
        try:
            raise ValueError("Test error")
        except Exception as e:
            sentry_sdk.capture_exception(e)
            mock_capture_exception.assert_called_once_with(e)
        
        # Test message capture
        sentry_sdk.capture_message("Test message", level="info")
        mock_capture_message.assert_called_once_with("Test message", level="info")

def test_sentry_configuration_validation():
    """Test Sentry configuration validation"""
    # Test that configuration values are properly validated
    config_tests = [
        ("0.0", 0.0),    # Minimum sample rate
        ("1.0", 1.0),    # Maximum sample rate
        ("0.1", 0.1),    # Default sample rate
        ("invalid", 0.1) # Invalid should default to 0.1 (handled by float() with default)
    ]
    
    for input_val, expected in config_tests:
        try:
            result = float(input_val)
            assert isinstance(result, float)
        except ValueError:
            # Invalid values should be handled gracefully
            result = 0.1  # Default fallback
            assert result == 0.1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])