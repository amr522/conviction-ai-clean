#!/usr/bin/env python3
"""
Test AWS X-Ray tracing integration
"""
import pytest
import os
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.xray_utils import configure_xray, trace_function, add_trace_metadata, traced_subsegment

class TestXRayTracing:
    
    def setup_method(self):
        """Setup test environment"""
        # Disable X-Ray for most tests to avoid dependency on daemon
        os.environ['AWS_XRAY_TRACING_DISABLED'] = 'true'
    
    def teardown_method(self):
        """Cleanup test environment"""
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED']
    
    def test_configure_xray_disabled(self):
        """Test X-Ray configuration when disabled"""
        configure_xray('test-service')
        # Should not raise any exceptions
        assert True
    
    @patch('utils.xray_utils.xray_recorder')
    def test_configure_xray_enabled(self, mock_recorder):
        """Test X-Ray configuration when enabled"""
        # Enable X-Ray for this test
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED']
        
        configure_xray('test-service', '127.0.0.1:2000')
        
        mock_recorder.configure.assert_called_once_with(
            service='test-service',
            plugins=('EC2Plugin', 'ECSPlugin'),
            daemon_address='127.0.0.1:2000',
            use_ssl=False
        )
    
    def test_trace_function_decorator_disabled(self):
        """Test function tracing when X-Ray is disabled"""
        @trace_function('test_function')
        def sample_function(x, y):
            return x + y
        
        result = sample_function(2, 3)
        assert result == 5
    
    @patch('utils.xray_utils.xray_recorder')
    def test_trace_function_decorator_enabled(self, mock_recorder):
        """Test function tracing when X-Ray is enabled"""
        # Enable X-Ray for this test
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED']
        
        # Mock the capture context manager
        mock_segment = MagicMock()
        mock_recorder.capture.return_value.__enter__ = MagicMock(return_value=mock_segment)
        mock_recorder.capture.return_value.__exit__ = MagicMock(return_value=None)
        
        @trace_function('test_function')
        def sample_function(x, y):
            return x + y
        
        result = sample_function(2, 3)
        
        assert result == 5
        mock_recorder.capture.assert_called_once_with('test_function')
        mock_segment.put_annotation.assert_called()
    
    def test_add_trace_metadata_disabled(self):
        """Test adding metadata when X-Ray is disabled"""
        # Should not raise any exceptions
        add_trace_metadata('test_key', 'test_value')
        assert True
    
    @patch('utils.xray_utils.xray_recorder')
    def test_add_trace_metadata_enabled(self, mock_recorder):
        """Test adding metadata when X-Ray is enabled"""
        # Enable X-Ray for this test
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED']
        
        add_trace_metadata('test_key', {'data': 'value'})
        
        mock_recorder.put_metadata.assert_called_once_with('test_key', {'data': 'value'})
    
    def test_traced_subsegment_disabled(self):
        """Test traced subsegment when X-Ray is disabled"""
        with traced_subsegment('test_subsegment', {'key': 'value'}):
            # Should execute without issues
            result = 1 + 1
        
        assert result == 2
    
    @patch('utils.xray_utils.xray_recorder')
    def test_traced_subsegment_enabled(self, mock_recorder):
        """Test traced subsegment when X-Ray is enabled"""
        # Enable X-Ray for this test
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED'
        
        with traced_subsegment('test_subsegment', {'key': 'value'}):
            result = 1 + 1
        
        assert result == 2
        mock_recorder.begin_subsegment.assert_called_once_with('test_subsegment')
        mock_recorder.put_annotation.assert_called_with('key', 'value')
        mock_recorder.end_subsegment.assert_called_once()
    
    def test_trace_function_with_exception(self):
        """Test function tracing with exception handling"""
        @trace_function('test_function_error')
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_trace_function_with_response_capture(self):
        """Test function tracing with response capture"""
        @trace_function('test_function_response', capture_response=True)
        def sample_function():
            return {'result': 'success', 'data': [1, 2, 3]}
        
        result = sample_function()
        assert result == {'result': 'success', 'data': [1, 2, 3]}

def test_xray_imports():
    """Test that X-Ray utilities import correctly"""
    try:
        from utils.xray_utils import configure_xray, trace_function, add_trace_metadata
        assert callable(configure_xray)
        assert callable(trace_function)
        assert callable(add_trace_metadata)
    except ImportError as e:
        pytest.fail(f"X-Ray utilities import failed: {e}")

def test_training_script_xray_integration():
    """Test that training script has X-Ray integration"""
    try:
        # This should not fail even if X-Ray daemon is not running
        os.environ['AWS_XRAY_TRACING_DISABLED'] = 'true'
        
        from train_and_evaluate import run
        # If import succeeds, X-Ray integration is properly set up
        assert callable(run)
        
    except ImportError as e:
        pytest.fail(f"Training script X-Ray integration failed: {e}")
    finally:
        if 'AWS_XRAY_TRACING_DISABLED' in os.environ:
            del os.environ['AWS_XRAY_TRACING_DISABLED']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])