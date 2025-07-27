"""Tests for OpenLineage integration."""

import pytest
from unittest.mock import Mock, patch
from src.utils.lineage_utils import LineageTracker, track_lineage


class TestLineageTracker:
    """Test LineageTracker functionality."""
    
    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_tracker_with_client(self, mock_client_class):
        """Test tracker when OpenLineage client is available."""
        mock_client = Mock()
        mock_client_class.from_environment.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENLINEAGE_URL': 'http://localhost:5000'}):
            tracker = LineageTracker()
            
            # Test start_run
            tracker.start_run(
                "test_job",
                inputs=["input1", "input2"],
                outputs=["output1"]
            )
            
            assert mock_client.emit.called
            
            # Test complete_run
            tracker.complete_run(success=True)
            assert mock_client.emit.call_count == 2
    
    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_tracker_without_client(self, mock_client_class):
        """Test tracker when OpenLineage client is not available."""
        mock_client_class.from_environment.return_value = None
        
        tracker = LineageTracker()
        
        # Should not raise errors
        tracker.start_run("test_job", inputs=["input1"], outputs=["output1"])
        tracker.complete_run(success=True)
    
    @patch('src.utils.lineage_utils.LineageTracker')
    def test_track_lineage_decorator(self, mock_tracker_class):
        """Test track_lineage decorator."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        @track_lineage("test_job", ["input1"], ["output1"])
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_tracker.start_run.assert_called_once()
        mock_tracker.complete_run.assert_called_with(success=True)
    
    @patch('src.utils.lineage_utils.LineageTracker')
    def test_track_lineage_decorator_with_exception(self, mock_tracker_class):
        """Test track_lineage decorator when function raises exception."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        @track_lineage("test_job", ["input1"], ["output1"])
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        mock_tracker.start_run.assert_called_once()
        mock_tracker.complete_run.assert_called_with(success=False)