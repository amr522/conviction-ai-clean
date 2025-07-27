"""Tests for src.utils.lineage_utils module."""

import pytest
from unittest.mock import Mock, patch
from src.utils.lineage_utils import LineageTracker, track_lineage


class TestLineageUtils:
    """Test class for lineage_utils module."""

    def test_track_lineage_exists(self):
        """Test that track_lineage function exists."""
        assert callable(track_lineage)
    
    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_track_lineage_basic(self, mock_client):
        """Test basic functionality of track_lineage."""
        mock_client.from_environment.return_value = Mock()
        
        @track_lineage('test_job', ['input.csv'], ['output.csv'])
        def dummy_function():
            return "test"
        
        result = dummy_function()
        assert result == "test"

    def test_lineagetracker_exists(self):
        """Test that LineageTracker class exists."""
        assert LineageTracker is not None
    
    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_lineagetracker_instantiation(self, mock_client):
        """Test LineageTracker can be instantiated."""
        mock_client.from_environment.return_value = Mock()
        tracker = LineageTracker('test_namespace')
        assert tracker is not None
        assert hasattr(tracker, 'namespace')
        assert tracker.namespace == 'test_namespace'

    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_lineagetracker_start_run(self, mock_client):
        """Test LineageTracker start_run method."""
        mock_client.from_environment.return_value = Mock()
        tracker = LineageTracker()
        
        # Test with no client (should return None)
        tracker.client = None
        result = tracker.start_run('test_job', ['input'], ['output'])
        assert result is None

    @patch('src.utils.lineage_utils.OpenLineageClient')
    def test_lineagetracker_complete_run(self, mock_client):
        """Test LineageTracker complete_run method."""
        mock_client.from_environment.return_value = Mock()
        tracker = LineageTracker()
        
        # Test with no client (should not raise error)
        tracker.client = None
        tracker.complete_run()  # Should not raise

    def test_lineage_utils_integration(self):
        """Integration test for lineage_utils module."""
        # Test that we can import and instantiate without OpenLineage client
        tracker = LineageTracker()
        assert tracker.namespace == "conviction_ai"
        assert tracker.client is None  # No environment setup