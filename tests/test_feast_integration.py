#!/usr/bin/env python3
"""
Test Feast feature store integration
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feast_materialize import (get_online_features, list_feature_views,
                               materialize_features)


class TestFeastIntegration:
    def setup_method(self):
        """Setup test data and temporary directories"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test data
        self.test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=100, freq="H"),
                "ticker": ["AAPL"] * 100,
                "close": [150.0 + i for i in range(100)],
                "volume": [1000 + i * 10 for i in range(100)],
            }
        )

    @patch("feast_materialize.FeatureStore")
    def test_materialize_features_success(self, mock_fs_class):
        """Test successful feature materialization"""
        # Mock FeatureStore
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Test materialization
        result = materialize_features(
            start_date="2025-01-01", end_date="2025-01-16", repo_path=self.temp_dir
        )

        assert result is True
        mock_fs.materialize.assert_called_once()

    @patch("feast_materialize.FeatureStore")
    def test_materialize_features_specific_views(self, mock_fs_class):
        """Test materialization of specific feature views"""
        # Mock FeatureStore
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Test materialization with specific views
        result = materialize_features(
            start_date="2025-01-01",
            end_date="2025-01-16",
            feature_views=["stocks_30min", "options_30min"],
            repo_path=self.temp_dir,
        )

        assert result is True
        # Should be called twice (once for each feature view)
        assert mock_fs.materialize.call_count == 2

    @patch("feast_materialize.FeatureStore")
    def test_get_online_features_success(self, mock_fs_class):
        """Test successful online feature retrieval"""
        # Mock FeatureStore
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Mock feature vector
        mock_feature_vector = MagicMock()
        mock_feature_vector.to_dict.return_value = {
            "ticker": ["AAPL"],
            "stocks_30min:close": [150.0],
            "stocks_30min:volume": [1000],
        }
        mock_fs.get_online_features.return_value = mock_feature_vector

        # Test online feature retrieval
        result = get_online_features(
            entity_rows=[{"ticker": "AAPL"}],
            feature_names=["stocks_30min:close", "stocks_30min:volume"],
            repo_path=self.temp_dir,
        )

        assert result is not None
        assert "ticker" in result
        assert "stocks_30min:close" in result
        mock_fs.get_online_features.assert_called_once()

    @patch("feast_materialize.FeatureStore")
    def test_list_feature_views(self, mock_fs_class):
        """Test listing feature views"""
        # Mock FeatureStore
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs

        # Mock feature views
        mock_fv1 = MagicMock()
        mock_fv1.name = "stocks_30min"
        mock_fv2 = MagicMock()
        mock_fv2.name = "options_30min"

        mock_fs.list_feature_views.return_value = [mock_fv1, mock_fv2]

        # Test listing
        result = list_feature_views(self.temp_dir)

        assert len(result) == 2
        assert "stocks_30min" in result
        assert "options_30min" in result

    @patch("feast_materialize.FeatureStore")
    def test_materialize_features_error_handling(self, mock_fs_class):
        """Test error handling in feature materialization"""
        # Mock FeatureStore to raise exception
        mock_fs_class.side_effect = Exception("Connection failed")

        # Test materialization with error
        result = materialize_features(start_date="2025-01-01", repo_path=self.temp_dir)

        assert result is False

    @patch("feast_materialize.FeatureStore")
    def test_get_online_features_error_handling(self, mock_fs_class):
        """Test error handling in online feature retrieval"""
        # Mock FeatureStore to raise exception
        mock_fs_class.side_effect = Exception("Connection failed")

        # Test online feature retrieval with error
        result = get_online_features(
            entity_rows=[{"ticker": "AAPL"}],
            feature_names=["stocks_30min:close"],
            repo_path=self.temp_dir,
        )

        assert result is None


def test_feast_integration_imports():
    """Test that Feast integration utilities import correctly"""
    try:
        from feast_materialize import get_online_features, materialize_features
        from inference_with_feast import get_inference_features, run_inference

        assert callable(materialize_features)
        assert callable(get_online_features)
        assert callable(run_inference)
        assert callable(get_inference_features)
    except ImportError as e:
        pytest.fail(f"Feast integration import failed: {e}")


@patch("feast_materialize.FeatureStore")
def test_inference_with_feast_mock(mock_fs_class):
    """Test inference workflow with mocked Feast"""
    from inference_with_feast import get_inference_features

    # Mock FeatureStore
    mock_fs = MagicMock()
    mock_fs_class.return_value = mock_fs

    # Mock feature vector
    mock_feature_vector = MagicMock()
    mock_feature_vector.to_dict.return_value = {
        "ticker": ["AAPL"],
        "stocks_30min:close": [150.0],
        "stocks_30min:volume": [1000],
    }
    mock_fs.get_online_features.return_value = mock_feature_vector

    # Test inference feature retrieval
    result = get_inference_features(tickers=["AAPL"], feature_views=["stocks_30min"])

    assert result is not None
    assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
