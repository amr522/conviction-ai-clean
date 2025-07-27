#!/usr/bin/env python3
"""
Test FastAPI inference service
"""
import pytest
import json
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock X-Ray to avoid dependency during testing
with patch('aws_xray_sdk.core.xray_recorder'), \
     patch('aws_xray_sdk.core.patch_all'), \
     patch('aws_xray_sdk.ext.fastapi.XRayMiddleware'):
    from app.main import app

class TestInferenceAPI:
    
    def setup_method(self):
        """Setup test client and mock data"""
        self.client = TestClient(app)
        
        # Mock model for testing
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [0.0234]
        
        # Mock model metadata
        self.mock_metadata = {
            'version': '1.0.0',
            'created_at': '2025-01-16T10:00:00',
            'features': [
                'stocks_30min:close',
                'stocks_30min:volume',
                'options_30min:opt30_close',
                'options_daily:optd_iv30'
            ]
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "feature_store_connected" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Conviction AI Inference API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    @patch('app.main._model')
    @patch('app.main._model_metadata')
    @patch('app.main.get_features_from_store')
    def test_predict_endpoint_with_feature_store(self, mock_get_features, mock_metadata, mock_model):
        """Test prediction endpoint with feature store"""
        # Setup mocks
        mock_model.return_value = self.mock_model
        mock_metadata.return_value = self.mock_metadata
        mock_get_features.return_value = {
            'stocks_30min:close': 150.0,
            'stocks_30min:volume': 100000,
            'options_30min:opt30_close': 5.0,
            'options_daily:optd_iv30': 0.25
        }
        
        # Make request
        request_data = {
            "ticker": "AAPL",
            "timestamp": "2025-01-16T15:30:00Z"
        }
        
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "prediction" in data
        assert "features_used" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
    
    @patch('app.main._model')
    @patch('app.main._model_metadata')
    def test_predict_endpoint_with_manual_features(self, mock_metadata, mock_model):
        """Test prediction endpoint with manual features"""
        # Setup mocks
        mock_model.return_value = self.mock_model
        mock_metadata.return_value = self.mock_metadata
        
        # Make request with manual features
        request_data = {
            "ticker": "MSFT",
            "features": {
                "stocks_30min:close": 200.0,
                "stocks_30min:volume": 150000,
                "options_30min:opt30_close": 8.0,
                "options_daily:optd_iv30": 0.30
            }
        }
        
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["ticker"] == "MSFT"
        assert data["features_used"]["stocks_30min:close"] == 200.0
    
    def test_predict_endpoint_no_model(self):
        """Test prediction endpoint when model is not loaded"""
        request_data = {
            "ticker": "AAPL"
        }
        
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_endpoint_invalid_ticker(self):
        """Test prediction endpoint with invalid ticker"""
        request_data = {
            "ticker": "INVALID_TICKER_TOO_LONG"
        }
        
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_timestamp(self):
        """Test prediction endpoint with invalid timestamp"""
        request_data = {
            "ticker": "AAPL",
            "timestamp": "invalid-timestamp"
        }
        
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('app.main._model')
    @patch('app.main._model_metadata')
    @patch('app.main.get_features_from_store')
    def test_batch_predict_endpoint(self, mock_get_features, mock_metadata, mock_model):
        """Test batch prediction endpoint"""
        # Setup mocks
        mock_model.return_value = self.mock_model
        mock_metadata.return_value = self.mock_metadata
        mock_get_features.return_value = {
            'stocks_30min:close': 150.0,
            'stocks_30min:volume': 100000,
            'options_30min:opt30_close': 5.0,
            'options_daily:optd_iv30': 0.25
        }
        
        # Make batch request
        request_data = {
            "requests": [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"},
                {"ticker": "GOOGL"}
            ]
        }
        
        response = self.client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_requests"] == 3
        assert len(data["predictions"]) == 3
        assert "successful_predictions" in data
        assert "failed_predictions" in data
    
    def test_batch_predict_too_many_requests(self):
        """Test batch prediction with too many requests"""
        request_data = {
            "requests": [{"ticker": f"TICK{i}"} for i in range(101)]  # Over limit
        }
        
        response = self.client.post("/predict/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('app.main.load_model')
    def test_model_reload_endpoint(self, mock_load_model):
        """Test model reload endpoint"""
        mock_load_model.return_value = True
        
        response = self.client.post("/model/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "reloaded successfully" in data["message"]
    
    @patch('app.main.load_model')
    def test_model_reload_failure(self, mock_load_model):
        """Test model reload endpoint failure"""
        mock_load_model.return_value = False
        
        response = self.client.post("/model/reload")
        assert response.status_code == 500
        assert "Failed to reload model" in response.json()["detail"]
    
    @patch('app.main._model')
    @patch('app.main._model_metadata')
    def test_model_info_endpoint(self, mock_metadata, mock_model):
        """Test model info endpoint"""
        mock_model.return_value = self.mock_model
        mock_metadata.return_value = self.mock_metadata
        
        response = self.client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_loaded"] is True
        assert "model_type" in data
        assert "metadata" in data
        assert "feature_count" in data
    
    def test_model_info_no_model(self):
        """Test model info endpoint when no model is loaded"""
        response = self.client.get("/model/info")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

def test_feature_store_utils():
    """Test feature store utility functions"""
    try:
        from utils.feature_store import generate_mock_features, validate_features
        
        # Test mock feature generation
        mock_features = generate_mock_features("AAPL")
        assert isinstance(mock_features, dict)
        assert len(mock_features) > 0
        
        # Test feature validation
        required_features = ["feature1", "feature2", "feature3"]
        raw_features = {"feature1": 1.0, "feature2": True, "feature4": "invalid"}
        
        validated = validate_features(raw_features, required_features)
        assert len(validated) == len(required_features)
        assert validated["feature1"] == 1.0
        assert validated["feature2"] == 1.0  # Boolean converted to float
        assert validated["feature3"] == 0.0  # Missing feature gets default
        
    except ImportError as e:
        pytest.fail(f"Feature store utilities import failed: {e}")

def test_api_imports():
    """Test that API modules import correctly"""
    try:
        from app.main import app, InferenceRequest, InferenceResponse
        assert app is not None
        assert InferenceRequest is not None
        assert InferenceResponse is not None
    except ImportError as e:
        pytest.fail(f"API import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])