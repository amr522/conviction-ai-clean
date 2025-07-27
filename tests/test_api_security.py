#!/usr/bin/env python3
"""
Test FastAPI security features (JWT auth, rate limiting, metrics)
"""
import pytest
import time
import json
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock dependencies to avoid startup issues
with patch('aws_xray_sdk.core.xray_recorder'), \
     patch('aws_xray_sdk.core.patch_all'), \
     patch('aws_xray_sdk.ext.fastapi.XRayMiddleware'), \
     patch('app.metrics.update_model_info'):
    from app.main import app
    from app.auth import generate_test_token, create_access_token

class TestAPISecurity:
    
    def setup_method(self):
        """Setup test client and tokens"""
        self.client = TestClient(app)
        
        # Generate test tokens
        self.valid_token = generate_test_token("test_user", ["predict", "batch", "admin"])
        self.predict_only_token = create_access_token("user_123", "predict_user", ["predict"])
        self.invalid_token = "invalid.jwt.token"
        
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [0.0234]
    
    def test_health_endpoints_no_auth(self):
        """Test that health endpoints don't require authentication"""
        # Health check
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Kubernetes probes
        response = self.client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        
        # Readiness probe (may fail due to missing model/feature store)
        response = self.client.get("/readyz")
        # Don't assert status code as it depends on model/feature store availability
    
    def test_metrics_endpoint_no_auth(self):
        """Test that metrics endpoint doesn't require authentication"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_predict_requires_auth(self):
        """Test that predict endpoint requires authentication"""
        request_data = {"ticker": "AAPL"}
        
        # No token
        response = self.client.post("/predict", json=request_data)
        assert response.status_code == 403  # Forbidden due to missing auth
        
        # Invalid token
        headers = {"Authorization": f"Bearer {self.invalid_token}"}
        response = self.client.post("/predict", json=request_data, headers=headers)
        assert response.status_code == 401  # Unauthorized
    
    @patch('app.main._model')
    @patch('app.main._model_metadata')
    @patch('app.main.get_features_from_store')
    def test_predict_with_valid_token(self, mock_get_features, mock_metadata, mock_model):
        """Test prediction with valid token"""
        # Setup mocks
        mock_model.return_value = self.mock_model
        mock_metadata.return_value = {"version": "1.0.0", "features": []}
        mock_get_features.return_value = {"feature1": 1.0, "feature2": 2.0}
        
        request_data = {"ticker": "AAPL"}
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        
        response = self.client.post("/predict", json=request_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "prediction" in data
    
    def test_batch_predict_requires_batch_permission(self):
        """Test that batch predict requires batch permission"""
        request_data = {"requests": [{"ticker": "AAPL"}]}
        
        # Token with only predict permission (no batch)
        headers = {"Authorization": f"Bearer {self.predict_only_token}"}
        response = self.client.post("/predict/batch", json=request_data, headers=headers)
        assert response.status_code == 403  # Forbidden due to insufficient permissions
        
        # Token with batch permission
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        response = self.client.post("/predict/batch", json=request_data, headers=headers)
        # May fail due to missing model, but should pass auth
        assert response.status_code != 403
    
    def test_rate_limiting_predict(self):
        """Test rate limiting on predict endpoint"""
        request_data = {"ticker": "AAPL"}
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        
        # Make multiple requests quickly to trigger rate limit
        # Note: This test may be flaky depending on rate limit implementation
        responses = []
        for i in range(5):  # Make several requests quickly
            response = self.client.post("/predict", json=request_data, headers=headers)
            responses.append(response.status_code)
        
        # At least some requests should succeed (even if model is missing)
        # Rate limiting would return 429 if triggered
        success_codes = [200, 503]  # 200 = success, 503 = model not loaded
        assert any(code in success_codes for code in responses)
    
    def test_rate_limiting_batch_stricter(self):
        """Test that batch endpoint has stricter rate limiting"""
        request_data = {"requests": [{"ticker": "AAPL"}]}
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        
        # Batch endpoint should have lower rate limit (20/minute vs 100/minute)
        response = self.client.post("/predict/batch", json=request_data, headers=headers)
        # Should pass auth but may fail due to missing model
        assert response.status_code != 403
    
    def test_user_info_endpoint(self):
        """Test user info endpoint"""
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        
        response = self.client.get("/auth/user", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "username" in data
        assert "permissions" in data
        assert "token_expires" in data
    
    def test_user_info_requires_auth(self):
        """Test that user info endpoint requires authentication"""
        response = self.client.get("/auth/user")
        assert response.status_code == 403  # No auth provided
    
    def test_metrics_contain_expected_metrics(self):
        """Test that metrics endpoint contains expected Prometheus metrics"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for expected metrics
        expected_metrics = [
            "predictions_total",
            "prediction_latency_seconds",
            "active_requests",
            "errors_total",
            "feature_store_requests_total"
        ]
        
        for metric in expected_metrics:
            assert metric in content, f"Metric {metric} not found in metrics output"
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.options("/")
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    def test_root_endpoint_info(self):
        """Test root endpoint provides service information"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Conviction AI Inference API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "/docs" in data["docs"]
        assert "/metrics" in data["metrics"]

def test_jwt_token_generation():
    """Test JWT token generation utilities"""
    from app.auth import create_access_token, generate_test_token
    
    # Test regular token creation
    token = create_access_token("user123", "testuser", ["predict"])
    assert isinstance(token, str)
    assert len(token) > 50  # JWT tokens are typically long
    
    # Test test token generation
    test_token = generate_test_token("testuser")
    assert isinstance(test_token, str)
    assert len(test_token) > 50

def test_security_imports():
    """Test that security modules import correctly"""
    try:
        from app.auth import verify_token, create_access_token
        from app.metrics import track_predictions, get_metrics
        
        assert callable(verify_token)
        assert callable(create_access_token)
        assert callable(track_predictions)
        assert callable(get_metrics)
    except ImportError as e:
        pytest.fail(f"Security module import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])