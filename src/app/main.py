#!/usr/bin/env python3
"""
FastAPI microservice for live ML predictions with GPU support and feature store integration
"""
# Initialize Sentry before other imports
import logging
import os

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# Configure Sentry
sentry_logging = LoggingIntegration(
    level=logging.INFO,  # Capture info and above as breadcrumbs
    event_level=logging.ERROR,  # Send errors as events
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    integrations=[
        FastApiIntegration(),
        sentry_logging,
    ],
    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
    profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
    environment=os.getenv("ENVIRONMENT", "production"),
    release=os.getenv("RELEASE", "unknown"),
    send_default_pii=False,  # Don't send personally identifiable information
    attach_stacktrace=True,
    max_breadcrumbs=50,
)

import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import uvicorn
# X-Ray tracing for FastAPI
from aws_xray_sdk.core import patch_all, xray_recorder
from aws_xray_sdk.ext.fastapi import XRayMiddleware
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
# Sentry imports
from sentry_sdk import (capture_exception, capture_message, set_context,
                        set_tag, start_span)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Authentication and metrics
from app.auth import (TokenData, get_user_info, verify_batch_permission,
                      verify_predict_permission, verify_token)
from app.metrics import (get_metrics, metrics_middleware,
                         track_batch_predictions, track_feature_store_request,
                         track_predictions, update_model_info)

# Patch AWS services
patch_all()
xray_recorder.configure(
    service="conviction-ai-inference",
    plugins=("EC2Plugin", "ECSPlugin"),
    daemon_address="127.0.0.1:2000",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Conviction AI Inference API",
    description="Real-time ML predictions for volatility trading",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add X-Ray middleware
app.add_middleware(XRayMiddleware, app, tracing_name="conviction-ai-inference")

# Add rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# Add metrics middleware
app.middleware("http")(metrics_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
_model = None
_model_metadata = {}


class InferenceRequest(BaseModel):
    """Request model for predictions"""

    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp for point-in-time features",
        example="2025-01-16T15:30:00Z",
    )
    features: Optional[Dict[str, Union[float, int, bool]]] = Field(
        default=None, description="Optional manual feature override"
    )

    @validator("ticker")
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError("Ticker must be 1-10 characters")
        return v.upper()

    @validator("timestamp")
    def validate_timestamp(cls, v):
        if v:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("Invalid ISO8601 timestamp format")
        return v


class InferenceResponse(BaseModel):
    """Response model for predictions"""

    ticker: str
    prediction: float
    confidence: Optional[float] = None
    features_used: Dict[str, Union[float, int, bool]]
    model_version: str
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_version: str
    gpu_available: bool
    feature_store_connected: bool


class BatchInferenceRequest(BaseModel):
    """Batch prediction request"""

    requests: List[InferenceRequest] = Field(..., max_items=100)


class BatchInferenceResponse(BaseModel):
    """Batch prediction response"""

    predictions: List[InferenceResponse]
    total_requests: int
    successful_predictions: int
    failed_predictions: int


def load_model():
    """Load the trained model"""
    global _model, _model_metadata

    with start_span(op="model_loading", description="Load ML model from disk"):
        try:
            model_path = os.getenv("MODEL_PATH", "models/latest.pkl")

            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}"
                logger.error(error_msg)
                capture_message(error_msg, level="error")
                return False

            with open(model_path, "rb") as f:
                _model = pickle.load(f)

            # Load model metadata if available
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                import json

                with open(metadata_path, "r") as f:
                    _model_metadata = json.load(f)
            else:
                _model_metadata = {
                    "version": "1.0.0",
                    "created_at": datetime.now().isoformat(),
                    "features": [],
                }

            # Set Sentry context
            set_context(
                "model",
                {
                    "path": model_path,
                    "version": _model_metadata.get("version", "unknown"),
                    "type": type(_model).__name__,
                    "features_count": len(_model_metadata.get("features", [])),
                },
            )

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            capture_exception(e)
            return False


def get_features_from_store(ticker: str, timestamp: Optional[str] = None) -> Dict:
    """Fetch features from Feast feature store"""
    with start_span(
        op="feature_fetch", description="Fetch features from Feast store"
    ) as span:
        try:
            # Set span data
            span.set_data("ticker", ticker)
            span.set_data("timestamp", timestamp or "current")

            # Import here to avoid startup dependency
            from feast_materialize import get_online_features

            # Use current time if no timestamp provided
            if not timestamp:
                timestamp = datetime.now().isoformat()

            # Define feature names to fetch
            feature_names = [
                "stocks_30min:close",
                "stocks_30min:volume",
                "stocks_30min:returns",
                "stocks_30min:volatility",
                "options_30min:opt30_close",
                "options_30min:opt30_volume",
                "options_30min:opt30_gamma_squeeze",
                "options_30min:opt30_implied_volatility",
                "stocks_daily:close",
                "stocks_daily:rsi_14",
                "options_daily:optd_iv30",
                "options_daily:optd_vrp_30d",
            ]

            # Fetch features from online store
            features_dict = get_online_features(
                entity_rows=[{"ticker": ticker}], feature_names=feature_names
            )

            if features_dict and len(features_dict.get("ticker", [])) > 0:
                # Convert to flat dictionary
                features = {}
                for key, values in features_dict.items():
                    if key != "ticker" and len(values) > 0:
                        features[key] = values[0]

                return features
            else:
                warning_msg = f"No features found for ticker {ticker}"
                logger.warning(warning_msg)
                capture_message(warning_msg, level="warning")
                return {}

        except Exception as e:
            logger.error(f"Failed to fetch features from store: {str(e)}")
            capture_exception(e)
            return {}


def check_gpu_availability() -> bool:
    """Check if GPU is available"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def check_feature_store_connection() -> bool:
    """Check if feature store is accessible"""
    try:
        from feast_materialize import get_feature_store

        fs = get_feature_store()
        return fs is not None
    except Exception:
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("Starting Conviction AI Inference API...")

    # Set Sentry tags for this instance
    set_tag("service", "conviction-ai-inference")
    set_tag("version", "1.0.0")
    set_tag("environment", os.getenv("ENVIRONMENT", "production"))

    # Load model
    if not load_model():
        error_msg = "Failed to load model on startup"
        logger.error(error_msg)
        capture_message(error_msg, level="error")

    # Check GPU availability
    gpu_available = check_gpu_availability()
    logger.info(f"GPU available: {gpu_available}")
    set_tag("gpu_available", str(gpu_available))

    # Check feature store connection
    fs_connected = check_feature_store_connection()
    logger.info(f"Feature store connected: {fs_connected}")
    set_tag("feature_store_connected", str(fs_connected))

    # Update model info metrics
    if _model is not None:
        update_model_info(
            model_type=type(_model).__name__,
            version=_model_metadata.get("version", "1.0.0"),
            features_count=len(_model_metadata.get("features", [])),
        )

    capture_message("Conviction AI Inference API started successfully", level="info")


# Add global exception handler for Sentry
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with Sentry integration"""
    # Capture exception in Sentry
    capture_exception(exc)

    # Log the error
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    # Return generic error response
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
@xray_recorder.capture("health_check")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if _model is not None else "unhealthy",
        model_loaded=_model is not None,
        model_version=_model_metadata.get("version", "unknown"),
        gpu_available=check_gpu_availability(),
        feature_store_connected=check_feature_store_connection(),
    )


@app.post(
    "/predict",
    response_model=InferenceResponse,
    dependencies=[Depends(verify_predict_permission)],
)
@limiter.limit("100/minute")
@xray_recorder.capture("single_prediction")
@track_predictions
async def predict(
    request: Request,
    inference_request: InferenceRequest,
    token_data: TokenData = Depends(verify_predict_permission),
):
    """Single prediction endpoint"""
    start_time = datetime.now()

    # Check if model is loaded
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Add trace annotations
        xray_recorder.put_annotation("ticker", request.ticker)
        xray_recorder.put_annotation(
            "has_manual_features", request.features is not None
        )

        # Get features
        if inference_request.features:
            # Use provided features
            features = inference_request.features
            xray_recorder.put_annotation("feature_source", "manual")
            track_feature_store_request(True)  # Manual features always "succeed"
        else:
            # Fetch from feature store
            try:
                features = get_features_from_store(
                    inference_request.ticker, inference_request.timestamp
                )
                xray_recorder.put_annotation("feature_source", "feast")
                track_feature_store_request(True)
            except Exception as e:
                track_feature_store_request(False)
                raise

            if not features:
                track_feature_store_request(False)
                raise HTTPException(
                    status_code=404,
                    detail=f"No features found for ticker {inference_request.ticker}",
                )

        # Add feature metadata to trace
        xray_recorder.put_metadata("features_count", len(features))
        xray_recorder.put_metadata("feature_names", list(features.keys()))

        # Prepare input DataFrame
        df = pd.DataFrame([features])

        # Handle missing features with defaults
        expected_features = _model_metadata.get("features", [])
        if expected_features:
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0  # Default value for missing features

            # Reorder columns to match training
            df = df.reindex(columns=expected_features, fill_value=0.0)

        # Run prediction
        with xray_recorder.capture("model_inference"):
            prediction = _model.predict(df)

            # Handle different prediction formats
            if hasattr(prediction, "__len__") and len(prediction) > 0:
                pred_value = float(prediction[0])
            else:
                pred_value = float(prediction)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Add performance metadata
        xray_recorder.put_annotation("processing_time_ms", processing_time)
        xray_recorder.put_annotation("prediction_value", pred_value)

        return InferenceResponse(
            ticker=inference_request.ticker,
            prediction=pred_value,
            confidence=None,  # Could add confidence intervals if model supports it
            features_used=features,
            model_version=_model_metadata.get("version", "1.0.0"),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {inference_request.ticker}: {str(e)}")
        xray_recorder.put_annotation("error", True)
        xray_recorder.put_metadata("error_details", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchInferenceResponse,
    dependencies=[Depends(verify_batch_permission)],
)
@limiter.limit("20/minute")
@xray_recorder.capture("batch_prediction")
@track_batch_predictions
async def predict_internal(inference_request: InferenceRequest) -> InferenceResponse:
    """Internal prediction function without auth"""
    start_time = datetime.now()

    # Set Sentry context for this prediction
    set_context(
        "prediction_request",
        {
            "ticker": inference_request.ticker,
            "has_manual_features": inference_request.features is not None,
            "timestamp": inference_request.timestamp,
        },
    )

    with start_span(
        op="prediction", description="Complete prediction workflow"
    ) as span:
        span.set_data("ticker", inference_request.ticker)

        # Get features
        if inference_request.features:
            features = inference_request.features
            track_feature_store_request(True)
            span.set_data("feature_source", "manual")
        else:
            try:
                features = get_features_from_store(
                    inference_request.ticker, inference_request.timestamp
                )
                track_feature_store_request(True)
                span.set_data("feature_source", "feast")
            except Exception as e:
                track_feature_store_request(False)
                capture_exception(e)
                raise

            if not features:
                track_feature_store_request(False)
                error_msg = f"No features found for ticker {inference_request.ticker}"
                capture_message(error_msg, level="error")
                raise HTTPException(status_code=404, detail=error_msg)

        span.set_data("features_count", len(features))

        # Prepare and run prediction
        with start_span(
            op="inference", description="Run model prediction"
        ) as inference_span:
            df = pd.DataFrame([features])
            expected_features = _model_metadata.get("features", [])
            if expected_features:
                for feature in expected_features:
                    if feature not in df.columns:
                        df[feature] = 0.0
                df = df.reindex(columns=expected_features, fill_value=0.0)

            inference_span.set_data("input_shape", df.shape)

            if _model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            prediction = _model.predict(df)
            pred_value = (
                float(prediction[0])
                if hasattr(prediction, "__len__")
                else float(prediction)
            )

            inference_span.set_data("prediction_value", pred_value)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        span.set_data("processing_time_ms", processing_time)

        return InferenceResponse(
            ticker=inference_request.ticker,
            prediction=pred_value,
            features_used=features,
            model_version=_model_metadata.get("version", "1.0.0"),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
        )


async def predict_batch(
    request: Request,
    batch_request: BatchInferenceRequest,
    token_data: TokenData = Depends(verify_batch_permission),
):
    """Batch prediction endpoint"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    successful = 0
    failed = 0

    xray_recorder.put_annotation("batch_size", len(batch_request.requests))

    for req in batch_request.requests:
        try:
            pred_response = await predict_internal(req)
            predictions.append(pred_response)
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction failed for {req.ticker}: {str(e)}")
            capture_exception(e)
            failed += 1
            predictions.append(
                InferenceResponse(
                    ticker=req.ticker,
                    prediction=0.0,
                    features_used={},
                    model_version=_model_metadata.get("version", "1.0.0"),
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0.0,
                )
            )

    xray_recorder.put_annotation("successful_predictions", successful)
    xray_recorder.put_annotation("failed_predictions", failed)

    return BatchInferenceResponse(
        predictions=predictions,
        total_requests=len(batch_request.requests),
        successful_predictions=successful,
        failed_predictions=failed,
    )


@app.post("/model/reload")
@xray_recorder.capture("model_reload")
async def reload_model():
    """Reload the model"""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_loaded": True,
        "model_type": type(_model).__name__,
        "metadata": _model_metadata,
        "feature_count": len(_model_metadata.get("features", [])),
        "gpu_available": check_gpu_availability(),
    }


# Health and monitoring endpoints
@app.get("/healthz")
async def healthz():
    """Kubernetes liveness probe"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Kubernetes readiness probe"""
    model_ok = _model is not None
    fs_ok = check_feature_store_connection()

    if model_ok and fs_ok:
        return {"status": "ready", "model": "ok", "feature_store": "ok"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()


@app.get("/auth/user")
async def get_current_user(user_info: dict = Depends(get_user_info)):
    """Get current user information"""
    return user_info


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Conviction AI Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    # For local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
