#!/usr/bin/env python3
"""
Prometheus metrics for FastAPI inference service
"""
import time
import logging
from typing import Callable
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['method', 'endpoint', 'status_code', 'user_id']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing predictions',
    ['method', 'endpoint', 'model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests being processed'
)

MODEL_INFO = Info(
    'model_info',
    'Information about the loaded model'
)

FEATURE_STORE_REQUESTS = Counter(
    'feature_store_requests_total',
    'Total requests to feature store',
    ['status']
)

BATCH_SIZE = Histogram(
    'batch_prediction_size',
    'Size of batch prediction requests',
    buckets=[1, 5, 10, 25, 50, 100]
)

ERROR_COUNTER = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type']
)

def track_predictions(func: Callable) -> Callable:
    """
    Decorator to track prediction metrics
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with metrics tracking
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track successful prediction
            PREDICTIONS_TOTAL.labels(
                method="POST",
                endpoint="/predict",
                status_code="200",
                user_id=kwargs.get('user_id', 'unknown')
            ).inc()
            
            return result
            
        except Exception as e:
            # Track error
            ERROR_COUNTER.labels(
                error_type=type(e).__name__,
                endpoint="/predict"
            ).inc()
            raise
            
        finally:
            # Track latency
            duration = time.time() - start_time
            PREDICTION_LATENCY.labels(
                method="POST",
                endpoint="/predict",
                model_version=kwargs.get('model_version', 'unknown')
            ).observe(duration)
            
            ACTIVE_REQUESTS.dec()
    
    return wrapper

def track_batch_predictions(func: Callable) -> Callable:
    """
    Decorator to track batch prediction metrics
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with batch metrics tracking
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track batch size
            if hasattr(result, 'total_requests'):
                BATCH_SIZE.observe(result.total_requests)
            
            # Track successful batch prediction
            PREDICTIONS_TOTAL.labels(
                method="POST",
                endpoint="/predict/batch",
                status_code="200",
                user_id=kwargs.get('user_id', 'unknown')
            ).inc()
            
            return result
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type=type(e).__name__,
                endpoint="/predict/batch"
            ).inc()
            raise
            
        finally:
            duration = time.time() - start_time
            PREDICTION_LATENCY.labels(
                method="POST",
                endpoint="/predict/batch",
                model_version=kwargs.get('model_version', 'unknown')
            ).observe(duration)
            
            ACTIVE_REQUESTS.dec()
    
    return wrapper

def track_feature_store_request(success: bool):
    """
    Track feature store request
    
    Args:
        success: Whether the request was successful
    """
    status = "success" if success else "error"
    FEATURE_STORE_REQUESTS.labels(status=status).inc()

def update_model_info(model_type: str, version: str, features_count: int):
    """
    Update model information metrics
    
    Args:
        model_type: Type of the model
        version: Model version
        features_count: Number of features
    """
    MODEL_INFO.info({
        'model_type': model_type,
        'version': version,
        'features_count': str(features_count),
        'updated_at': str(int(time.time()))
    })

def update_gpu_metrics():
    """Update GPU utilization metrics if available"""
    try:
        import torch
        if torch.cuda.is_available():
            # Get GPU utilization
            gpu_util = torch.cuda.utilization()
            GPU_UTILIZATION.set(gpu_util)
            
            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            MEMORY_USAGE.labels(type="gpu_allocated").set(memory_allocated)
            MEMORY_USAGE.labels(type="gpu_reserved").set(memory_reserved)
            
    except ImportError:
        pass  # PyTorch not available
    except Exception as e:
        logger.debug(f"Failed to update GPU metrics: {str(e)}")

def update_system_metrics():
    """Update system memory metrics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        MEMORY_USAGE.labels(type="rss").set(memory_info.rss)
        MEMORY_USAGE.labels(type="vms").set(memory_info.vms)
        
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        logger.debug(f"Failed to update system metrics: {str(e)}")

async def metrics_middleware(request: Request, call_next):
    """
    Middleware to collect HTTP metrics
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint
        
    Returns:
        Response with metrics collected
    """
    # Skip metrics collection for metrics endpoint
    if request.url.path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    # Get user ID from request if available
    user_id = "anonymous"
    if hasattr(request.state, 'user') and request.state.user:
        user_id = request.state.user.get('user_id', 'unknown')
    
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        
        # Track request
        PREDICTIONS_TOTAL.labels(
            method=method,
            endpoint=path,
            status_code=status_code,
            user_id=user_id
        ).inc()
        
        return response
        
    except Exception as e:
        # Track error
        ERROR_COUNTER.labels(
            error_type=type(e).__name__,
            endpoint=path
        ).inc()
        raise
        
    finally:
        # Track latency
        duration = time.time() - start_time
        PREDICTION_LATENCY.labels(
            method=method,
            endpoint=path,
            model_version="unknown"
        ).observe(duration)
        
        ACTIVE_REQUESTS.dec()
        
        # Update system metrics periodically
        if int(time.time()) % 30 == 0:  # Every 30 seconds
            update_gpu_metrics()
            update_system_metrics()

def get_metrics() -> FastAPIResponse:
    """
    Get Prometheus metrics in text format
    
    Returns:
        Response with metrics data
    """
    # Update metrics before returning
    update_gpu_metrics()
    update_system_metrics()
    
    metrics_data = generate_latest()
    return FastAPIResponse(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )