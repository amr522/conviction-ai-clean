#!/usr/bin/env python3
"""
AWS X-Ray utilities for tracing and monitoring
"""
import os
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional
from aws_xray_sdk.core import xray_recorder

logger = logging.getLogger(__name__)

def configure_xray(service_name: str = 'conviction-ai', daemon_address: str = '127.0.0.1:2000'):
    """
    Configure AWS X-Ray with service-specific settings
    
    Args:
        service_name: Name of the service for X-Ray traces
        daemon_address: Address of the X-Ray daemon
    """
    try:
        # Check if X-Ray is disabled via environment variable
        if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
            logger.info("X-Ray tracing is disabled via environment variable")
            return
        
        xray_recorder.configure(
            service=service_name,
            plugins=('EC2Plugin', 'ECSPlugin'),
            daemon_address=daemon_address,
            use_ssl=False
        )
        
        logger.info(f"X-Ray configured for service: {service_name}")
        
    except Exception as e:
        logger.warning(f"Failed to configure X-Ray: {str(e)}")

def trace_function(name: Optional[str] = None, capture_response: bool = False):
    """
    Decorator to trace function execution with X-Ray
    
    Args:
        name: Custom name for the trace segment
        capture_response: Whether to capture function response in metadata
    """
    def decorator(func: Callable) -> Callable:
        segment_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check if X-Ray is disabled
                if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
                    return func(*args, **kwargs)
                
                with xray_recorder.capture(segment_name) as segment:
                    # Add function metadata
                    segment.put_annotation('function_name', func.__name__)
                    segment.put_annotation('module', func.__module__)
                    
                    # Add arguments metadata (excluding sensitive data)
                    safe_kwargs = {k: v for k, v in kwargs.items() 
                                 if not any(sensitive in k.lower() 
                                          for sensitive in ['password', 'token', 'key', 'secret'])}
                    
                    if safe_kwargs:
                        segment.put_metadata('function_kwargs', safe_kwargs)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Capture response if requested
                    if capture_response and result is not None:
                        # Only capture small responses to avoid payload limits
                        if isinstance(result, (dict, list)) and len(str(result)) < 1000:
                            segment.put_metadata('function_response', result)
                        elif hasattr(result, '__len__'):
                            segment.put_metadata('response_length', len(result))
                    
                    return result
                    
            except Exception as e:
                # Add error information to trace
                if not os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
                    try:
                        xray_recorder.put_annotation('error', True)
                        xray_recorder.put_metadata('error_details', {
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        })
                    except:
                        pass  # Don't let X-Ray errors break the main function
                
                raise
        
        return wrapper
    return decorator

def add_trace_metadata(key: str, value: Any):
    """
    Add metadata to the current X-Ray trace
    
    Args:
        key: Metadata key
        value: Metadata value
    """
    try:
        if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() != 'true':
            xray_recorder.put_metadata(key, value)
    except Exception as e:
        logger.debug(f"Failed to add X-Ray metadata: {str(e)}")

def add_trace_annotation(key: str, value: str):
    """
    Add annotation to the current X-Ray trace
    
    Args:
        key: Annotation key
        value: Annotation value (must be string, number, or boolean)
    """
    try:
        if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() != 'true':
            xray_recorder.put_annotation(key, value)
    except Exception as e:
        logger.debug(f"Failed to add X-Ray annotation: {str(e)}")

def trace_data_processing(dates: list, process_func: Callable, *args, **kwargs):
    """
    Trace data processing across multiple dates with subsegments
    
    Args:
        dates: List of dates to process
        process_func: Function to call for each date
        *args, **kwargs: Arguments to pass to process_func
    """
    results = []
    
    for date in dates:
        subsegment_name = f'process_date_{date}'
        
        try:
            if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
                result = process_func(date, *args, **kwargs)
                results.append(result)
                continue
            
            subsegment = xray_recorder.begin_subsegment(subsegment_name)
            try:
                xray_recorder.put_annotation('processing_date', date)
                xray_recorder.put_annotation('batch_position', len(results) + 1)
                xray_recorder.put_annotation('total_dates', len(dates))
                
                result = process_func(date, *args, **kwargs)
                
                # Add result metadata
                if isinstance(result, dict) and 'rows_processed' in result:
                    xray_recorder.put_annotation('rows_processed', result['rows_processed'])
                
                results.append(result)
                
            finally:
                xray_recorder.end_subsegment()
                
        except Exception as e:
            logger.error(f"Error processing date {date}: {str(e)}")
            if not os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
                try:
                    xray_recorder.put_annotation('processing_error', True)
                    xray_recorder.put_metadata('error_date', date)
                except:
                    pass
            raise
    
    return results

class XRayContextManager:
    """Context manager for X-Ray subsegments"""
    
    def __init__(self, name: str, annotations: Optional[Dict] = None, metadata: Optional[Dict] = None):
        self.name = name
        self.annotations = annotations or {}
        self.metadata = metadata or {}
        self.subsegment = None
    
    def __enter__(self):
        if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
            return self
        
        try:
            self.subsegment = xray_recorder.begin_subsegment(self.name)
            
            # Add annotations
            for key, value in self.annotations.items():
                xray_recorder.put_annotation(key, value)
            
            # Add metadata
            for key, value in self.metadata.items():
                xray_recorder.put_metadata(key, value)
                
        except Exception as e:
            logger.debug(f"Failed to start X-Ray subsegment: {str(e)}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.getenv('AWS_XRAY_TRACING_DISABLED', 'false').lower() == 'true':
            return
        
        try:
            if exc_type is not None:
                xray_recorder.put_annotation('error', True)
                xray_recorder.put_metadata('exception', {
                    'type': exc_type.__name__,
                    'message': str(exc_val)
                })
            
            if self.subsegment:
                xray_recorder.end_subsegment()
                
        except Exception as e:
            logger.debug(f"Failed to end X-Ray subsegment: {str(e)}")

# Convenience function for creating traced subsegments
def traced_subsegment(name: str, annotations: Optional[Dict] = None, metadata: Optional[Dict] = None):
    """
    Create a traced subsegment context manager
    
    Args:
        name: Subsegment name
        annotations: Annotations to add
        metadata: Metadata to add
    
    Returns:
        XRayContextManager instance
    """
    return XRayContextManager(name, annotations, metadata)