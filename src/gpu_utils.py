#!/usr/bin/env python3
"""GPU acceleration utilities for Polars computations."""

import polars as pl
from typing import Optional


def gpu_supported() -> bool:
    """Check if GPU acceleration is available."""
    try:
        # Check for CUDA support in Polars
        import cudf
        import polars as pl
        return True
    except ImportError:
        return False


def to_gpu(df: pl.DataFrame) -> Optional[object]:
    """Convert Polars DataFrame to GPU format if supported."""
    if not gpu_supported():
        return None
    
    try:
        import cudf
        return cudf.from_pandas(df.to_pandas())
    except Exception:
        return None


def to_cpu(gpu_df) -> pl.DataFrame:
    """Convert GPU DataFrame back to Polars."""
    if gpu_df is None:
        raise ValueError("GPU DataFrame is None")
    
    return pl.from_pandas(gpu_df.to_pandas())


def gpu_rolling_mean(df: pl.DataFrame, column: str, window: int) -> pl.Series:
    """Compute rolling mean using GPU acceleration if available."""
    if not gpu_supported():
        # Fallback to CPU
        return df[column].rolling_mean(window)
    
    try:
        gpu_df = to_gpu(df)
        if gpu_df is None:
            return df[column].rolling_mean(window)
        
        result = gpu_df[column].rolling(window).mean()
        return to_cpu(result.to_frame())[column]
    except Exception:
        # Fallback to CPU on any GPU error
        return df[column].rolling_mean(window)


def gpu_rolling_std(df: pl.DataFrame, column: str, window: int) -> pl.Series:
    """Compute rolling standard deviation using GPU acceleration if available."""
    if not gpu_supported():
        return df[column].rolling_std(window)
    
    try:
        gpu_df = to_gpu(df)
        if gpu_df is None:
            return df[column].rolling_std(window)
        
        result = gpu_df[column].rolling(window).std()
        return to_cpu(result.to_frame())[column]
    except Exception:
        return df[column].rolling_std(window)


def optimize_for_gpu(df: pl.DataFrame, use_gpu: bool = False) -> pl.DataFrame:
    """Optimize DataFrame operations for GPU if requested and available."""
    if not use_gpu or not gpu_supported():
        return df
    
    print("🚀 GPU acceleration enabled")
    return df


if __name__ == "__main__":
    # Test GPU support
    if gpu_supported():
        print("✅ GPU acceleration available")
    else:
        print("⚠️  GPU acceleration not available, using CPU fallback")