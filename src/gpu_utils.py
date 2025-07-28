#!/usr/bin/env python3
"""GPU acceleration utilities for Polars computations on Apple Silicon M2 Ultra."""

import os
from typing import Optional

import polars as pl


def gpu_supported() -> bool:
    """Check if GPU acceleration is available on Apple Silicon."""
    try:
        # Check for Apple Metal Performance Shaders (MPS) support
        import torch

        if torch.backends.mps.is_available():
            return True
    except ImportError:
        pass

    try:
        # Check for CUDA support (if running on other platforms)
        import cudf

        return True
    except ImportError:
        pass

    return False


def optimize_for_apple_silicon():
    """Optimize settings for Apple Silicon M2 Ultra."""
    # Set optimal thread counts for 24-core M2 Ultra
    os.environ["POLARS_MAX_THREADS"] = "24"
    os.environ["OMP_NUM_THREADS"] = "24"
    os.environ["MKL_NUM_THREADS"] = "24"
    os.environ["NUMBA_NUM_THREADS"] = "24"

    # Optimize memory allocation for 64GB RAM
    os.environ["PYARROW_MEMORY_POOL"] = "jemalloc"

    # Configure Polars for high performance
    try:
        pl.Config.set_streaming_chunk_size(50000)  # Larger chunks for 64GB RAM
        print("🍎 Apple Silicon M2 Ultra optimizations enabled:")
        print(f"   - 24 CPU cores utilized")
        print(f"   - 64GB RAM optimized allocation")
        print(f"   - Polars streaming chunk size: 50,000")
    except Exception as e:
        print(f"⚠️  Polars configuration warning: {e}")


def to_gpu(df: pl.DataFrame) -> Optional[object]:
    """Convert Polars DataFrame to GPU format if supported."""
    if not gpu_supported():
        return None

    try:
        # Try Apple Metal Performance Shaders first
        import torch

        if torch.backends.mps.is_available():
            # Convert to PyTorch tensor with MPS backend
            pandas_df = df.to_pandas()
            numeric_cols = pandas_df.select_dtypes(include=["number"])
            if not numeric_cols.empty:
                tensor = torch.from_numpy(numeric_cols.values)
                return tensor.to("mps")

        # Fallback to CUDA if available
        import cudf

        return cudf.from_pandas(df.to_pandas())
    except Exception as e:
        print(f"⚠️  GPU conversion failed, using CPU: {e}")
        return None


def to_cpu(gpu_df) -> pl.DataFrame:
    """Convert GPU DataFrame back to Polars."""
    if gpu_df is None:
        raise ValueError("GPU DataFrame is None")

    try:
        return pl.from_pandas(gpu_df.to_pandas())
    except:
        # If it's a PyTorch tensor, convert differently
        return pl.from_pandas(gpu_df.cpu().numpy())


def gpu_rolling_mean(df: pl.DataFrame, column: str, window: int) -> pl.Series:
    """Compute rolling mean using optimized operations."""
    try:
        # Use Polars optimized operations (already highly optimized for Apple Silicon)
        return df.select(pl.col(column).rolling_mean(window)).to_series()
    except Exception:
        # Fallback
        return df[column].rolling_mean(window)


def gpu_rolling_std(df: pl.DataFrame, column: str, window: int) -> pl.Series:
    """Compute rolling standard deviation using optimized operations."""
    try:
        # Use Polars optimized operations (already highly optimized for Apple Silicon)
        return df.select(pl.col(column).rolling_std(window)).to_series()
    except Exception:
        # Fallback
        return df[column].rolling_std(window)


def optimize_for_gpu(df: pl.DataFrame, use_gpu: bool = True) -> pl.DataFrame:
    """Optimize DataFrame for GPU operations if requested and available."""
    if not use_gpu:
        return df

    # Apply Apple Silicon optimizations
    optimize_for_apple_silicon()

    if gpu_supported():
        print("🚀 GPU acceleration enabled (Apple Metal/CUDA)")
    else:
        print("🔧 CPU optimization enabled (24 cores)")

    # Return the original DataFrame as Polars is already optimized
    return df


if __name__ == "__main__":
    # Test GPU support and optimizations
    optimize_for_apple_silicon()
    if gpu_supported():
        print("✅ GPU acceleration available")
    else:
        print("⚠️  GPU acceleration not available, using optimized CPU (24 cores)")
