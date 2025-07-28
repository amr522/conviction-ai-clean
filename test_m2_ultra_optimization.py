#!/usr/bin/env python3
"""
M2 Ultra Optimization Test Script
Tests GPU and CPU optimization settings for the pipeline.
"""

import os
import sys
import time
from pathlib import Path

import polars as pl
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gpu_utils import gpu_supported, optimize_for_apple_silicon


def test_hardware_detection():
    """Test hardware detection and optimization."""
    print("🔍 Hardware Detection Test")
    print("=" * 50)

    # CPU cores
    cpu_count = os.cpu_count()
    print(f"CPU Cores Available: {cpu_count}")

    # Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"Total RAM: {memory_gb:.1f} GB")

    # Apple Silicon detection
    try:
        import platform

        if platform.processor() == "arm":
            print("✅ Apple Silicon detected")
        else:
            print("⚠️  Non-Apple Silicon detected")
    except:
        print("❓ Could not detect processor type")

    # GPU support
    if gpu_supported():
        print("✅ GPU acceleration available")
    else:
        print("⚠️  GPU acceleration not available")

    print()


def test_polars_optimization():
    """Test Polars optimization settings."""
    print("⚡ Polars Optimization Test")
    print("=" * 50)

    # Apply optimizations
    optimize_for_apple_silicon()

    # Create test data
    n_rows = 1000000
    print(f"Creating test dataset with {n_rows:,} rows...")

    start_time = time.time()

    # Create base arrays
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"] * (n_rows // 5)
    prices = list(range(n_rows))
    volumes = list(range(1000, n_rows + 1000))

    # Create date range and sample it properly
    date_range = pl.date_range(
        start=pl.date(2023, 1, 1), end=pl.date(2025, 12, 31), interval="1d", eager=True
    )
    dates = date_range.sample(n_rows, with_replacement=True).sort()

    df = pl.DataFrame(
        {"ticker": tickers, "price": prices, "volume": volumes, "date": dates.to_list()}
    )

    creation_time = time.time() - start_time
    print(f"✅ Dataset created in {creation_time:.2f} seconds")

    # Test rolling operations (CPU intensive)
    print("Testing rolling operations...")
    start_time = time.time()

    result = df.group_by("ticker").map_groups(
        lambda group: group.with_columns(
            [
                pl.col("price").rolling_mean(30).alias("price_ma30"),
                pl.col("price").rolling_std(30).alias("price_std30"),
                pl.col("volume").rolling_mean(30).alias("volume_ma30"),
            ]
        )
    )

    rolling_time = time.time() - start_time
    print(f"✅ Rolling operations completed in {rolling_time:.2f} seconds")
    print(f"📊 Result shape: {result.shape}")

    # Performance summary
    total_ops_per_sec = (n_rows * 3) / rolling_time  # 3 rolling operations
    print(f"🚀 Performance: {total_ops_per_sec:,.0f} operations/second")

    print()


def test_memory_usage():
    """Test memory usage optimization."""
    print("💾 Memory Usage Test")
    print("=" * 50)

    process = psutil.Process()

    # Before optimization
    memory_before = process.memory_info().rss / (1024**2)  # MB
    print(f"Memory before optimization: {memory_before:.1f} MB")

    # Apply optimizations
    optimize_for_apple_silicon()

    # Create large dataset
    n_rows = 5000000
    print(f"Creating large dataset with {n_rows:,} rows...")

    # Create base data first
    base_data = pl.arange(0, n_rows, eager=True)

    df = pl.DataFrame(
        {
            "col1": base_data.cast(pl.Float64),
            "col2": (base_data * 2).cast(pl.Float64),
            "col3": (base_data * 3).cast(pl.Float64),
            "col4": (base_data * 4).cast(pl.Float64),
        }
    )

    memory_after = process.memory_info().rss / (1024**2)  # MB
    print(f"Memory after dataset creation: {memory_after:.1f} MB")
    print(f"Dataset memory usage: {memory_after - memory_before:.1f} MB")

    # Memory efficiency
    expected_size = (n_rows * 4 * 8) / (1024**2)  # 4 columns * 8 bytes per float64
    efficiency = expected_size / (memory_after - memory_before) * 100
    print(f"Memory efficiency: {efficiency:.1f}%")

    print()


def test_environment_variables():
    """Test environment variable settings."""
    print("🌍 Environment Variables Test")
    print("=" * 50)

    optimize_for_apple_silicon()

    expected_vars = {
        "POLARS_MAX_THREADS": "24",
        "OMP_NUM_THREADS": "24",
        "MKL_NUM_THREADS": "24",
        "NUMBA_NUM_THREADS": "24",
        "PYARROW_MEMORY_POOL": "jemalloc",
    }

    for var, expected in expected_vars.items():
        actual = os.environ.get(var, "NOT SET")
        status = "✅" if actual == expected else "❌"
        print(f"{status} {var}: {actual}")

    print()


def main():
    """Run all optimization tests."""
    print("🍎 M2 Ultra Pipeline Optimization Test Suite")
    print("=" * 60)
    print()

    test_hardware_detection()
    test_environment_variables()
    test_memory_usage()
    test_polars_optimization()

    print("🎉 Optimization test suite completed!")
    print("💡 If all tests pass, your M2 Ultra is optimally configured.")


if __name__ == "__main__":
    main()
