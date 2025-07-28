#!/usr/bin/env python3
"""
Quick GPU acceleration test for M2 Ultra pipeline.
Tests Apple Metal GPU capabilities and performance.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from gpu_utils import gpu_supported, optimize_for_apple_silicon


def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    print("🍎 M2 Ultra GPU Acceleration Test")
    print("=" * 50)
    
    # Apply optimizations
    optimize_for_apple_silicon()
    
    # Check GPU support
    gpu_available = gpu_supported()
    print(f"🎮 GPU Available: {'✅ Yes' if gpu_available else '❌ No'}")
    
    if gpu_available:
        try:
            import torch
            if torch.backends.mps.is_available():
                print("🚀 Apple Metal Performance Shaders (MPS): Available")
                
                # Test basic Metal operations
                device = torch.device("mps")
                print(f"📱 Metal Device: {device}")
                
                # Create a test tensor on GPU
                test_tensor = torch.randn(1000, 1000, device=device)
                print(f"🧮 Created 1000x1000 tensor on Metal GPU")
                
                # Perform some operations
                start_time = time.time()
                result = torch.matmul(test_tensor, test_tensor.T)
                gpu_time = time.time() - start_time
                print(f"⚡ Matrix multiplication on GPU: {gpu_time:.4f} seconds")
                
                # Compare with CPU
                cpu_tensor = torch.randn(1000, 1000)
                start_time = time.time()
                cpu_result = torch.matmul(cpu_tensor, cpu_tensor.T)
                cpu_time = time.time() - start_time
                print(f"🖥️  Matrix multiplication on CPU: {cpu_time:.4f} seconds")
                
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU Speedup: {speedup:.2f}x faster")
                
            else:
                print("⚠️  MPS not available on this system")
        except ImportError:
            print("⚠️  PyTorch not available for Metal testing")
    
    print()
    
    # Test Polars performance with optimization
    print("⚡ Polars Performance Test (Optimized for M2 Ultra)")
    print("-" * 50)
    
    # Create test dataset
    n_rows = 1_000_000
    print(f"Creating dataset with {n_rows:,} rows...")
    
    start_time = time.time()
    df = pl.DataFrame({
        'price': range(n_rows),
        'volume': [x * 2 for x in range(n_rows)],
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * (n_rows // 5),
    })
    creation_time = time.time() - start_time
    print(f"✅ Dataset created in {creation_time:.4f} seconds")
    
    # Test rolling operations (what we use in the pipeline)
    print("Testing rolling calculations...")
    start_time = time.time()
    
    result_df = df.with_columns([
        pl.col('price').rolling_mean(30).alias('price_ma30'),
        pl.col('price').rolling_std(30).alias('price_std30'),
        pl.col('volume').rolling_mean(30).alias('volume_ma30'),
    ])
    
    rolling_time = time.time() - start_time
    operations_per_second = (n_rows * 3) / rolling_time
    
    print(f"✅ Rolling calculations completed in {rolling_time:.4f} seconds")
    print(f"🚀 Performance: {operations_per_second:,.0f} operations/second")
    print(f"📊 Result shape: {result_df.shape}")
    
    # Memory efficiency test
    print(f"💾 Memory usage: ~{result_df.estimated_size('mb'):.1f} MB")
    
    print()
    print("🎉 GPU acceleration test completed!")
    
    return gpu_available


if __name__ == "__main__":
    success = test_gpu_acceleration()
    if success:
        print("💡 Your M2 Ultra is ready for GPU-accelerated pipeline execution!")
    else:
        print("💡 Your M2 Ultra will use optimized 24-core CPU processing.")
