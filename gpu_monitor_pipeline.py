#!/usr/bin/env python3
"""
GPU-Enhanced Performance Monitor for M2 Ultra Pipeline
Monitors CPU, GPU (Apple Metal), memory, and system utilization during pipeline execution.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, "src")


def check_gpu_usage():
    """Check GPU memory usage on Apple Silicon."""
    try:
        # Use powermetrics to get GPU usage on macOS (requires sudo for detailed info)
        result = subprocess.run(
            ["sudo", "powermetrics", "-n", "1", "-s", "gpu_power"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse GPU power usage
            for line in result.stdout.split('\n'):
                if 'GPU HW active residency' in line:
                    return line.strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback - check for Metal compute units
    try:
        import torch
        if torch.backends.mps.is_available():
            return "🚀 Apple Metal GPU available (detailed metrics require sudo)"
    except ImportError:
        pass
    
    return "❓ GPU status unknown"


def monitor_system_performance():
    """Quick system performance snapshot."""
    try:
        import psutil
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # System load
        load_avg = psutil.getloadavg()
        
        print(f"📊 System Performance Snapshot - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   🔥 CPU: {cpu_percent:.1f}% average")
        print(f"   💾 Memory: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        print(f"   ⚖️  Load: {load_avg[0]:.2f} (1m) - {(load_avg[0] / 24) * 100:.1f}% of 24 cores")
        
        # GPU status
        gpu_status = check_gpu_usage()
        print(f"   🎮 GPU: {gpu_status}")
        
    except ImportError:
        print("⚠️  psutil not available for detailed monitoring")
    except Exception as e:
        print(f"⚠️  Monitoring error: {e}")


def run_with_gpu_monitoring(script_path):
    """Run the standalone pipeline with enhanced GPU monitoring."""
    print("🍎 GPU-Enhanced M2 Ultra Pipeline Monitor")
    print("=" * 60)
    print(f"🖥️  Hardware: 24-core M2 Ultra, 64GB RAM")
    print(f"🎯 Target: GPU + CPU optimization")
    print()
    
    # Initial system check
    monitor_system_performance()
    print()
    
    # Check GPU utilities
    try:
        from gpu_utils import gpu_supported, optimize_for_apple_silicon
        
        optimize_for_apple_silicon()
        if gpu_supported():
            print("✅ GPU acceleration available - Apple Metal enabled")
        else:
            print("⚠️  GPU acceleration not available - using 24-core CPU optimization")
    except ImportError:
        print("⚠️  GPU utilities not available")
    
    print(f"🚀 Starting pipeline: {script_path}")
    print("=" * 60)
    
    # Run the script
    start_time = time.time()
    
    try:
        # Monitor every 30 seconds during execution
        process = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        last_monitor_time = 0
        while True:
            # Check if process is done
            if process.poll() is not None:
                break
            
            # Read any output
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Monitor every 30 seconds
            current_time = time.time()
            if current_time - last_monitor_time >= 30:
                print("\n" + "="*40)
                monitor_system_performance()
                print("="*40 + "\n")
                last_monitor_time = current_time
            
            time.sleep(0.1)
        
        # Get final output
        remaining_output, _ = process.communicate()
        if remaining_output:
            print(remaining_output)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print(f"🎉 Pipeline completed in {total_time:.2f} seconds")
        print(f"📤 Exit code: {process.returncode}")
        
        # Final performance snapshot
        print("\n📈 Final Performance Summary:")
        monitor_system_performance()
        
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        return False
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python gpu_monitor_pipeline.py <script_path>")
        print("Example: python gpu_monitor_pipeline.py scripts/single_day_pipeline_standalone.sh")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)
    
    success = run_with_gpu_monitoring(script_path)
    
    if success:
        print("\n🎉 GPU-enhanced pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed or was interrupted")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
