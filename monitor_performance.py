#!/usr/bin/env python3
"""
Performance Monitor for M2 Ultra Pipeline
Monitors CPU, memory, and GPU utilization during pipeline execution.
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil


class PerformanceMonitor:
    """Monitor system performance during pipeline execution."""

    def __init__(self):
        self.monitoring = False
        self.stats = []

    def get_system_stats(self):
        """Get current system performance statistics."""
        # CPU usage per core
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()

        # Process count
        process_count = len(psutil.pids())

        # Load average (macOS specific)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            load_avg = (0, 0, 0)

        return {
            "timestamp": datetime.now(),
            "cpu_cores": cpu_percent,
            "cpu_avg": cpu_avg,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "process_count": process_count,
            "load_avg_1m": load_avg[0],
            "load_avg_5m": load_avg[1],
            "load_avg_15m": load_avg[2],
        }

    def monitor_background(self):
        """Background monitoring thread."""
        while self.monitoring:
            stats = self.get_system_stats()
            self.stats.append(stats)
            time.sleep(5)  # Monitor every 5 seconds

    def start_monitoring(self):
        """Start background monitoring."""
        self.monitoring = True
        self.stats = []
        self.monitor_thread = threading.Thread(target=self.monitor_background)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started...")

    def stop_monitoring(self):
        """Stop monitoring and return stats."""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2)
        print("⏹️  Performance monitoring stopped.")
        return self.stats

    def print_real_time_stats(self):
        """Print real-time performance statistics."""
        stats = self.get_system_stats()

        print(
            f"\n📊 Real-time Performance Stats - {stats['timestamp'].strftime('%H:%M:%S')}"
        )
        print("=" * 60)

        # CPU utilization
        print(f"🔥 CPU Average: {stats['cpu_avg']:.1f}%")
        core_usage = [f"{c:.0f}%" for c in stats["cpu_cores"][:8]]
        print(f"   Core Usage: {core_usage}...")  # Show first 8 cores

        # Memory usage
        print(
            f"💾 Memory: {stats['memory_used_gb']:.1f}GB / {stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%)"
        )

        # Load average
        print(
            f"⚖️  Load Average: {stats['load_avg_1m']:.2f} (1m), {stats['load_avg_5m']:.2f} (5m)"
        )

        # System load
        print(f"📈 Processes: {stats['process_count']}")

    def analyze_stats(self, stats):
        """Analyze collected performance statistics."""
        if not stats:
            print("❌ No performance data collected")
            return

        print(f"\n📈 Performance Analysis ({len(stats)} samples)")
        print("=" * 60)

        # CPU analysis
        cpu_avgs = [s["cpu_avg"] for s in stats]
        cpu_max = max(cpu_avgs)
        cpu_avg = sum(cpu_avgs) / len(cpu_avgs)

        print(f"🔥 CPU Utilization:")
        print(f"   Average: {cpu_avg:.1f}%")
        print(f"   Peak: {cpu_max:.1f}%")
        print(
            f"   Efficiency: {'High' if cpu_avg > 60 else 'Medium' if cpu_avg > 30 else 'Low'}"
        )

        # Memory analysis
        memory_usage = [s["memory_used_gb"] for s in stats]
        memory_max = max(memory_usage)
        memory_avg = sum(memory_usage) / len(memory_usage)

        print(f"\n💾 Memory Usage:")
        print(f"   Average: {memory_avg:.1f}GB")
        print(f"   Peak: {memory_max:.1f}GB")
        print(f"   Peak %: {(memory_max / stats[0]['memory_total_gb']) * 100:.1f}%")

        # Load analysis
        load_avgs = [s["load_avg_1m"] for s in stats]
        load_max = max(load_avgs)
        load_avg = sum(load_avgs) / len(load_avgs)

        print(f"\n⚖️  System Load:")
        print(f"   Average: {load_avg:.2f}")
        print(f"   Peak: {load_max:.2f}")
        print(f"   Utilization: {(load_avg / 24) * 100:.1f}% of 24 cores")

        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        if cpu_avg < 50:
            print(
                "   • CPU utilization could be higher - consider increasing parallelism"
            )
        if load_avg > 20:
            print(
                "   • High system load detected - consider reducing concurrent processes"
            )
        if memory_max > 48:  # > 75% of 64GB
            print("   • High memory usage - consider optimizing memory allocation")
        else:
            print("   • System performance looks optimal for M2 Ultra hardware")


def run_with_monitoring(command, description):
    """Run a command while monitoring performance."""
    monitor = PerformanceMonitor()

    print(f"🚀 Starting: {description}")
    print(f"📝 Command: {command}")
    print("=" * 80)

    # Start monitoring
    monitor.start_monitoring()

    try:
        # Show initial stats
        monitor.print_real_time_stats()

        # Run the command
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()

        # Stop monitoring
        stats = monitor.stop_monitoring()

        # Results
        duration = end_time - start_time
        print(f"\n✅ Command completed in {duration:.2f} seconds")
        print(f"📤 Exit code: {result.returncode}")

        if result.stdout:
            print(f"\n📋 Output:\n{result.stdout}")

        if result.stderr:
            print(f"\n⚠️  Errors:\n{result.stderr}")

        # Performance analysis
        monitor.analyze_stats(stats)

        return result.returncode == 0

    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n⏹️  Monitoring interrupted by user")
        return False
    except Exception as e:
        monitor.stop_monitoring()
        print(f"\n❌ Error during monitoring: {e}")
        return False


def main():
    """Main monitoring interface."""
    if len(sys.argv) < 2:
        print("Usage: python monitor_performance.py <command>")
        print("\nExamples:")
        print("  python monitor_performance.py 'python test_m2_ultra_optimization.py'")
        print(
            "  python monitor_performance.py './scripts/single_day_pipeline_standalone.sh'"
        )
        print(
            "  python monitor_performance.py 'python src/calculate_features.py --help'"
        )
        return

    command = " ".join(sys.argv[1:])
    description = f"M2 Ultra Performance Test"

    print("🍎 M2 Ultra Performance Monitor")
    print("=" * 60)
    print(f"🖥️  Hardware: 24-core M2 Ultra, 64GB RAM")
    print(f"🎯 Target: Full CPU/GPU utilization")
    print("")

    success = run_with_monitoring(command, description)

    if success:
        print("\n🎉 Command completed successfully with performance monitoring!")
    else:
        print("\n❌ Command failed or was interrupted")


if __name__ == "__main__":
    main()
