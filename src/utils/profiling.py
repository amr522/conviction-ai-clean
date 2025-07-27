"""Performance profiling utilities for the conviction-ai pipeline."""

import functools
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
from memory_profiler import profile as mem_profile

# Global profiling state
PROFILING_ENABLED = False
PROFILE_LINES = False
PROFILE_RESULTS = []


def enable_profiling(profile_lines=False):
    """Enable profiling globally."""
    global PROFILING_ENABLED, PROFILE_LINES
    PROFILING_ENABLED = True
    PROFILE_LINES = profile_lines


def profile_time(func):
    """Decorator for timing function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        profile_entry = {
            "function": func.__name__,
            "duration_seconds": duration,
            "memory_start_mb": start_memory,
            "memory_end_mb": end_memory,
            "memory_delta_mb": memory_delta,
            "timestamp": datetime.now().isoformat(),
        }

        PROFILE_RESULTS.append(profile_entry)
        print(
            f"[PROFILE] {func.__name__} took {duration:.3f}s, memory: {memory_delta:+.1f}MB"
        )

        return result

    return wrapper


def profile_memory_and_time(func):
    """Combined memory and time profiling decorator."""
    if PROFILING_ENABLED:
        return profile_time(mem_profile(func))
    return func


def save_profile_report(date: str):
    """Save profiling results to a report file."""
    if not PROFILE_RESULTS:
        return

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    report_path = logs_dir / f"profile_{date}.txt"

    with open(report_path, "w") as f:
        f.write(f"Performance Profile Report - {date}\n")
        f.write("=" * 50 + "\n\n")

        total_time = sum(r["duration_seconds"] for r in PROFILE_RESULTS)
        total_memory = sum(abs(r["memory_delta_mb"]) for r in PROFILE_RESULTS)

        f.write(f"Summary:\n")
        f.write(f"  Total execution time: {total_time:.3f}s\n")
        f.write(f"  Total memory delta: {total_memory:.1f}MB\n")
        f.write(f"  Functions profiled: {len(PROFILE_RESULTS)}\n\n")

        f.write("Function Details:\n")
        f.write("-" * 30 + "\n")

        # Sort by duration descending
        sorted_results = sorted(
            PROFILE_RESULTS, key=lambda x: x["duration_seconds"], reverse=True
        )

        for result in sorted_results:
            f.write(f"Function: {result['function']}\n")
            f.write(f"  Duration: {result['duration_seconds']:.3f}s\n")
            f.write(f"  Memory: {result['memory_delta_mb']:+.1f}MB\n")
            f.write(f"  Timestamp: {result['timestamp']}\n\n")

    print(f"Profile report saved to: {report_path}")
    return str(report_path)


def clear_profile_results():
    """Clear accumulated profile results."""
    global PROFILE_RESULTS
    PROFILE_RESULTS = []
