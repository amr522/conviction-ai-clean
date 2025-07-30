"""Tests for the profiling utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from src.utils.profiling import (PROFILE_RESULTS, clear_profile_results,
                                 enable_profiling, profile_memory_and_time,
                                 profile_time, save_profile_report)


def test_profiling_disabled_by_default():
    """Test that profiling is disabled by default."""

    @profile_time
    def dummy_function():
        return "test"

    # Should not add to results when disabled
    initial_count = len(PROFILE_RESULTS)
    result = dummy_function()
    assert result == "test"
    assert len(PROFILE_RESULTS) == initial_count


def test_profiling_enabled():
    """Test that profiling works when enabled."""
    enable_profiling()
    clear_profile_results()

    @profile_time
    def test_function():
        import time

        time.sleep(0.01)  # Small delay for measurable timing
        return "profiled"

    result = test_function()
    assert result == "profiled"
    assert len(PROFILE_RESULTS) == 1

    profile_entry = PROFILE_RESULTS[0]
    assert profile_entry["function"] == "test_function"
    assert profile_entry["duration_seconds"] > 0.005  # Should be at least 5ms
    assert "memory_start_mb" in profile_entry
    assert "memory_end_mb" in profile_entry
    assert "timestamp" in profile_entry


def test_memory_and_time_profiling():
    """Test combined memory and time profiling."""
    enable_profiling()
    clear_profile_results()

    @profile_memory_and_time
    def memory_intensive_function():
        # Create some data to use memory
        data = [i for i in range(1000)]
        return len(data)

    result = memory_intensive_function()
    assert result == 1000
    assert len(PROFILE_RESULTS) == 1


def test_save_profile_report():
    """Test profile report generation."""
    enable_profiling()
    clear_profile_results()

    @profile_time
    def sample_function():
        return "sample"

    sample_function()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            report_path = save_profile_report("2025-01-01")
            assert os.path.exists(report_path)

            # Check report content
            with open(report_path, "r") as f:
                content = f.read()
                assert "Performance Profile Report - 2025-01-01" in content
                assert "sample_function" in content
                assert "Duration:" in content
                assert "Memory:" in content
        finally:
            os.chdir(original_cwd)


def test_clear_profile_results():
    """Test clearing profile results."""
    enable_profiling()

    @profile_time
    def test_function():
        return "test"

    test_function()
    assert len(PROFILE_RESULTS) > 0

    clear_profile_results()
    assert len(PROFILE_RESULTS) == 0


def test_profile_report_empty():
    """Test profile report with no results."""
    clear_profile_results()

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            report_path = save_profile_report("2025-01-01")
            assert report_path is None  # Should return None when no results
        finally:
            os.chdir(original_cwd)
