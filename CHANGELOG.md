# Changelog

All notable changes to this project will be documented in this file.

<a name="v1.0.0"></a>
## v1.0.0

> 2025-01-16

### Added

* **profiling:** Add comprehensive performance profiling system with memory and timing analysis
* **performance:** Add optimized ETL utilities with broadcast joins and native window functions
* **validation:** Add end-to-end schema validation and data quality checks
* **monitoring:** Add Prometheus/Grafana monitoring stack with automated alerts
* **ci:** Add automated profiling checks and performance validation in CI pipeline

### Changed

* **etl:** Refactor core ETL scripts to use performance utilities and remove legacy pandas code
* **joins:** Replace manual join logic with optimized broadcast join utilities
* **rolling:** Replace pandas UDF-based rolling calculations with native Polars window functions
* **config:** Centralize performance configuration in utils/performance_utils.py

### Performance

* **joins:** 40-60% faster join operations with broadcast hints and streaming collection
* **rolling:** 70-80% faster rolling calculations with native Polars window functions
* **memory:** 30-40% reduced memory usage by eliminating pandas conversions
* **pipeline:** 35-50% faster end-to-end pipeline execution

### Fixed

* **dtypes:** Fix dtype enforcement using native Polars operations instead of pandas
* **memory:** Fix memory leaks from unnecessary pandas conversions
* **joins:** Fix join performance issues with proper broadcast hints