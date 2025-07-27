# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-push Git hook to enforce full validation before push
- Master local validation script (`scripts/run-all-validations.sh`)
- Signal optimization utilities (`optimize_signal_generation`, `enhance_gamma_detection`)
- Enhanced gamma detection with configurable multipliers
- Risk mitigation alerts (`EnhancedGammaMissed`)
- Advanced signal validation with comprehensive test suite
- Anomaly monitoring dashboard and alerts
- Incident response automation and runbook generation
- Release-on-tag workflow for automated Docker and Helm publishing

### Changed
- Chart version bumped to 3.4.0 with appVersion 2025.07.27
- OpenLineage imports made optional to prevent import errors
- Enhanced signal validation thresholds and configuration

### Fixed
- Feature smoke test arguments corrected
- Import errors in lineage utilities resolved

## [3.4.0] - 2025-07-27

### Added
- Complete monitoring and optimization stack
- 9 phases of pipeline enhancements completed
- Comprehensive signal validation and risk mitigation

### Changed
- Enhanced performance utilities with signal optimization
- Improved validation coverage and testing

## [Previous Versions]

See git history for previous version details.