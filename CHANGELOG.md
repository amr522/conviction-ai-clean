# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-07-22

### Added
- AutoML V2 support with improved model training using SageMaker AutoML V2 API
- Multi-target evaluation for simultaneously evaluating multiple prediction targets
- Option-features ETL with Glue for preprocessing financial options data
- Volatility stack (EGARCH forecasting and IV-HV spread prediction)
- CI Slack notifications for automated pipeline status updates
- Automatic time-series split validation for all training scripts
- IBKR integration for automated trading execution
- Advanced regime detection with multi-factor analysis
- Feature lagging framework to prevent look-ahead bias in time series models
- Data validation framework with extensive integrity checks

### Changed
- Enhanced endpoint information tracking to include API version
- Improved backtesting harness with more realistic trading costs
- Refactored pipeline to use staged processing for better fault tolerance
- Upgraded AWS Step Functions integration for more robust workflow orchestration
- Optimized data preprocessing to handle larger datasets more efficiently

### Fixed
- Resolved issue with non-stationary time series features affecting model performance
- Fixed memory leak in blender ensemble training process
- Corrected forward-looking bias in several economic indicator features
- Addressed data leakage in cross-validation splits
- Improved error handling in AWS service interaction

## [2.0.0] - 2025-03-15

### Added
- Initial SageMaker Autopilot integration
- Basic option features ETL pipeline
- CI/CD workflow with GitHub Actions
- AWS Step Functions for workflow orchestration
- Baseline model evaluation framework
- Initial backtesting harness

### Changed
- Migrated from local training to AWS SageMaker
- Switched from CSV to Parquet for data storage
- Improved data preprocessing pipeline

### Fixed
- Addressed data quality issues in feature engineering
- Fixed configuration management for AWS resources
