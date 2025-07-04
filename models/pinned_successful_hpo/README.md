# Pinned Successful HPO Training Artifacts

This directory contains the preserved artifacts from the successful full-universe HPO job that completed on July 4, 2025.

## Successful Training Details
- **HPO Job:** hpo-full-1751604591
- **Best Training Job:** hpo-full-1751604591-044-b07b4aa3
- **Validation AUC:** 1.0 (perfect score)
- **Completion Time:** 2025-07-04T05:03:06.795Z
- **Training Jobs:** 50/50 completed successfully, 0 failed

## Preserved Artifacts
- `best_model_hpo-full-1751604591-044-b07b4aa3.tar.gz` - Best model artifact from S3
- `best_hyperparameters.json` - Optimal hyperparameters from best training job
- `hpo_config_pinned.json` - Complete configuration and metadata

## Dataset Used
- **S3 URI:** s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv
- **Universe:** 46-stock filtered universe
- **Features:** Enhanced features with news sentiment and technical indicators

## Usage for Future Training
To use these artifacts for future training:
1. Reference the dataset URI from `hpo_config_pinned.json`
2. Use hyperparameters from `best_hyperparameters.json` as starting point
3. Deploy model from `best_model_hpo-full-1751604591-044-b07b4aa3.tar.gz`
