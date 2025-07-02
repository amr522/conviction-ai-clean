I need you to train machine learning models for stock prediction on a dataset of 46 stocks. We've already implemented a robust pipeline with the following features:

1. **Robust Data Cleaning**: 
   - Date cleaning with proper error handling
   - Numeric conversion with validation
   - Missing value handling with conditional strategies

2. **Base Model Training**:
   - 11 different model architectures
   - Consistent hyperparameters across models
   - Cross-validation and evaluation metrics

3. **AWS SageMaker Integration**:
   - HPO configuration for XGBoost models
   - S3 data uploads and validation
   - Model registration and deployment

**CURRENT STATUS:** Base models trained for 46 stocks and SageMaker data prep completed locally with proper splits. Blocked on AWS credentials for S3 upload and SageMaker HPO. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION in environment variables so I can proceed with cloud steps.

Here's what I need you to do:

1. Clone the GitHub repository:
   ```bash
   git clone git@github.com:amr522/conviction-ai-clean.git
   cd conviction-ai-clean
   ```

2. The data for the 46 stocks is located at:
   - `data/processed_features/all_symbols_features.csv` (combined features for all symbols)
   - `config/models_to_train_46.txt` (list of 46 ticker symbols to process)
   - `config/hpo_config.yaml` (hyperparameter optimization configuration)

3. Train the base models for all 46 tickers using the same hyperparameters and pipeline as used for AAPL:
   ```bash
   python run_base_models.py --features-dir data/processed_features --symbols-file config/models_to_train_46.txt --output-dir data/base_model_outputs/46_models
   ```
   After this step, check that the model files exist in `data/base_model_outputs/46_models/`.

4. Bundle the SageMaker input for all 46 stocks:
   ```bash
   python prepare_sagemaker_data.py --input-file data/processed_features/all_symbols_features.csv --s3-bucket your-s3-bucket-name --s3-prefix 46_stocks
   ```
   This will create the necessary files in `data/sagemaker/46_stocks/` and upload them to S3.

5. Launch the HPO on AWS SageMaker:
   ```bash
   # IMPORTANT: Set these environment variables with your AWS credentials
   # Do NOT hardcode credentials in any files
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
   
   python aws_hpo_launch.py --config config/hpo_config.yaml --models-file config/models_to_train_46.txt --s3-prefix 46_stocks
   ```

6. Monitor the HPO progress:
   ```bash
   python check_hpo_progress.py
   ```

At each step, check for success and report progress. If any step fails, report the error and exit.

Important notes:
- All necessary data is already in the repository; no additional data processing is needed
- Use the existing 46 tickers in `config/models_to_train_46.txt` without any filtering
- Make sure AWS credentials are set via environment variables, not hardcoded
- The prepare_sagemaker_data.py script performs train/validation/test splits automatically

Let me know when you've completed each major phase (base model training, SageMaker data preparation, HPO launch) and the results.
