# Conviction-AI Machine Learning Pipeline

This repository contains the Conviction-AI machine learning pipeline, which automates data processing, model training, and deployment to AWS.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Repository Structure](#repository-structure)
- [AWS Pipeline](#aws-pipeline)
- [Running Tests Locally](#running-tests-locally)
- [CI/CD Pipeline](#cicd-pipeline)
- [Slack Notifications](#slack-notifications)
- [AWS Resources](#aws-resources)
- [Additional Documentation](#additional-documentation)

## 🔧 Environment Setup

### Prerequisites

- Python 3.9+
- AWS CLI installed and configured
- Appropriate AWS IAM permissions for S3, Glue, and SageMaker
- Git

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/conviction-ai-clean.git
   cd conviction-ai-clean
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   
   Create a `.env` file with the following required variables:
   ```
   # AWS credentials (or use IAM roles)
   AWS_ACCESS_KEY_ID=your_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   
   # S3 bucket for data storage
   S3_BUCKET_NAME=your-bucket-name
   
   # Slack webhook for notifications
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
   ```

4. For Slack notifications integration:
   - Create a Slack App in your workspace
   - Enable Incoming Webhooks
   - Create a webhook for your desired channel
   - Add the webhook URL to your `.env` file

## 📁 Repository Structure

The repository is organized into the following key directories:

```
conviction-ai-clean/
├── aws_pipeline/            # AWS SageMaker and pipeline components
│   ├── model_analysis.py    # Tools for model evaluation and analysis
│   ├── run_aws_pipeline.sh  # Main script to execute the full AWS pipeline
│   ├── setup_aws_env.sh     # Setup script for AWS environment
│   └── README.md            # AWS pipeline documentation
├── data/                    # Data directory (not tracked in git)
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── predictions/         # Model predictions
├── tests/                   # Test suite
├── .env                     # Environment variables (not tracked in git)
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # Main documentation
```

## ☁️ AWS Pipeline

The `aws_pipeline` directory contains all components for the AWS SageMaker and data processing pipeline. This includes scripts for model training, deployment, and analysis.

### Setup and Configuration

To set up the AWS environment:

1. Navigate to the aws_pipeline directory:
   ```bash
   cd aws_pipeline
   ```

2. Make the setup script executable and run it:
   ```bash
   chmod +x setup_aws_env.sh
   ./setup_aws_env.sh
   ```

### Running the Pipeline

To execute the full AWS pipeline:

1. Make the pipeline script executable:
   ```bash
   chmod +x run_aws_pipeline.sh
   ```

2. Run the pipeline:
   ```bash
   ./run_aws_pipeline.sh
   ```

### Model Analysis

To analyze a deployed SageMaker model for data leakage and overfitting:

```bash
python aws_pipeline/model_analysis.py
```

The script will:
- Connect to your AWS account
- Load data from S3
- Evaluate model predictions
- Generate analysis reports for data leakage and model performance

## 🧪 Running Tests Locally

The project includes a comprehensive test suite to validate functionality. To run tests locally:

### Install test dependencies:

```bash
pip install pytest pytest-xdist coverage
```

### Run all tests:

```bash
python -m pytest tests/ -v
```

### Run tests in parallel:

```bash
python -m pytest tests/ -n auto
```

### Generate a coverage report:

```bash
coverage run -m pytest tests/
coverage report
coverage html  # Creates an HTML report in htmlcov/
```

Tests must meet a minimum of 80% code coverage to pass.

## 🚀 CI/CD Pipeline

The CI/CD pipeline is implemented using GitHub Actions and defined in `.github/workflows/aws_ml_pipeline.yml`. It automates testing, code quality checks, and deployment to AWS.

### Pipeline Triggers

The pipeline runs automatically:
- On every push to the `main` branch
- On a daily schedule at 03:00 UTC

### Pipeline Workflow

The pipeline includes the following key steps:

1. **Setup Environment**: 
   - Checkout code
   - Configure AWS credentials
   - Set up Python 3.9
   - Install dependencies

2. **Code Quality Checks**:
   - Linting with flake8
   - Static code analysis

3. **Run Tests**:
   - Execute tests in parallel using pytest-xdist
   - Generate and check code coverage
   - Fail if coverage is below 80%
   - Upload coverage report as an artifact

4. **Slack Notification (Tests)**:
   - Notify team on test success or failure
   - Include repository, workflow, and run details

5. **Data Processing**:
   - Reorganize S3 bucket
   - Start AWS Glue ETL job
   - Poll for job completion

6. **Model Training & Deployment**:
   - Run SageMaker Autopilot job
   - Deploy trained model to endpoint
   - Log results and endpoint URL

7. **Final Notification**:
   - Send comprehensive Slack notification with pipeline results
   - Include links to deployed resources

### Concurrency Control

The pipeline uses GitHub Actions concurrency controls to prevent overlapping runs, ensuring that only one instance runs at a time.

## 💬 Slack Notifications

The pipeline integrates with Slack to provide real-time notifications about pipeline status.

### Notification Types

1. **Test Results Notification**:
   - Sent immediately after tests complete
   - Indicates test success or failure
   - Includes repository and workflow details
   - For failures, includes a link to GitHub Actions logs

2. **Pipeline Completion Notification**:
   - Sent when the entire pipeline completes
   - On success: includes Glue job, AutoML job, and endpoint details
   - On failure: includes link to GitHub Actions logs for troubleshooting

### Setting Up Slack Notifications

1. Create a Slack App in your workspace
2. Enable Incoming Webhooks
3. Create a webhook for your channel
4. Add the webhook URL as a GitHub secret named `SLACK_WEBHOOK_URL`

## ☁️ AWS Resources

The pipeline interacts with the following AWS resources:

- **S3 Bucket**: Stores raw and processed data
- **AWS Glue**: Runs ETL jobs to transform and prepare data
- **SageMaker Autopilot**: Automatically trains and tunes machine learning models
- **SageMaker Endpoint**: Hosts the deployed model for inference

### AWS Pipeline Components

The `aws_pipeline` directory contains all the necessary components for AWS interaction:

- **run_sagemaker_autopilot.py**: Launches and monitors SageMaker Autopilot jobs
- **cleanup_sagemaker_resources.py**: Manages SageMaker resource cleanup
- **model_analysis.py**: Analyzes model performance and checks for data leakage
- **run_aws_pipeline.sh**: Main script that orchestrates the entire AWS workflow
- **setup_aws_env.sh**: Sets up required environment variables and dependencies

### Infrastructure as Code

The project includes CloudFormation templates to provision all required AWS resources:

#### Deploying the Infrastructure:

You can deploy the infrastructure using the AWS pipeline scripts:

```bash
# Set up the environment first
cd aws_pipeline
./setup_aws_env.sh

# Deploy the infrastructure and run the pipeline
./run_aws_pipeline.sh
```

#### Cleanup Resources:

To clean up AWS resources after use:

```bash
python aws_pipeline/cleanup_sagemaker_resources.py
```

## 📚 Additional Documentation

- For AWS pipeline details, see [aws_pipeline/README.md](aws_pipeline/README.md)
- For GPU training information, see [GPU_TRAINING_INSTRUCTIONS.md](GPU_TRAINING_INSTRUCTIONS.md)
- For model performance analysis, see [MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md)

## 📖 Tutorials

- [Autopilot_V2_TimeSeries_Split_Tutorial.ipynb](Autopilot_V2_TimeSeries_Split_Tutorial.ipynb) - Tutorial on using SageMaker Autopilot V2 with time-series splits

## License

[Specify your license here]

## Contact

[Your contact information]
