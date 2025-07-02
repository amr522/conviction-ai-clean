#!/bin/bash
# Script to push only the necessary files for 11-model and HPO training to GitHub

# Create a temporary directory for the essential files
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# List of essential files for 11-model and HPO training
ESSENTIAL_FILES=(
  # Core scripts
  "prepare_sagemaker_data.py"
  "run_base_models.py"
  "train_models_and_prepare_56_new.sh"
  "run_full_robust_pipeline.sh"
  "requirements.txt"
  "environment.yml"
  "Dockerfile"
  
  # AWS related files
  "aws_sagemaker_hpo.py"
  "aws_hpo_setup.py"
  "aws_hpo_setup_updated.py"
  "aws_hpo_final.py"
  "aws_simple_setup.py"
  "aws_direct_access.py"
  "aws_setup_quick.py"
  "aws_hpo_launch.py"
  "create_sagemaker_role.py"
  "cleanup_sagemaker_resources.py"
  "register_model.py"
  "setup_monitoring.py"
  "create_monitoring_baseline.py"
  
  # HPO and Model Management
  "hpo_progress_compare.py"
  "check_hpo_progress.py"
  "check_hpo_status.py"
  "automate_hpo_monitoring.py"
  "analyze_hpo_results.py"
  
  # Config files
  "config.yaml"
  "config/hpo_config.yaml"
  "config/models_to_train.txt"
  
  # Symbol files
  "enhanced_symbols.txt"
  "regular_symbols.txt"
  "all_enhanced_symbols.txt"
  "all_regular_symbols.txt"
  
  # Utility scripts
  "run_with_timeout.sh"
  "promote_model.sh"
  "list_registered_models.sh"
  "setup_model_alerts.sh"
  "robust_hpo.sh"
  
  # Documentation
  "README.md"
  "IMPLEMENTATION_STATUS.md"
  "AWS_HPO_GUIDE.md"
  "AWS_HPO_IMPLEMENTATION_REPORT.md"
  "ENHANCED_TRAINING_README.md"
)

# Create necessary directories in the temp location
mkdir -p "$TEMP_DIR/config"
mkdir -p "$TEMP_DIR/pipeline_logs"
mkdir -p "$TEMP_DIR/data/base_model_outputs/11_models"
mkdir -p "$TEMP_DIR/data/sagemaker"
mkdir -p "$TEMP_DIR/data/processed_features"
mkdir -p "$TEMP_DIR/models/debug_hpo"

# Copy essential files to the temp directory
for file in "${ESSENTIAL_FILES[@]}"; do
  # Create directory if needed
  dir=$(dirname "$file")
  if [ "$dir" != "." ]; then
    mkdir -p "$TEMP_DIR/$dir"
  fi
  
  # Copy file if it exists
  if [ -f "$file" ]; then
    cp "$file" "$TEMP_DIR/$file"
    echo "Copied: $file"
  else
    echo "Warning: $file not found, skipping..."
  fi
done

# Check if any required files were missing
MISSING_FILES=0
REQUIRED_FILES=(
  "prepare_sagemaker_data.py"
  "run_base_models.py"
  "train_models_and_prepare_56_new.sh"
  "run_full_robust_pipeline.sh"
  "requirements.txt"
  "environment.yml"
  "Dockerfile"
  "config/hpo_config.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$TEMP_DIR/$file" ]; then
    echo "ERROR: Required file missing: $file"
    MISSING_FILES=$((MISSING_FILES+1))
  fi
done

if [ $MISSING_FILES -gt 0 ]; then
  echo "WARNING: $MISSING_FILES required files are missing. The GitHub repo may not work correctly."
  read -p "Do you want to continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please ensure all required files are available."
    exit 1
  fi
fi

# Create a simple .gitignore file
cat > "$TEMP_DIR/.gitignore" << EOL
# Ignore large data files
*.csv
*.parquet
*.feather
*.h5
*.pkl

# Ignore logs except example ones
*.log
!example_logs/*.log

# Ignore Python artifacts
__pycache__/
*.py[cod]
*\$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Ignore Jupyter Notebook
.ipynb_checkpoints

# Ignore VS Code settings
.vscode/

# Ignore Mac OS specific files
.DS_Store

# Ignore environment
.env
.venv

# Ignore large model files
*.model
*.bin
*.joblib
*.onnx
EOL

# Add README.md if it doesn't exist
if [ ! -f "$TEMP_DIR/README.md" ]; then
  cat > "$TEMP_DIR/README.md" << EOL
# Conviction AI ML Pipeline

This repository contains the essential files for running the 11-model and HPO training pipeline.

## Key Components

- Base Model Training (11 different models)
- Hyperparameter Optimization (HPO)
- AWS SageMaker Integration
- Robust Data Cleaning
- Model Registry Integration
- Monitoring and Alerts

## Quick Start

1. Clone this repository
2. Set up the Python environment: \`conda env create -f environment.yml\`
3. Activate the environment: \`conda activate conviction-ai\`
4. Configure AWS credentials (if using SageMaker)
5. Run the pipeline: \`./train_models_and_prepare_56_new.sh\`

## Documentation

- See AWS_HPO_GUIDE.md for AWS HPO setup instructions
- See IMPLEMENTATION_STATUS.md for current implementation status
- See ENHANCED_TRAINING_README.md for enhanced training documentation
EOL
  echo "Created default README.md"
fi

# Initialize git in the temp directory
cd "$TEMP_DIR"
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit with essential ML pipeline files"

echo "==============================================================="
echo "Repository prepared in: $TEMP_DIR"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Run the following commands to push to your GitHub repository:"
echo ""
echo "   cd $TEMP_DIR"
echo "   git remote add origin https://github.com/YOUR_USERNAME/conviction-ai.git"
echo "   git push -u origin main"
echo ""
echo "Replace 'YOUR_USERNAME' with your actual GitHub username"
echo "==============================================================="
