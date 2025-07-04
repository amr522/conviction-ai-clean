#!/usr/bin/env python3
"""
Train Price-GRU model using PyTorch on SageMaker
"""
import boto3
import json
import time
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_gru_training_script():
    """Create the GRU training script for SageMaker"""
    script_content = '''
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import argparse
import os
import json

class PriceGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(PriceGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def prepare_sequences(data, sequence_length=10):
    """Prepare sequences for GRU training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 1:])  # Features (excluding target)
        y.append(data[i, 0])  # Target
    return np.array(X), np.array(y)

def train_model(args):
    """Train the GRU model"""
    
    train_data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data.values)
    
    X, y = prepare_sequences(scaled_data, sequence_length=args.sequence_length)
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    
    input_size = X_train.shape[2]
    model = PriceGRU(input_size, args.hidden_size, args.num_layers, args.dropout)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_auc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                val_pred = val_outputs.numpy().flatten()
                val_true = y_val.numpy().flatten()
                auc = roc_auc_score(val_true, val_pred)
                
                print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, AUC: {auc:.4f}')
                
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= args.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
    
    metrics = {
        'best_validation_auc': float(best_auc),
        'final_epoch': epoch
    }
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print(f'Training completed. Best AUC: {best_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    
    args = parser.parse_args()
    train_model(args)
'''
    return script_content

def launch_gru_training(input_data_s3: str, epochs: int = 50, dry_run: bool = False) -> str | None:
    """Launch GRU training job on SageMaker"""
    
    timestamp = int(time.time())
    job_name = f"price-gru-{timestamp}"
    
    if dry_run:
        logger.info(f"🧪 DRY RUN: Would launch Price-GRU training job: {job_name}")
        return job_name
    
    sagemaker = boto3.client('sagemaker')
    
    script_content = create_gru_training_script()
    
    s3 = boto3.client('s3')
    bucket = "hpo-bucket-773934887314"
    script_key = f"gru-training/{timestamp}/train.py"
    
    s3.put_object(
        Bucket=bucket,
        Key=script_key,
        Body=script_content.encode('utf-8')
    )
    
    training_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"
    role_arn = "arn:aws:iam::773934887314:role/SageMakerExecutionRole"
    output_path = f"s3://{bucket}/gru-models/{timestamp}/"
    
    hyperparameters = {
        'epochs': str(epochs),
        'hidden-size': '64',
        'num-layers': '2',
        'dropout': '0.2',
        'learning-rate': '0.001',
        'sequence-length': '10',
        'early-stopping-patience': '10'
    }
    
    training_config = {
        'TrainingJobName': job_name,
        'AlgorithmSpecification': {
            'TrainingImage': training_image,
            'TrainingInputMode': 'File'
        },
        'RoleArn': role_arn,
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_data_s3,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv',
                'CompressionType': 'None'
            },
            {
                'ChannelName': 'code',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{bucket}/gru-training/{timestamp}/",
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': output_path
        },
        'ResourceConfig': {
            'InstanceType': 'ml.p3.2xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200
        },
        'HyperParameters': hyperparameters,
        'Environment': {
            'SAGEMAKER_PROGRAM': 'train.py'
        }
    }
    
    try:
        logger.info(f"🚀 Launching Price-GRU training job: {job_name}")
        logger.info(f"📊 Input data: {input_data_s3}")
        logger.info(f"📁 Output path: {output_path}")
        logger.info(f"🔧 Epochs: {epochs}")
        
        response = sagemaker.create_training_job(**training_config)
        
        logger.info(f"✅ Successfully launched Price-GRU training job: {job_name}")
        logger.info(f"🔗 Job ARN: {response['TrainingJobArn']}")
        
        return job_name
        
    except Exception as e:
        logger.error(f"❌ Failed to launch Price-GRU training job: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Price-GRU model on SageMaker')
    parser.add_argument('--input-data-s3', type=str, required=True,
                        help='S3 URI for training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without launching actual job')
    
    args = parser.parse_args()
    
    job_name = launch_gru_training(args.input_data_s3, args.epochs, args.dry_run)
    
    if job_name:
        print(f"✅ Price-GRU training job launched: {job_name}")
    else:
        print("❌ Failed to launch Price-GRU training job")
        exit(1)

if __name__ == "__main__":
    main()
