import boto3
import os
import pandas as pd
from datetime import datetime

def download_best_models():
    """Download the best performing models from S3"""
    
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'hpo-bucket-773934887314'
    
    print("🔍 SEARCHING FOR MODEL ARTIFACTS...")
    
    if os.path.exists('sagemaker_hpo_results.csv'):
        results_df = pd.read_csv('sagemaker_hpo_results.csv')
        
        top_models = results_df.nlargest(10, 'objective_value')
        
        print(f"📦 DOWNLOADING TOP 10 MODELS:")
        print("-" * 60)
        
        os.makedirs('models/best_hpo', exist_ok=True)
        
        for idx, row in top_models.iterrows():
            job_name = row['job_name']
            auc_score = row['objective_value']
            
            print(f"Model {idx+1}: {job_name} (AUC: {auc_score:.4f})")
            
            possible_paths = [
                f"{job_name}/output/model.tar.gz",
                f"models/{job_name}/output/model.tar.gz", 
                f"output/{job_name}/model.tar.gz",
                f"{job_name}/model.tar.gz"
            ]
            
            downloaded = False
            for path in possible_paths:
                try:
                    s3.head_object(Bucket=bucket_name, Key=path)
                    
                    local_path = f"models/best_hpo/{job_name}_model.tar.gz"
                    s3.download_file(bucket_name, path, local_path)
                    
                    print(f"  ✅ Downloaded: {path} -> {local_path}")
                    downloaded = True
                    break
                    
                except Exception as e:
                    continue
            
            if not downloaded:
                print(f"  ❌ Model not found in S3: {job_name}")
        
        print(f"\n📊 DOWNLOAD SUMMARY:")
        downloaded_files = os.listdir('models/best_hpo') if os.path.exists('models/best_hpo') else []
        print(f"Successfully downloaded: {len(downloaded_files)} models")
        
        return len(downloaded_files)
    
    else:
        print("❌ HPO results file not found. Run analyze_sagemaker_hpo.py first.")
        return 0

def list_s3_contents():
    """List all S3 contents to understand structure"""
    
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'hpo-bucket-773934887314'
    
    print("\n🗂️ S3 BUCKET STRUCTURE:")
    print("-" * 40)
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        all_keys = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_keys.append(obj['Key'])
        
        prefixes = {}
        for key in all_keys:
            prefix = key.split('/')[0] if '/' in key else 'root'
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(key)
        
        for prefix, keys in prefixes.items():
            print(f"\n📁 {prefix}/ ({len(keys)} files)")
            for key in keys[:5]:  # Show first 5 files
                print(f"  - {key}")
            if len(keys) > 5:
                print(f"  ... and {len(keys)-5} more files")
                
    except Exception as e:
        print(f"Error listing S3 contents: {e}")

if __name__ == "__main__":
    list_s3_contents()
    download_best_models()
