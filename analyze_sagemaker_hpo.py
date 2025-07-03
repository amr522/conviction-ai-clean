import boto3
import json
import pandas as pd
from datetime import datetime
import os

def analyze_sagemaker_hpo():
    """Analyze the completed SageMaker HPO job results"""
    
    sm = boto3.client('sagemaker', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    
    job_name = 'hpo-full-1751555388'
    bucket_name = 'hpo-bucket-773934887314'
    
    print("🎉 SAGEMAKER HPO ANALYSIS")
    print("=" * 60)
    print(f"Analyzing HPO job: {job_name}")
    
    try:
        response = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )
        
        print(f"Status: {response['HyperParameterTuningJobStatus']}")
        print(f"Training Jobs: {response['TrainingJobStatusCounters']}")
        print(f"Creation Time: {response['CreationTime']}")
        
        if 'HyperParameterTuningEndTime' in response:
            print(f"End Time: {response['HyperParameterTuningEndTime']}")
            duration = response['HyperParameterTuningEndTime'] - response['CreationTime']
            print(f"Duration: {duration}")
        
        if 'BestTrainingJob' in response:
            best_job = response['BestTrainingJob']
            print(f"\n🏆 BEST TRAINING JOB:")
            print(f"Name: {best_job['TrainingJobName']}")
            
            if 'FinalHyperParameterTuningJobObjectiveMetric' in best_job:
                metric = best_job['FinalHyperParameterTuningJobObjectiveMetric']
                print(f"Best AUC Score: {metric.get('Value', 'N/A'):.4f}")
                print(f"Metric Name: {metric.get('MetricName', 'N/A')}")
        
        print(f"\n📊 RETRIEVING ALL TRAINING JOBS...")
        training_jobs = sm.list_training_jobs_for_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name,
            MaxResults=100
        )
        
        results = []
        for job in training_jobs['TrainingJobSummaries']:
            job_details = {
                'job_name': job['TrainingJobName'],
                'status': job['TrainingJobStatus'],
                'creation_time': job['CreationTime'],
                'training_end_time': job.get('TrainingEndTime'),
                'objective_value': None
            }
            
            if 'FinalHyperParameterTuningJobObjectiveMetric' in job:
                job_details['objective_value'] = job['FinalHyperParameterTuningJobObjectiveMetric'].get('Value')
            
            results.append(job_details)
        
        results_with_scores = [r for r in results if r['objective_value'] is not None]
        results_with_scores.sort(key=lambda x: x['objective_value'], reverse=True)
        
        print(f"\n🏅 TOP 10 TRAINING JOBS BY AUC SCORE:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Job Name':<45} {'Status':<12} {'AUC Score':<10}")
        print("-" * 80)
        
        for i, result in enumerate(results_with_scores[:10]):
            print(f"{i+1:<4} {result['job_name']:<45} {result['status']:<12} {result['objective_value']:<10.4f}")
        
        scores = [r['objective_value'] for r in results_with_scores]
        if scores:
            print(f"\n📈 PERFORMANCE STATISTICS:")
            print(f"Total completed jobs: {len(scores)}")
            print(f"Best AUC: {max(scores):.4f}")
            print(f"Worst AUC: {min(scores):.4f}")
            print(f"Average AUC: {sum(scores)/len(scores):.4f}")
            print(f"Median AUC: {sorted(scores)[len(scores)//2]:.4f}")
            
            above_55 = len([s for s in scores if s >= 0.55])
            above_60 = len([s for s in scores if s >= 0.60])
            above_70 = len([s for s in scores if s >= 0.70])
            above_80 = len([s for s in scores if s >= 0.80])
            
            print(f"\n🎯 THRESHOLD ANALYSIS:")
            print(f"Jobs with AUC ≥ 0.55: {above_55}/{len(scores)} ({above_55/len(scores)*100:.1f}%)")
            print(f"Jobs with AUC ≥ 0.60: {above_60}/{len(scores)} ({above_60/len(scores)*100:.1f}%)")
            print(f"Jobs with AUC ≥ 0.70: {above_70}/{len(scores)} ({above_70/len(scores)*100:.1f}%)")
            print(f"Jobs with AUC ≥ 0.80: {above_80}/{len(scores)} ({above_80/len(scores)*100:.1f}%)")
        
        print(f"\n💾 CHECKING S3 MODEL ARTIFACTS...")
        try:
            objects = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f'{job_name}',
                MaxKeys=50
            )
            
            if 'Contents' in objects:
                print(f"Found {len(objects['Contents'])} artifacts in S3:")
                model_files = []
                for obj in objects['Contents']:
                    if 'model.tar.gz' in obj['Key']:
                        model_files.append(obj)
                        print(f"  📦 {obj['Key']} ({obj['Size']} bytes)")
                
                print(f"\nTotal model files: {len(model_files)}")
            else:
                print("No artifacts found with HPO job prefix")
                
                print("Searching for any recent model artifacts...")
                objects = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix='',
                    MaxKeys=20
                )
                
                if 'Contents' in objects:
                    recent_objects = sorted(objects['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    print("Recent S3 objects:")
                    for obj in recent_objects[:10]:
                        print(f"  - {obj['Key']} ({obj['LastModified']})")
                
        except Exception as s3_error:
            print(f"Error checking S3: {s3_error}")
        
        if results_with_scores:
            results_df = pd.DataFrame(results_with_scores)
            results_df.to_csv('sagemaker_hpo_results.csv', index=False)
            print(f"\n💾 Results saved to sagemaker_hpo_results.csv")
        
        print(f"\n🚀 NEXT STEPS:")
        if len(results_with_scores) >= 20:  # Good coverage
            print("✅ HPO completed successfully with good coverage")
            print("1. Download best models from S3")
            print("2. Create ensemble from top performing models")
            print("3. Generate comprehensive performance report")
            print("4. Prepare for production deployment")
        else:
            print("⚠️ Limited results - may need to investigate")
            print("1. Check if more training jobs are still running")
            print("2. Verify S3 model artifacts are accessible")
            print("3. Consider re-running HPO if needed")
        
        return results_with_scores
        
    except Exception as e:
        print(f"❌ Error analyzing HPO results: {e}")
        return None

if __name__ == "__main__":
    analyze_sagemaker_hpo()
