import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_all_hpo_jobs():
    """Comprehensive analysis of all HPO jobs from the last 7 days"""
    
    sm = boto3.client('sagemaker', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    
    print("📊 COMPREHENSIVE HPO ANALYSIS - LAST 7 DAYS")
    print("=" * 80)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        response = sm.list_hyper_parameter_tuning_jobs(
            CreationTimeAfter=start_date,
            CreationTimeBefore=end_date,
            MaxResults=50
        )
        
        hpo_jobs = response.get('HyperParameterTuningJobSummaries', [])
        
        print(f"\n🎯 FOUND {len(hpo_jobs)} HPO JOBS:")
        print("-" * 80)
        
        all_results = []
        
        for i, job_summary in enumerate(hpo_jobs):
            job_name = job_summary['HyperParameterTuningJobName']
            status = job_summary['HyperParameterTuningJobStatus']
            creation_time = job_summary['CreationTime']
            
            print(f"\n{i+1}. JOB: {job_name}")
            print(f"   Status: {status}")
            print(f"   Created: {creation_time}")
            
            try:
                job_details = sm.describe_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=job_name
                )
                
                job_info = {
                    'job_name': job_name,
                    'status': status,
                    'creation_time': creation_time,
                    'training_job_counts': job_details.get('TrainingJobStatusCounters', {}),
                    'best_auc': None,
                    'data_source': None,
                    'container_image': None,
                    'instance_type': None
                }
                
                if 'BestTrainingJob' in job_details:
                    best_job = job_details['BestTrainingJob']
                    if 'FinalHyperParameterTuningJobObjectiveMetric' in best_job:
                        job_info['best_auc'] = best_job['FinalHyperParameterTuningJobObjectiveMetric'].get('Value')
                
                if 'TrainingJobDefinition' in job_details:
                    training_def = job_details['TrainingJobDefinition']
                    
                    if 'AlgorithmSpecification' in training_def:
                        job_info['container_image'] = training_def['AlgorithmSpecification'].get('TrainingImage', 'Unknown')
                    
                    if 'ResourceConfig' in training_def:
                        job_info['instance_type'] = training_def['ResourceConfig'].get('InstanceType', 'Unknown')
                    
                    if 'InputDataConfig' in training_def:
                        for input_config in training_def['InputDataConfig']:
                            if 'DataSource' in input_config and 'S3DataSource' in input_config['DataSource']:
                                s3_uri = input_config['DataSource']['S3DataSource'].get('S3Uri', 'Unknown')
                                job_info['data_source'] = s3_uri
                                break
                
                print(f"   Training Jobs: {job_info['training_job_counts']}")
                print(f"   Best AUC: {job_info['best_auc']:.4f}" if job_info['best_auc'] else "   Best AUC: N/A")
                print(f"   Data Source: {job_info['data_source']}")
                print(f"   Container: {job_info['container_image']}")
                print(f"   Instance: {job_info['instance_type']}")
                
                if status == 'Completed':
                    try:
                        training_jobs = sm.list_training_jobs_for_hyper_parameter_tuning_job(
                            HyperParameterTuningJobName=job_name,
                            MaxResults=100
                        )
                        
                        job_scores = []
                        for tj in training_jobs['TrainingJobSummaries']:
                            if 'FinalHyperParameterTuningJobObjectiveMetric' in tj:
                                score = tj['FinalHyperParameterTuningJobObjectiveMetric'].get('Value')
                                if score is not None:
                                    job_scores.append(score)
                        
                        if job_scores:
                            job_info['num_completed'] = len(job_scores)
                            job_info['avg_auc'] = sum(job_scores) / len(job_scores)
                            job_info['min_auc'] = min(job_scores)
                            job_info['max_auc'] = max(job_scores)
                            job_info['scores_above_55'] = len([s for s in job_scores if s >= 0.55])
                            job_info['scores_above_80'] = len([s for s in job_scores if s >= 0.80])
                            job_info['scores_above_95'] = len([s for s in job_scores if s >= 0.95])
                            job_info['perfect_scores'] = len([s for s in job_scores if s >= 0.999])
                            
                            print(f"   Completed Jobs: {job_info['num_completed']}")
                            print(f"   Average AUC: {job_info['avg_auc']:.4f}")
                            print(f"   AUC Range: {job_info['min_auc']:.4f} - {job_info['max_auc']:.4f}")
                            print(f"   Jobs ≥ 0.55 AUC: {job_info['scores_above_55']}/{job_info['num_completed']}")
                            print(f"   Jobs ≥ 0.80 AUC: {job_info['scores_above_80']}/{job_info['num_completed']}")
                            print(f"   Jobs ≥ 0.95 AUC: {job_info['scores_above_95']}/{job_info['num_completed']}")
                            print(f"   Perfect Scores (≥0.999): {job_info['perfect_scores']}")
                            
                            if job_info['perfect_scores'] > 0:
                                print(f"   🚨 SUSPICIOUS: {job_info['perfect_scores']} perfect/near-perfect scores detected!")
                            if job_info['avg_auc'] > 0.90:
                                print(f"   ⚠️ UNUSUAL: Very high average AUC ({job_info['avg_auc']:.4f}) for stock prediction")
                    
                    except Exception as e:
                        print(f"   Error getting training job details: {e}")
                
                all_results.append(job_info)
                
            except Exception as e:
                print(f"   Error getting job details: {e}")
                all_results.append({
                    'job_name': job_name,
                    'status': status,
                    'creation_time': creation_time,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('comprehensive_hpo_analysis.csv', index=False)
        
        print(f"\n📋 SUMMARY ANALYSIS:")
        print("-" * 40)
        
        completed_jobs = [r for r in all_results if r.get('status') == 'Completed']
        print(f"Total HPO Jobs: {len(all_results)}")
        print(f"Completed Jobs: {len(completed_jobs)}")
        
        if completed_jobs:
            data_sources = {}
            for job in completed_jobs:
                source = job.get('data_source', 'Unknown')
                if source not in data_sources:
                    data_sources[source] = []
                data_sources[source].append(job['job_name'])
            
            print(f"\n📁 DATA SOURCES USED:")
            for source, jobs in data_sources.items():
                print(f"  {source}:")
                for job_name in jobs:
                    print(f"    - {job_name}")
            
            jobs_with_scores = [j for j in completed_jobs if j.get('avg_auc') is not None]
            if jobs_with_scores:
                print(f"\n🎯 PERFORMANCE ANALYSIS:")
                for job in jobs_with_scores:
                    print(f"  {job['job_name']}:")
                    print(f"    Average AUC: {job['avg_auc']:.4f}")
                    print(f"    Perfect Scores: {job.get('perfect_scores', 0)}")
                    
                    if job.get('perfect_scores', 0) > 0:
                        print(f"    🚨 DATA QUALITY CONCERN: Perfect scores indicate potential data leakage")
                    elif job['avg_auc'] > 0.85:
                        print(f"    ⚠️ REVIEW NEEDED: Unusually high performance for stock prediction")
                    elif job['avg_auc'] > 0.70:
                        print(f"    ✅ GOOD: Strong performance within reasonable range")
                    else:
                        print(f"    📊 BASELINE: Performance within expected range")
        
        print(f"\n🚀 RECOMMENDATIONS:")
        suspicious_jobs = [j for j in completed_jobs if j.get('perfect_scores', 0) > 0 or j.get('avg_auc', 0) > 0.90]
        
        if suspicious_jobs:
            print("🚨 IMMEDIATE ACTION REQUIRED:")
            print("  1. Investigate data leakage in training data")
            print("  2. Verify target generation process")
            print("  3. Re-validate feature engineering pipeline")
            print("  4. Consider re-running HPO with clean data")
            print(f"  5. Suspicious jobs: {[j['job_name'] for j in suspicious_jobs]}")
        else:
            print("✅ No obvious data quality issues detected")
            print("  1. Proceed with ensemble training")
            print("  2. Validate on holdout test set")
            print("  3. Prepare for production deployment")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Error analyzing HPO jobs: {e}")
        return None

if __name__ == "__main__":
    analyze_all_hpo_jobs()
