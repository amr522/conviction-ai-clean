# AWS SageMaker HPO Setup Guide 🚀

**Complete your HPO in 2-4 hours for $20-40 instead of days locally**

## 🎯 Quick Setup (5 steps)

### 1. **Install Requirements**
```bash
pip install boto3 sagemaker awscli
```

### 2. **Configure AWS**
```bash
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)
```

### 3. **Create S3 Bucket & Upload Data**
```bash
aws s3 mb s3://your-hpo-bucket-unique-name
aws s3 sync data/processed_with_news_20250628/ s3://your-hpo-bucket-unique-name/data/
```

### 4. **Run Setup Script**
```bash
python aws_hpo_setup.py
```

### 5. **Launch HPO**
```bash
# Edit launch_hpo.py with your bucket name
python launch_hpo.py
```

## 💰 **Cost Breakdown**

| Component | Cost | Duration |
|-----------|------|----------|
| **ml.c5.xlarge** | $0.20/hour | 2-4 hours |
| **20 parallel jobs** | $4/hour total | 2-4 hours |
| **S3 storage** | $0.50 | One-time |
| **Total** | **$20-40** | **Complete in hours** |

## 🚀 **Speed Comparison**

- **Your Mac**: 5 processes, 2-3 days remaining
- **SageMaker**: 20 parallel jobs, 2-4 hours total
- **Speedup**: **10-20x faster**

## 🔧 **What It Does**

1. **Uploads** your data to AWS S3
2. **Launches** 20 parallel HPO jobs
3. **Tests** 25 hyperparameter combinations per symbol
4. **Downloads** results automatically
5. **Costs** $20-40 total vs days of local compute

## ⚡ **Alternative: Simple EC2**

If SageMaker seems complex:

```bash
# Launch big EC2 instance
aws ec2 run-instances --image-id ami-0abcdef1234567890 --instance-type c5.24xlarge --key-name your-key

# SSH and run your existing code
ssh -i your-key.pem ec2-user@instance-ip
# Upload and run: python run_hpo.py --symbols all --trials 25
```

**EC2 Cost**: ~$2-4/hour, complete in 4-8 hours = $10-30 total

## 🎯 **Recommendation**

**Start with EC2** (simpler):
1. Launch `c5.24xlarge` (96 CPUs)
2. Upload your existing code
3. Run: `python run_hpo_resumable.py --resume --trials 25`
4. Complete in 4-8 hours for $10-30

**Total savings**: Complete in hours instead of days for $10-40.

---

*Need help? The setup creates all files automatically.*