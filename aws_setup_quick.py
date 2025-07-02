#!/usr/bin/env python3

def setup_aws_credentials():
    """Quick AWS setup before launching HPO"""
    
    print("🚀 AWS HPO LAUNCH SETUP")
    print("=" * 25)
    
    print("📋 REQUIRED UPDATES:")
    print("1. AWS Role ARN")
    print("2. S3 Bucket name")
    print("3. Upload data to S3")
    
    # Get user inputs
    role_arn = input("Enter your AWS Role ARN (or press Enter for default): ").strip()
    if not role_arn:
        role_arn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    
    bucket_name = input("Enter your S3 bucket name: ").strip()
    if not bucket_name:
        bucket_name = "hpo-bucket-" + str(int(time.time()))
        print(f"Using generated bucket name: {bucket_name}")
    
    # Update launch script
    with open('launch_hpo_final.py', 'r') as f:
        content = f.read()
    
    content = content.replace("'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'", f"'{role_arn}'")
    content = content.replace("'s3://your-hpo-bucket/", f"'s3://{bucket_name}/")
    content = content.replace("import sagemaker", "import sagemaker\nimport time")
    
    with open('launch_hpo_ready.py', 'w') as f:
        f.write(content)
    
    print(f"\n✅ Created launch_hpo_ready.py")
    print(f"   Role: {role_arn}")
    print(f"   Bucket: {bucket_name}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Create S3 bucket:")
    print(f"   aws s3 mb s3://{bucket_name}")
    print(f"2. Upload data:")
    print(f"   aws s3 sync data/processed_with_news_20250628/ s3://{bucket_name}/data/")
    print(f"3. Launch HPO:")
    print(f"   python launch_hpo_ready.py")
    
    return bucket_name, role_arn

if __name__ == "__main__":
    import time
    setup_aws_credentials()