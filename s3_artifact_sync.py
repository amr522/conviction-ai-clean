#!/usr/bin/env python3
"""
Reusable S3 artifact synchronization utility
Usage: python s3_artifact_sync.py --s3-bucket my-bucket --s3-prefix path/to/artifacts --local-dir ./artifacts
"""
import os
import argparse
import boto3
import subprocess
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

class S3ArtifactSync:
    def __init__(self, region='us-east-1'):
        self.region = region
        
        try:
            self.s3 = boto3.client('s3', region_name=region)
            self.use_boto3 = True
            print("✅ Using boto3 for S3 operations")
        except (NoCredentialsError, Exception) as e:
            print(f"⚠️ boto3 not available ({e}), falling back to AWS CLI")
            self.use_boto3 = False
    
    def sync_from_s3(self, bucket, s3_prefix, local_dir, file_patterns=None, preserve_structure=True):
        """
        Download artifacts from S3 to local directory
        
        Args:
            bucket: S3 bucket name
            s3_prefix: S3 prefix/path to sync from
            local_dir: Local directory to sync to
            file_patterns: List of file patterns to include (e.g., ['*.pkl', '*.tar.gz'])
            preserve_structure: Whether to preserve S3 folder structure locally
        """
        print(f"📥 Syncing from s3://{bucket}/{s3_prefix} to {local_dir}")
        
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            if self.use_boto3:
                return self._sync_from_s3_boto3(bucket, s3_prefix, local_dir, file_patterns, preserve_structure)
            else:
                return self._sync_from_s3_cli(bucket, s3_prefix, local_dir, file_patterns, preserve_structure)
        except Exception as e:
            print(f"❌ Error syncing from S3: {e}")
            return False
    
    def sync_to_s3(self, local_dir, bucket, s3_prefix, file_patterns=None, preserve_structure=True):
        """
        Upload artifacts from local directory to S3
        
        Args:
            local_dir: Local directory to sync from
            bucket: S3 bucket name
            s3_prefix: S3 prefix/path to sync to
            file_patterns: List of file patterns to include
            preserve_structure: Whether to preserve local folder structure in S3
        """
        print(f"📤 Syncing from {local_dir} to s3://{bucket}/{s3_prefix}")
        
        try:
            if self.use_boto3:
                return self._sync_to_s3_boto3(local_dir, bucket, s3_prefix, file_patterns, preserve_structure)
            else:
                return self._sync_to_s3_cli(local_dir, bucket, s3_prefix, file_patterns, preserve_structure)
        except Exception as e:
            print(f"❌ Error syncing to S3: {e}")
            return False
    
    def _sync_from_s3_boto3(self, bucket, s3_prefix, local_dir, file_patterns, preserve_structure):
        """Sync from S3 using boto3"""
        downloaded_count = 0
        
        s3_prefix = s3_prefix.lstrip('/')
        
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                if file_patterns and not any(self._matches_pattern(key, pattern) for pattern in file_patterns):
                    continue
                
                if preserve_structure:
                    relative_path = key[len(s3_prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)
                else:
                    filename = os.path.basename(key)
                    local_path = os.path.join(local_dir, filename)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                should_download = True
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    s3_size = obj['Size']
                    if local_size == s3_size:
                        print(f"⏭️ Skipping {key} (already exists)")
                        should_download = False
                
                if should_download:
                    print(f"📥 Downloading {key}...")
                    self.s3.download_file(bucket, key, local_path)
                    downloaded_count += 1
        
        print(f"✅ Downloaded {downloaded_count} artifacts")
        return downloaded_count > 0
    
    def _sync_to_s3_boto3(self, local_dir, bucket, s3_prefix, file_patterns, preserve_structure):
        """Sync to S3 using boto3"""
        uploaded_count = 0
        
        s3_prefix = s3_prefix.lstrip('/')
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                if file_patterns and not any(self._matches_pattern(file, pattern) for pattern in file_patterns):
                    continue
                
                if preserve_structure:
                    relative_path = os.path.relpath(local_path, local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                else:
                    s3_key = f"{s3_prefix}/{file}"
                
                print(f"📤 Uploading {local_path} to {s3_key}...")
                self.s3.upload_file(local_path, bucket, s3_key)
                uploaded_count += 1
        
        print(f"✅ Uploaded {uploaded_count} artifacts")
        return uploaded_count > 0
    
    def _sync_from_s3_cli(self, bucket, s3_prefix, local_dir, file_patterns, preserve_structure):
        """Sync from S3 using AWS CLI"""
        try:
            s3_path = f"s3://{bucket}/{s3_prefix.lstrip('/')}"
            cmd = ['aws', 's3', 'sync', s3_path, local_dir]
            
            if file_patterns:
                cmd.extend(['--exclude', '*'])
                for pattern in file_patterns:
                    cmd.extend(['--include', pattern])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ AWS CLI sync completed: {result.stdout}")
            return True
            
        except Exception as e:
            print(f"❌ AWS CLI sync failed: {e}")
            return False
    
    def _sync_to_s3_cli(self, local_dir, bucket, s3_prefix, file_patterns, preserve_structure):
        """Sync to S3 using AWS CLI"""
        try:
            s3_path = f"s3://{bucket}/{s3_prefix.lstrip('/')}"
            cmd = ['aws', 's3', 'sync', local_dir, s3_path]
            
            if file_patterns:
                cmd.extend(['--exclude', '*'])
                for pattern in file_patterns:
                    cmd.extend(['--include', pattern])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ AWS CLI sync completed: {result.stdout}")
            return True
            
        except Exception as e:
            print(f"❌ AWS CLI sync failed: {e}")
            return False
    
    def _matches_pattern(self, filename, pattern):
        """Check if filename matches pattern (supports * wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def list_s3_artifacts(self, bucket, s3_prefix):
        """List all artifacts in S3 prefix"""
        print(f"📋 Listing artifacts in s3://{bucket}/{s3_prefix}")
        
        artifacts = []
        
        try:
            if self.use_boto3:
                paginator = self.s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix.lstrip('/'))
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        artifacts.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'modified': obj['LastModified']
                        })
            else:
                cmd = ['aws', 's3', 'ls', f"s3://{bucket}/{s3_prefix.lstrip('/')}", '--recursive']
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            artifacts.append({
                                'key': ' '.join(parts[3:]),
                                'size': int(parts[2]),
                                'modified': f"{parts[0]} {parts[1]}"
                            })
            
            print(f"✅ Found {len(artifacts)} artifacts")
            return artifacts
            
        except Exception as e:
            print(f"❌ Error listing S3 artifacts: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description='S3 artifact synchronization utility')
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket name')
    parser.add_argument('--s3-prefix', type=str, required=True,
                        help='S3 prefix/path')
    parser.add_argument('--local-dir', type=str, required=True,
                        help='Local directory')
    parser.add_argument('--direction', choices=['download', 'upload', 'list'], default='download',
                        help='Sync direction or list artifacts')
    parser.add_argument('--file-patterns', nargs='*',
                        help='File patterns to include (e.g., *.pkl *.tar.gz)')
    parser.add_argument('--preserve-structure', action='store_true', default=True,
                        help='Preserve folder structure')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region')
    
    args = parser.parse_args()
    
    sync = S3ArtifactSync(region=args.region)
    
    if args.direction == 'download':
        success = sync.sync_from_s3(
            args.s3_bucket, args.s3_prefix, args.local_dir,
            args.file_patterns, args.preserve_structure
        )
    elif args.direction == 'upload':
        success = sync.sync_to_s3(
            args.local_dir, args.s3_bucket, args.s3_prefix,
            args.file_patterns, args.preserve_structure
        )
    elif args.direction == 'list':
        artifacts = sync.list_s3_artifacts(args.s3_bucket, args.s3_prefix)
        for artifact in artifacts:
            print(f"{artifact['key']} ({artifact['size']} bytes)")
        success = len(artifacts) > 0
    
    if not success:
        print("❌ Operation failed")
        exit(1)
    
    print("✅ Operation completed successfully")

if __name__ == "__main__":
    main()
