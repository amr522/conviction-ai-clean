#!/usr/bin/env python3
"""
Unified artifact management utility
Usage: 
  python artifact_manager.py inventory --input-dir models --output-file inventory.md
  python artifact_manager.py sync --s3-bucket my-bucket --s3-prefix path --local-dir ./artifacts --direction download
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_artifact_inventory import ArtifactInventory
from s3_artifact_sync import S3ArtifactSync

def main():
    parser = argparse.ArgumentParser(description='Unified artifact management utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    inventory_parser = subparsers.add_parser('inventory', help='Generate artifact inventory')
    inventory_parser.add_argument('--input-dir', type=str, required=True)
    inventory_parser.add_argument('--output-file', type=str, default='artifacts_inventory.md')
    inventory_parser.add_argument('--title', type=str, default='Artifact Inventory')
    
    sync_parser = subparsers.add_parser('sync', help='Sync artifacts with S3')
    sync_parser.add_argument('--s3-bucket', type=str, required=True)
    sync_parser.add_argument('--s3-prefix', type=str, required=True)
    sync_parser.add_argument('--local-dir', type=str, required=True)
    sync_parser.add_argument('--direction', choices=['download', 'upload', 'list'], default='download')
    sync_parser.add_argument('--file-patterns', nargs='*')
    sync_parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    if args.command == 'inventory':
        inventory = ArtifactInventory()
        artifacts = inventory.discover_artifacts(args.input_dir)
        if artifacts:
            inventory.generate_markdown_table(artifacts, args.output_file, args.title)
            print("✅ Inventory generation complete")
        else:
            print("⚠️ No artifacts found")
    
    elif args.command == 'sync':
        sync = S3ArtifactSync(region=args.region)
        if args.direction == 'download':
            success = sync.sync_from_s3(args.s3_bucket, args.s3_prefix, args.local_dir, args.file_patterns)
        elif args.direction == 'upload':
            success = sync.sync_to_s3(args.local_dir, args.s3_bucket, args.s3_prefix, args.file_patterns)
        elif args.direction == 'list':
            artifacts = sync.list_s3_artifacts(args.s3_bucket, args.s3_prefix)
            success = len(artifacts) > 0
        
        if not success:
            exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
