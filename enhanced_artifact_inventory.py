#!/usr/bin/env python3
"""
Enhanced artifact discovery and inventory management
Usage: python enhanced_artifact_inventory.py --input-dir /path/to/artifacts --output-file inventory.md
"""
import os
import argparse
from datetime import datetime

class ArtifactInventory:
    def __init__(self):
        self.supported_extensions = ['.pkl', '.joblib', '.tar.gz', '.zip', '.h5', '.pt', '.pth', '.onnx']
        self.artifact_patterns = ['model.tar.gz', 'xgboost-model', 'best_model', 'ensemble']
    
    def discover_artifacts(self, base_dir, recursive=True):
        """Recursively discover all model artifacts"""
        print(f"🔍 Discovering artifacts in {base_dir}")
        
        artifacts = []
        
        if recursive:
            for root, dirs, files in os.walk(base_dir):
                artifacts.extend(self._process_directory(root, files))
        else:
            if os.path.exists(base_dir):
                files = os.listdir(base_dir)
                artifacts.extend(self._process_directory(base_dir, files))
        
        artifacts.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        print(f"✅ Discovered {len(artifacts)} artifacts")
        return artifacts
    
    def _process_directory(self, directory, files):
        """Process files in a single directory"""
        artifacts = []
        
        for file in files:
            if self._is_artifact(file):
                full_path = os.path.join(directory, file)
                rel_path = os.path.relpath(full_path, '/workspace/conviction-ai-clean')
                
                try:
                    size_bytes = os.path.getsize(full_path)
                    size_str = self._format_size(size_bytes)
                    
                    artifacts.append({
                        'path': rel_path,
                        'size': size_str,
                        'size_bytes': size_bytes,
                        'type': self._get_artifact_type(file)
                    })
                    
                except Exception as e:
                    artifacts.append({
                        'path': rel_path,
                        'size': 'Unknown',
                        'size_bytes': 0,
                        'type': 'Unknown'
                    })
        
        return artifacts
    
    def _is_artifact(self, filename):
        """Check if file is a model artifact"""
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                return True
        
        for pattern in self.artifact_patterns:
            if pattern in filename:
                return True
        
        return False
    
    def _get_artifact_type(self, filename):
        """Determine artifact type"""
        if filename.endswith('.pkl'):
            return 'Pickle'
        elif filename.endswith('.joblib'):
            return 'Joblib'
        elif filename.endswith('.tar.gz'):
            return 'Compressed'
        elif filename.endswith(('.h5', '.pt', '.pth')):
            return 'Deep Learning'
        elif filename.endswith('.onnx'):
            return 'ONNX'
        else:
            return 'Other'
    
    def _format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        elif size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes} B"
    
    def generate_markdown_table(self, artifacts, output_file, title="Model Artifacts Inventory"):
        """Generate comprehensive Markdown table"""
        print(f"📋 Generating Markdown table...")
        
        total_size = sum(a['size_bytes'] for a in artifacts)
        
        content = f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Artifacts:** {len(artifacts)}  
**Total Size:** {self._format_size(total_size)}


"""
        
        type_summary = {}
        for artifact in artifacts:
            artifact_type = artifact['type']
            if artifact_type not in type_summary:
                type_summary[artifact_type] = {'count': 0, 'size': 0}
            type_summary[artifact_type]['count'] += 1
            type_summary[artifact_type]['size'] += artifact['size_bytes']
        
        content += "| Type | Count | Total Size |\n|------|-------|------------|\n"
        for artifact_type, stats in sorted(type_summary.items()):
            content += f"| {artifact_type} | {stats['count']} | {self._format_size(stats['size'])} |\n"
        
        content += f"""


| Path | Size | Type |
|------|------|------|
"""
        
        for artifact in artifacts:
            content += f"| {artifact['path']} | {artifact['size']} | {artifact['type']} |\n"
        
        content += f"""


- **Largest File:** {artifacts[0]['size'] if artifacts else 'N/A'}
- **Smallest File:** {artifacts[-1]['size'] if artifacts else 'N/A'}
- **Average Size:** {self._format_size(total_size // len(artifacts)) if artifacts else 'N/A'}

---
*Generated by enhanced artifact discovery system*
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Markdown table written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced artifact discovery and inventory')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory to search for artifacts')
    parser.add_argument('--output-file', type=str, default='artifacts_inventory.md',
                        help='Output markdown file')
    parser.add_argument('--title', type=str, default='Model Artifacts Inventory',
                        help='Title for the inventory report')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search recursively (default: True)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        exit(1)
    
    inventory = ArtifactInventory()
    artifacts = inventory.discover_artifacts(args.input_dir, args.recursive)
    
    if artifacts:
        inventory.generate_markdown_table(artifacts, args.output_file, args.title)
        print("✅ Artifact inventory complete")
    else:
        print("⚠️ No artifacts found")

if __name__ == "__main__":
    main()
