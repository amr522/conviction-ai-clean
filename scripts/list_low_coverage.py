#!/usr/bin/env python3
"""List modules with low coverage for test generation."""

import xml.etree.ElementTree as ET
import sys


def main():
    try:
        root = ET.parse('coverage.xml').getroot()
        low_cov_files = []
        
        print("Modules with coverage below 50%:")
        print("=" * 40)
        
        for pkg in root.findall('packages/package'):
            for cls in pkg.findall('classes/class'):
                cov = float(cls.get('line-rate', 0)) * 100
                filename = cls.get('filename', '')
                
                if cov < 50 and 'src/' in filename and not filename.endswith('__init__.py'):
                    print(f"{filename:<40} {cov:>6.1f}%")
                    low_cov_files.append(filename)
        
        print("\nFiles for test generation:")
        print(" ".join(low_cov_files))
        
    except FileNotFoundError:
        print("coverage.xml not found. Run 'coverage xml' first.")
        sys.exit(1)


if __name__ == '__main__':
    main()