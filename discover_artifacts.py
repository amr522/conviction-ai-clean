#!/usr/bin/env python3
"""
List all artifact files and sizes in markdown.
"""
import os
def main():
    artifacts = []
    for root, _, files in os.walk(sys.argv[1]):
        for f in files:
            if f.endswith(('.tar.gz','.pkl','.joblib')):
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                artifacts.append((os.path.relpath(path), size))
    # TODO: output markdown table
if __name__ == "__main__":
    import sys
    main()
