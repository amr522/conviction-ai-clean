#!/usr/bin/env python3
"""Generate test stubs for modules with low coverage."""

import ast
import argparse
import os
from pathlib import Path


def extract_functions_and_classes(file_path):
    """Extract public functions and classes from a Python file."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []
    
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            classes.append(node.name)
    
    return functions, classes


def generate_test_stub(module_path, output_dir):
    """Generate test stub for a module."""
    module_name = Path(module_path).stem
    relative_path = Path(module_path).relative_to('src')
    import_path = str(relative_path).replace('/', '.').replace('.py', '')
    
    functions, classes = extract_functions_and_classes(module_path)
    
    test_content = f'''"""Tests for {import_path} module."""

import pytest
from unittest.mock import Mock, patch
from {import_path} import *


class Test{module_name.title().replace('_', '')}:
    """Test class for {module_name} module."""
'''
    
    # Add function tests
    for func in functions:
        test_content += f'''
    def test_{func}_exists(self):
        """Test that {func} function exists."""
        assert callable({func})
    
    def test_{func}_basic(self):
        """Test basic functionality of {func}."""
        # TODO: Add meaningful test implementation
        pass
'''
    
    # Add class tests
    for cls in classes:
        test_content += f'''
    def test_{cls.lower()}_exists(self):
        """Test that {cls} class exists."""
        assert {cls} is not None
    
    def test_{cls.lower()}_instantiation(self):
        """Test {cls} can be instantiated."""
        # TODO: Add proper instantiation test
        pass
'''
    
    # Add integration test
    test_content += f'''
    def test_{module_name}_integration(self):
        """Integration test for {module_name} module."""
        # TODO: Add integration test
        pass
'''
    
    # Write test file
    test_file = output_dir / f'test_{module_name}.py'
    if not test_file.exists():
        with open(test_file, 'w') as f:
            f.write(test_content)
        print(f"Generated test stub: {test_file}")
    else:
        print(f"Test file already exists: {test_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate test stubs for low-coverage modules')
    parser.add_argument('--files', nargs='+', help='Files to generate tests for')
    parser.add_argument('--output-dir', default='tests', help='Output directory for tests')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.files:
        for file_path in args.files:
            if os.path.exists(file_path):
                generate_test_stub(file_path, output_dir)
            else:
                print(f"File not found: {file_path}")


if __name__ == '__main__':
    main()