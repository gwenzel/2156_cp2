#!/usr/bin/env python3
"""
Quick script to verify all notebooks can import from utils correctly
"""
import os
import glob

def check_notebook_imports():
    """Check if notebooks can import from utils"""
    notebook_files = glob.glob("*.ipynb")
    
    print("🔍 Checking import statements in notebooks...")
    
    for notebook in notebook_files:
        print(f"\n📓 Checking: {notebook}")
        
        with open(notebook, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for old import patterns
        old_patterns = [
            'from oracle import',
            'from data_loading import',
            'from grid_augmentation import',
            'from diversity_optimization import',
            'from oracle_predictor import'
        ]
        
        new_patterns = [
            'from utils.oracle import',
            'from utils.data_loading import', 
            'from utils.grid_augmentation import',
            'from utils.diversity_optimization import',
            'from utils.oracle_predictor import'
        ]
        
        for old_pattern in old_patterns:
            if old_pattern in content and f'utils.{old_pattern.split(" ")[1]}' not in content:
                print(f"  ⚠️  Found old import: {old_pattern}")
        
        for new_pattern in new_patterns:
            if new_pattern in content:
                print(f"  ✅ Found updated import: {new_pattern}")
    
    print(f"\n✅ Import check complete!")

if __name__ == "__main__":
    check_notebook_imports()