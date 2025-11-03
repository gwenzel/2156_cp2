#!/usr/bin/env python3
"""
Test script to verify DataLoader can access files from utils folder
"""
import sys
import os

# Add the project root to path so we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_loading import DataLoader
    
    print("🧪 Testing DataLoader from utils folder...")
    print("=" * 50)
    
    # Initialize DataLoader
    data_loader = DataLoader()
    
    print(f"📁 DataLoader paths:")
    print(f"   Project root: {data_loader.project_root}")
    print(f"   Data dir: {data_loader.data_dir}")
    print(f"   Submissions dir: {data_loader.submissions_dir}")
    print(f"   Generated dir: {data_loader.generated_dir}")
    
    print(f"\n🔍 Checking if directories exist:")
    print(f"   Data dir exists: {os.path.exists(data_loader.data_dir)}")
    print(f"   Submissions dir exists: {os.path.exists(data_loader.submissions_dir)}")
    print(f"   Generated dir exists: {os.path.exists(data_loader.generated_dir)}")
    
    # Check for original grid files
    grid_files_base = [
        '2155-Challenge-Problem-2/datasets/grids_0.npy', 
        '2155-Challenge-Problem-2/datasets/grids_1.npy', 
        '2155-Challenge-Problem-2/datasets/grids_2.npy', 
        '2155-Challenge-Problem-2/datasets/grids_3.npy', 
        '2155-Challenge-Problem-2/datasets/grids_4.npy'
    ]
    
    print(f"\n📊 Checking original grid files:")
    for i, grid_file in enumerate(grid_files_base):
        full_path = os.path.join(data_loader.project_root, grid_file)
        exists = os.path.exists(full_path)
        print(f"   grids_{i}.npy: {exists} - {full_path}")
    
    print(f"\n✅ DataLoader path test complete!")
    
    # Try a quick load test (just check if method works)
    print(f"\n🔄 Testing load methods...")
    try:
        original_grids, original_predictions = data_loader.load_original_grids()
        if original_grids is not None:
            print(f"✅ Original grids loaded successfully: {len(original_grids)} grids")
        else:
            print(f"⚠️  No original grids loaded (expected if no data files)")
            
    except Exception as e:
        print(f"❌ Error loading original grids: {e}")
    
    try:
        generated_grids, generated_predictions = data_loader.load_generated_grids()
        if generated_grids is not None:
            print(f"✅ Generated grids loaded successfully: {len(generated_grids)} grids")
        else:
            print(f"⚠️  No generated grids loaded (expected if no generated files)")
            
    except Exception as e:
        print(f"❌ Error loading generated grids: {e}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the utils folder exists and contains data_loading.py")
except Exception as e:
    print(f"❌ Unexpected error: {e}")