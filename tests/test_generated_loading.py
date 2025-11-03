#!/usr/bin/env python3
"""
Test the fixed data_loading.py to ensure generated grids and predictions load correctly
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loading import DataLoader

def test_generated_grids_loading():
    """Test that generated grids and predictions load correctly"""
    print("🧪 Testing Generated Grids Loading")
    print("=" * 40)
    
    # Initialize DataLoader
    data_loader = DataLoader()
    
    print(f"📁 DataLoader paths:")
    print(f"   Generated dir: {data_loader.generated_dir}")
    print(f"   Directory exists: {os.path.exists(data_loader.generated_dir)}")
    
    # Check available files
    if os.path.exists(data_loader.generated_dir):
        all_files = os.listdir(data_loader.generated_dir)
        grid_files = [f for f in all_files if f.startswith('combined_constraint_free_grids_')]
        pred_files = [f for f in all_files if f.startswith('combined_constraint_free_predictions_')]
        
        print(f"\n📊 Available files:")
        print(f"   Grid files: {len(grid_files)}")
        for f in grid_files:
            print(f"     • {f}")
        print(f"   Prediction files: {len(pred_files)}")
        for f in pred_files:
            print(f"     • {f}")
    
    # Test loading
    print(f"\n🔄 Testing load_generated_grids()...")
    try:
        generated_grids, generated_predictions = data_loader.load_generated_grids()
        
        print(f"\n📊 Loading Results:")
        if generated_grids is not None:
            print(f"   ✅ Grids loaded: {generated_grids.shape}")
            print(f"   ✅ Grid data type: {generated_grids.dtype}")
            print(f"   ✅ Grid value range: [{generated_grids.min()}, {generated_grids.max()}]")
        else:
            print(f"   ❌ No grids loaded")
            
        if generated_predictions is not None:
            print(f"   ✅ Predictions loaded: {generated_predictions.shape}")
            print(f"   ✅ Prediction data type: {generated_predictions.dtype}")
            print(f"   ✅ Prediction range: [{generated_predictions.min():.4f}, {generated_predictions.max():.4f}]")
            
            # Test minimum scores calculation
            min_scores = np.min(generated_predictions, axis=1)
            print(f"   ✅ Min scores range: [{min_scores.min():.4f}, {min_scores.max():.4f}]")
            print(f"   ✅ Mean min score: {min_scores.mean():.4f}")
            
            # Test threshold filtering
            threshold = 0.75
            valid_mask = min_scores >= threshold
            valid_count = np.sum(valid_mask)
            print(f"   ✅ Valid grids (>={threshold}): {valid_count}/{len(min_scores)} ({valid_count/len(min_scores)*100:.1f}%)")
            
        else:
            print(f"   ❌ No predictions loaded")
            
    except Exception as e:
        print(f"   ❌ Error during loading: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_generated_grids_loading()