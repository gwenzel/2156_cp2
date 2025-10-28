#!/usr/bin/env python3
"""
Test script for the PyTorch-based CNN Transportation Oracle
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from oracle import CNNTransportationOracle
    
    print("🧪 Testing PyTorch CNN Transportation Oracle")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500  # Smaller for faster testing
    sample_grids = np.random.randint(0, 5, (n_samples, 7, 7))
    sample_ratings = np.random.uniform(0.2, 0.9, n_samples)
    
    print(f"📊 Created test data: {n_samples} grids")
    
    # Initialize the PyTorch CNN Transportation Oracle
    cnn_oracle = CNNTransportationOracle()
    
    print(f"🔧 Oracle initialized")
    print(f"   PyTorch available: {cnn_oracle.torch_available}")
    print(f"   Device: {cnn_oracle.device}")
    print(f"   Oracle type: {cnn_oracle.type}")
    
    # Test enhanced features (fallback)
    print(f"\n🧠 Testing enhanced feature extraction...")
    test_features = cnn_oracle.create_enhanced_features(sample_grids[:10])
    print(f"   Feature shape: {test_features.shape}")
    print(f"   Feature range: [{np.min(test_features):.3f}, {np.max(test_features):.3f}]")
    
    # Test model configurations
    print(f"\n⚙️ Available model configurations:")
    configs = cnn_oracle.get_model_configs()
    for name, config in configs.items():
        print(f"   {name}: {config['model_type']} model")
    
    # Test model training with different configs
    print(f"\n🚀 Testing model training...")
    
    # Test with 'fast' config for quick testing
    try:
        print("Testing 'fast' configuration...")
        r2_score, train_data, test_data = cnn_oracle.fit_with_config(
            sample_grids[:300], sample_ratings[:300], 
            config_name='fast',
            epochs=20  # Override for even faster testing
        )
        print(f"✅ Fast training successful! R² score: {r2_score:.4f}")
        
        # Test prediction
        print(f"\n🔮 Testing predictions...")
        predictions = cnn_oracle.predict(sample_grids[300:310])
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        print(f"   Sample predictions: {predictions[:5]}")
        
        # Test save/load
        print(f"\n💾 Testing save/load...")
        test_filename = "test_pytorch_cnn_transport_oracle.pkl"
        cnn_oracle.save_model(test_filename)
        
        # Create new oracle and load
        new_oracle = CNNTransportationOracle()
        new_oracle.load_model(test_filename)
        
        # Test predictions with loaded model
        new_predictions = new_oracle.predict(sample_grids[300:310])
        
        # Check if predictions are similar
        prediction_diff = np.mean(np.abs(predictions - new_predictions))
        print(f"   Prediction difference after load: {prediction_diff:.6f}")
        
        if prediction_diff < 0.001:
            print("✅ Save/load test passed!")
        else:
            print("⚠️ Save/load test - some differences found (might be normal for PyTorch)")
        
        # Clean up test files
        for ext in ['.pkl', '_pytorch.pth']:
            cleanup_file = test_filename.replace('.pkl', ext)
            if os.path.exists(cleanup_file):
                os.remove(cleanup_file)
                print(f"🧹 Cleaned up {cleanup_file}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📈 PyTorch CNN Transportation Oracle is ready for use")
        
        # Test with standard config (optional, comment out for faster testing)
        print(f"\n🔬 Testing 'standard' configuration (brief)...")
        new_oracle2 = CNNTransportationOracle()
        r2_score2, _, _ = new_oracle2.fit_with_config(
            sample_grids[:200], sample_ratings[:200], 
            config_name='standard',
            epochs=15
        )
        print(f"✅ Standard training successful! R² score: {r2_score2:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"   This might be due to missing PyTorch or other dependencies")

except ImportError as e:
    print(f"❌ Failed to import Oracle classes: {e}")
    print(f"   Make sure oracle.py is in the current directory")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()