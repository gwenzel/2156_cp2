"""
Oracle Predictor Module - Reusable Oracle Loading and Prediction Functionality

This module provides a unified interface for loading and using pre-trained oracle models
across different notebooks. It handles the loading of all four advisor models and provides
convenient prediction methods.
"""

import os
import numpy as np

# Try to import oracle classes - handle missing dependencies gracefully
try:
    from oracle import CNNTransportationOracle, CNNWellnessOracle
    from oracle import CNNBusinessOracle, CNNTaxOracle
    ORACLE_CLASSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Oracle classes not available: {e}")
    print("💡 Make sure required dependencies are installed (autogluon, torch, etc.)")
    ORACLE_CLASSES_AVAILABLE = False
    CNNTransportationOracle = None
    CNNWellnessOracle = None
    CNNBusinessOracle = None 
    CNNTaxOracle = None


class PreTrainedOraclePredictor:
    """
    A unified predictor class that manages all pre-trained oracle models
    and provides convenient prediction methods for individual advisors or all advisors.
    """
    
    def __init__(self, loaded_models):
        """
        Initialize the predictor with loaded oracle models
        
        Args:
            loaded_models: Dict mapping advisor names to loaded oracle instances
        """
        self.loaded_models = loaded_models
        self.advisor_names = ["Wellness", "Tax", "Transportation", "Business"]  # Corrected order
        
    def predict_advisor(self, grids, advisor_idx):
        """
        Predict using pre-trained model for specific advisor
        
        Args:
            grids: Array of grids to predict on
            advisor_idx: Index of advisor (0=Wellness, 1=Tax, 2=Transportation, 3=Business)
            
        Returns:
            Array of predictions for the specified advisor
        """
        advisor_name = self.advisor_names[advisor_idx]
        
        if advisor_name in self.loaded_models:
            oracle = self.loaded_models[advisor_name]
            print(f"🔮 Predicting with {advisor_name} Oracle...")
            return oracle.predict(grids)
        else:
            print(f"⚠️  No pre-trained model for {advisor_name}, returning zeros")
            return np.zeros(len(grids))
    
    def predict_all_advisors(self, grids):
        """
        Predict for all advisors using available pre-trained models
        
        Args:
            grids: Array of grids to predict on
            
        Returns:
            Array of shape (n_grids, 4) with predictions for all advisors
        """
        print(f"🚀 Generating predictions for all advisors on {len(grids):,} grids...")
        predictions = []
        for advisor_idx in range(4):
            advisor_predictions = self.predict_advisor(grids, advisor_idx)
            predictions.append(advisor_predictions)
        return np.stack(predictions).T
    
    def get_loaded_advisors(self):
        """Return list of successfully loaded advisor names"""
        return list(self.loaded_models.keys())
    
    def get_advisor_info(self):
        """Return detailed information about loaded advisors"""
        info = {}
        for advisor_name in self.loaded_models.keys():
            oracle = self.loaded_models[advisor_name]
            advisor_id = getattr(oracle, 'advisor_id', 'Unknown')
            architecture = "CityCNN1Plus (deep)" if advisor_id in [0, 2] else "CityCNN1 (standard)"
            info[advisor_name] = {
                'advisor_id': advisor_id,
                'architecture': architecture,
                'model_type': type(oracle).__name__
            }
        return info


def load_oracle_models(model_dir='data/models', verbose=True):
    """
    Load all available pre-trained oracle models from the specified directory
    
    Args:
        model_dir: Directory containing the oracle model files
        verbose: Whether to print loading progress and results
        
    Returns:
        dict: Dictionary mapping advisor names to loaded oracle instances
    """
    
    if not ORACLE_CLASSES_AVAILABLE:
        if verbose:
            print("❌ Oracle classes not available - cannot load models")
        return {}
    
    if verbose:
        print("🔄 Loading pre-trained Oracle models...")
    
    # Define available oracle classes with corrected advisor order
    oracle_classes = {
        'Wellness': CNNWellnessOracle,      # Advisor 0 - uses CityCNN1Plus (deep)
        'Tax': CNNTaxOracle,                # Advisor 1 - uses CityCNN1 (standard)
        'Transportation': CNNTransportationOracle,  # Advisor 2 - uses CityCNN1Plus (deep) 
        'Business': CNNBusinessOracle       # Advisor 3 - uses CityCNN1 (standard)
    }

    # Define oracle model file paths
    oracle_files = {
        'Wellness': os.path.join(model_dir, 'wellness_oracle_model.pkl'),
        'Tax': os.path.join(model_dir, 'tax_oracle_model.pkl'),
        'Transportation': os.path.join(model_dir, 'transportation_oracle_model.pkl'),
        'Business': os.path.join(model_dir, 'business_oracle_model.pkl')
    }

    # Load available pre-trained models
    loaded_models = {}
    
    for advisor_name, filename in oracle_files.items():
        if os.path.exists(filename):
            if verbose:
                print(f"📁 Loading {advisor_name} Oracle from {filename}...")
            
            try:
                # Create oracle instance using the appropriate class
                oracle_class = oracle_classes[advisor_name]
                oracle = oracle_class()
                oracle.load_model(filename)
                
                loaded_models[advisor_name] = oracle
                
                if verbose:
                    print(f"✅ {advisor_name} Oracle loaded successfully!")
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to load {advisor_name} Oracle: {e}")
                    print(f"   Error details: {type(e).__name__}")
        else:
            if verbose:
                print(f"⚠️  {filename} not found - will need fallback for {advisor_name}")

    if verbose:
        print(f"\n📊 Successfully loaded {len(loaded_models)} pre-trained models:")
        for advisor_name in loaded_models.keys():
            oracle = loaded_models[advisor_name]
            advisor_id = getattr(oracle, 'advisor_id', 'Unknown')
            architecture = "CityCNN1Plus (deep)" if advisor_id in [0, 2] else "CityCNN1 (standard)"
            print(f"   ✓ {advisor_name} Oracle (Advisor {advisor_id}) - {architecture}")
    
    return loaded_models


def create_oracle_predictor(model_dir='data/models', verbose=True):
    """
    Convenience function to load oracle models and create a predictor instance
    
    Args:
        model_dir: Directory containing the oracle model files
        verbose: Whether to print loading progress and results
        
    Returns:
        PreTrainedOraclePredictor: Ready-to-use predictor instance
    """
    
    loaded_models = load_oracle_models(model_dir=model_dir, verbose=verbose)
    predictor = PreTrainedOraclePredictor(loaded_models)
    
    if verbose:
        print(f"✅ Pre-trained Oracle predictor ready!")
        print(f"🎯 Can predict for {len(loaded_models)} advisors without training")
        print(f"🧠 Architecture mapping:")
        print(f"   • Wellness (0) & Transportation (2): CityCNN1Plus (deep)")
        print(f"   • Tax (1) & Business (3): CityCNN1 (standard)")
    
    return predictor


# Convenience functions for backward compatibility
def load_oracle_predictor(model_dir='data/models', verbose=True):
    """Alias for create_oracle_predictor for backward compatibility"""
    return create_oracle_predictor(model_dir=model_dir, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Oracle Predictor Module")
    print("=" * 40)
    
    # Load oracle models
    predictor = create_oracle_predictor()
    
    # Show loaded models info
    if predictor.get_loaded_advisors():
        print(f"\n📊 Loaded Advisors: {', '.join(predictor.get_loaded_advisors())}")
        
        # Show detailed info
        advisor_info = predictor.get_advisor_info()
        for name, info in advisor_info.items():
            print(f"   {name}: ID={info['advisor_id']}, {info['architecture']}")
    else:
        print("\n❌ No oracle models were loaded successfully")