"""
Alternative penalty implementation that preserves validation thresholds
"""
import numpy as np


def smart_penalty_system(predictions, labels, original_threshold=0.85, validation_threshold=0.78):
    """
    Apply smart penalty that ensures augmented grids still meet validation requirements
    
    Args:
        predictions: Original prediction scores
        labels: Grid labels indicating original vs augmented
        original_threshold: Threshold used for original filtering
        validation_threshold: Minimum threshold for final validation
    
    Returns:
        penalized_predictions: Adjusted predictions with smart penalty
    """
    
    penalized_predictions = predictions.copy()
    
    for i, label in enumerate(labels):
        if not label.endswith('_original'):  # This is an augmented grid
            # Calculate max penalty that keeps grid above validation threshold
            min_score = np.min(predictions[i])
            
            if min_score > validation_threshold:
                # Calculate safe penalty percentage
                max_penalty = (min_score - validation_threshold) / min_score
                # Apply smaller of 10% or safe penalty
                penalty_rate = min(0.1, max_penalty * 0.9)  # 90% of safe penalty for buffer
                penalized_predictions[i] = predictions[i] * (1 - penalty_rate)
            else:
                # Grid already close to threshold, apply minimal penalty
                penalized_predictions[i] = predictions[i] * 0.995  # 0.5% penalty only
    
    return penalized_predictions


# Example usage in grid_augmentation.py modification:
"""
# In augment_grids_batch method, replace the penalty logic with:

if selected_predictions is not None:
    if transform_name == 'original':
        augmented_predictions[idx] = predictions[i]
    else:
        # Apply smart penalty that preserves validation threshold
        penalty_rate = min(0.1, (np.min(predictions[i]) - 0.75) / np.min(predictions[i]) * 0.9)
        penalty_rate = max(0.005, penalty_rate)  # Minimum 0.5% penalty
        augmented_predictions[idx] = predictions[i] * (1 - penalty_rate)
"""

def analyze_penalty_impact(original_predictions, penalized_predictions, labels, thresholds=[0.70, 0.75, 0.78, 0.80]):
    """Analyze the impact of penalty system on score distributions"""
    
    print("🔍 PENALTY IMPACT ANALYSIS")
    print("=" * 40)
    
    original_mask = np.array([label.endswith('_original') for label in labels])
    
    print(f"Original grids: {np.sum(original_mask)}")
    print(f"Augmented grids: {np.sum(~original_mask)}")
    
    print(f"\n📊 SCORE COMPARISON:")
    for threshold in thresholds:
        orig_above = np.sum(np.min(original_predictions[original_mask], axis=1) >= threshold) if np.any(original_mask) else 0
        pen_above = np.sum(np.min(penalized_predictions, axis=1) >= threshold)
        
        print(f"≥{threshold:.2f}: Before penalty {orig_above}, After penalty {pen_above}")
    
    if np.any(~original_mask):
        aug_original = np.mean(np.min(original_predictions[~original_mask], axis=1))
        aug_penalized = np.mean(np.min(penalized_predictions[~original_mask], axis=1))
        penalty_pct = (1 - aug_penalized/aug_original) * 100
        
        print(f"\n⚖️ AUGMENTED GRID PENALTY:")
        print(f"Average penalty applied: {penalty_pct:.1f}%")
        print(f"Mean score change: {aug_original:.4f} → {aug_penalized:.4f}")


if __name__ == "__main__":
    # Test the smart penalty system
    test_predictions = np.array([
        [0.90, 0.85, 0.88, 0.87],  # High quality
        [0.82, 0.79, 0.81, 0.80],  # Medium quality  
        [0.76, 0.75, 0.77, 0.78],  # Near threshold
    ])
    
    test_labels = ['grid_0_original', 'grid_1_rot_90', 'grid_2_reflect_h']
    
    penalized = smart_penalty_system(test_predictions, test_labels)
    analyze_penalty_impact(test_predictions, penalized, test_labels)