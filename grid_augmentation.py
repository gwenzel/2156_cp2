"""
Grid augmentation utilities for creating rotations and reflections
"""
import numpy as np


def rotate_90(grid):
    """Rotate grid 90 degrees clockwise"""
    return np.rot90(grid, k=-1)


def rotate_180(grid):
    """Rotate grid 180 degrees"""
    return np.rot90(grid, k=2)


def rotate_270(grid):
    """Rotate grid 270 degrees clockwise (90 degrees counterclockwise)"""
    return np.rot90(grid, k=1)


def reflect_horizontal(grid):
    """Reflect grid horizontally (flip left-right)"""
    return np.fliplr(grid)


def reflect_vertical(grid):
    """Reflect grid vertically (flip up-down)"""
    return np.flipud(grid)


def reflect_diagonal(grid):
    """Reflect grid along main diagonal (transpose)"""
    return grid.T


def reflect_anti_diagonal(grid):
    """Reflect grid along anti-diagonal"""
    return np.rot90(np.fliplr(grid))


class GridAugmenter:
    """
    Generate all rotations and reflections of grids
    
    Preserves original scores for all transformations. Selection preference
    for original vs augmented grids is handled in the optimization algorithms.
    """
    
    def __init__(self):
        self.transformations = {
            'original': lambda x: x,
            'rot_90': rotate_90,
            'rot_180': rotate_180, 
            'rot_270': rotate_270,
            'reflect_h': reflect_horizontal,
            'reflect_v': reflect_vertical,
            'reflect_d': reflect_diagonal,
            'reflect_ad': reflect_anti_diagonal
        }
    
    def get_all_transformations(self, grid):
        """Get all 8 transformations of a grid (D4 dihedral group)"""
        transformations = []
        labels = []
        
        for name, transform in self.transformations.items():
            transformed = transform(grid)
            transformations.append(transformed)
            labels.append(name)
            
        return np.array(transformations), labels
    
    def augment_grids_batch(self, grids, predictions=None, include_original=True):
        """
        Augment a batch of grids with all rotations and reflections
        
        Parameters:
        - grids: Array of grids to augment (N, H, W)
        - predictions: Optional predictions to replicate (N, num_advisors)
        - include_original: Whether to include original grids
        
        Returns:
        - augmented_grids: All augmented grids
        - augmented_predictions: Replicated predictions (if provided)
        - augmented_labels: Source labels for tracking
        """
        n_grids = len(grids)
        n_transforms = len(self.transformations) if include_original else len(self.transformations) - 1
        
        # Pre-allocate arrays
        augmented_grids = np.zeros((n_grids * n_transforms, 7, 7), dtype=grids.dtype)
        augmented_labels = []
        
        if predictions is not None:
            augmented_predictions = np.zeros((n_grids * n_transforms, predictions.shape[1]), dtype=predictions.dtype)
        else:
            augmented_predictions = None
        
        print(f"🔄 Augmenting {n_grids:,} grids with {n_transforms} transformations each...")
        
        idx = 0
        transforms_to_use = list(self.transformations.items())
        if not include_original:
            transforms_to_use = transforms_to_use[1:]  # Skip 'original'
        
        for i, grid in enumerate(grids):
            for transform_name, transform_func in transforms_to_use:
                # Apply transformation
                transformed_grid = transform_func(grid)
                augmented_grids[idx] = transformed_grid
                augmented_labels.append(f"grid_{i}_{transform_name}")
                
                # Replicate predictions without penalty
                if predictions is not None:
                    augmented_predictions[idx] = predictions[i]
                
                idx += 1
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1:,}/{n_grids:,} grids...")
        
        print(f"✅ Generated {len(augmented_grids):,} augmented grids")
        
        return augmented_grids, augmented_predictions, np.array(augmented_labels)
    
    def remove_duplicates(self, grids, predictions=None, labels=None):
        """Remove duplicate grids from augmented set"""
        print("🔍 Removing duplicate grids from augmented set...")
        
        n_original = len(grids)
        
        # Find unique grids
        unique_grids = []
        unique_indices = []
        
        for i, grid in enumerate(grids):
            is_duplicate = False
            for j, unique_grid in enumerate(unique_grids):
                if np.array_equal(grid, unique_grid):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_grids.append(grid)
                unique_indices.append(i)
            
            if (i + 1) % 5000 == 0:
                print(f"   Checked {i + 1:,}/{n_original:,} grids...")
        
        unique_grids = np.array(unique_grids)
        unique_indices = np.array(unique_indices)
        
        unique_predictions = None
        unique_labels = None
        
        if predictions is not None:
            unique_predictions = predictions[unique_indices]
        
        if labels is not None:
            unique_labels = labels[unique_indices]
        
        print(f"✅ Removed {n_original - len(unique_grids):,} duplicates")
        print(f"   Unique grids: {len(unique_grids):,} ({len(unique_grids)/n_original*100:.1f}%)")
        
        return unique_grids, unique_predictions, unique_labels
    
    def smart_augment(self, grids, predictions=None, max_augmented=50000, oracle_predictor=None):
        """
        Smart augmentation that selects best performing transformations
        and limits total count. Can optionally use oracle predictor to calculate
        fresh predictions for augmented grids.
        
        Args:
            grids: Input grids to augment
            predictions: Original predictions (used for selection only)
            max_augmented: Maximum number of augmented grids to generate
            oracle_predictor: Optional PreTrainedOraclePredictor to calculate fresh predictions
        
        Returns:
            augmented_grids, augmented_predictions, augmented_labels
        """
        print(f"🧠 Smart augmentation (max {max_augmented:,} grids)...")
        use_oracle_predictions = oracle_predictor is not None
        
        if predictions is not None:
            min_scores = np.min(predictions, axis=1)
            # Sort by performance, take top performers for augmentation
            top_indices = np.argsort(min_scores)[-len(grids):]
            selected_grids = grids[top_indices]
            selected_predictions = predictions[top_indices]
        else:
            # Random selection if no predictions
            n_select = min(len(grids), max_augmented // 4)  # Assume ~4 good transforms per grid
            indices = np.random.choice(len(grids), n_select, replace=False)
            selected_grids = grids[indices]
            selected_predictions = predictions[indices] if predictions is not None else None
        
        # Apply selective transformations (skip some that might create poor grids)
        good_transforms = ['original', 'rot_90', 'rot_180', 'rot_270', 'reflect_h', 'reflect_v']
        
        augmented_grids = []
        augmented_labels = []
        
        for i, grid in enumerate(selected_grids):
            for transform_name in good_transforms:
                if len(augmented_grids) >= max_augmented:
                    break
                    
                transform_func = self.transformations[transform_name]
                transformed_grid = transform_func(grid)
                
                augmented_grids.append(transformed_grid)
                augmented_labels.append(f"smart_{i}_{transform_name}")
            
            if len(augmented_grids) >= max_augmented:
                break
        
        augmented_grids = np.array(augmented_grids)
        augmented_labels = np.array(augmented_labels)
        
        # Calculate predictions for augmented grids
        if use_oracle_predictions:
            print(f"🔮 Calculating fresh oracle predictions for {len(augmented_grids):,} augmented grids...")
            augmented_predictions = oracle_predictor.predict_all_advisors(augmented_grids)
            print(f"✅ Fresh oracle predictions calculated!")
        else:
            # Fallback: replicate original predictions (old behavior)
            print(f"📋 Using replicated predictions (no oracle predictor provided)")
            augmented_predictions = []
            grid_idx = 0
            for i, grid in enumerate(selected_grids):
                for transform_name in good_transforms:
                    if grid_idx >= len(augmented_grids):
                        break
                    if selected_predictions is not None:
                        augmented_predictions.append(selected_predictions[i])
                    grid_idx += 1
                if grid_idx >= len(augmented_grids):
                    break
            augmented_predictions = np.array(augmented_predictions) if augmented_predictions else None
        
        print(f"✅ Smart augmentation complete: {len(augmented_grids):,} grids")
        if augmented_predictions is not None:
            print(f"   Predictions shape: {augmented_predictions.shape}")
            if use_oracle_predictions:
                print(f"   Using fresh oracle predictions ✨")
            else:
                print(f"   Using replicated original predictions")
        
        return augmented_grids, augmented_predictions, augmented_labels