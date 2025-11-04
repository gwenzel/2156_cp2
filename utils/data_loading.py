"""
Data loading utilities for submission optimization
"""
import numpy as np
import pandas as pd
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handle loading of generated grids, original grids, and oracle predictions"""
    
    def __init__(self, data_dir='data'):
        # Get the parent directory (project root) from utils folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Construct paths relative to project root
        self.data_dir = os.path.join(project_root, data_dir)
        self.submissions_dir = os.path.join(self.data_dir, 'submissions')
        self.generated_dir = os.path.join(self.data_dir, 'generated_grids')
        self.project_root = project_root
        
    def load_generated_grids(self):
        """Load the most recent generated grids and predictions"""
        generated_grids = None
        generated_predictions = None
        
        try:
            # Find the most recent generated grids file
            if not os.path.exists(self.generated_dir):
                print(f"⚠️  Generated grids directory not found: {self.generated_dir}")
                return None, None
                
            generated_files = [f for f in os.listdir(self.generated_dir) 
                             if f.startswith('template_based_grids_') and f.endswith('.npy')]
            
            if not generated_files:
                print("⚠️  No generated grids files found")
                return None, None
                
            latest_file = sorted(generated_files)[-1]
            latest_file_path = os.path.join(self.generated_dir, latest_file)
            print(f"   Loading generated: {latest_file_path}")
            
            generated_grids = np.load(latest_file_path)
            
            # Try to load corresponding predictions
            predictions_file = latest_file_path.replace('template_based_grids_', 'template_scores_')
            if os.path.exists(predictions_file):
                generated_predictions = np.load(predictions_file)
                print(f"✅ Loaded predictions from: {os.path.basename(predictions_file)}")
            else:
                print(f"⚠️  Predictions file not found: {predictions_file}")
                
            print(f"✅ Loaded {len(generated_grids):,} generated grids")
            if generated_predictions is not None:
                print(f"✅ Loaded predictions: {generated_predictions.shape}")
                
        except Exception as e:
            print(f"❌ Error loading generated grids: {e}")
            
        return generated_grids, generated_predictions
    
    def load_original_grids(self):
        """Load original grids and oracle predictions"""
        original_grids = None
        original_predictions = None
        
        try:
            # Load original grid data (construct paths relative to project root)
            grid_files = [
                os.path.join(self.project_root, '2155-Challenge-Problem-2/datasets/grids_0.npy'), 
                os.path.join(self.project_root, '2155-Challenge-Problem-2/datasets/grids_1.npy'), 
                os.path.join(self.project_root, '2155-Challenge-Problem-2/datasets/grids_2.npy'), 
                os.path.join(self.project_root, '2155-Challenge-Problem-2/datasets/grids_3.npy'), 
                os.path.join(self.project_root, '2155-Challenge-Problem-2/datasets/grids_4.npy')
            ]
            
            existing_files = [f for f in grid_files if os.path.exists(f)]
            if not existing_files:
                print("⚠️  No original grid files found")
                return None, None
                
            grid_data_chunks = [np.load(file) for file in existing_files]
            original_grids = np.vstack(grid_data_chunks)
            print(f"✅ Loaded {len(original_grids):,} original grids from {len(existing_files)} files")
            
            # Load oracle predictions
            if os.path.exists(self.submissions_dir):
                oracle_files = [f for f in os.listdir(self.submissions_dir) 
                               if f.endswith('_oracle_predictions_full.npy')]
                
                if oracle_files:
                    latest_oracle_file = sorted(oracle_files)[-1]
                    oracle_path = os.path.join(self.submissions_dir, latest_oracle_file)
                    original_predictions = np.load(oracle_path)
                    print(f"✅ Loaded oracle predictions: {original_predictions.shape}")
                else:
                    print("⚠️  No oracle predictions found")
            else:
                print(f"⚠️  Submissions directory not found: {self.submissions_dir}")
                
        except Exception as e:
            print(f"❌ Error loading original grids: {e}")
            
        return original_grids, original_predictions
    
    def create_combined_pool(self, generated_grids=None, generated_predictions=None,
                           original_grids=None, original_predictions=None, 
                           max_original=10000):
        """Combine generated and original grids into a single candidate pool"""
        
        all_grids = []
        all_predictions = []
        source_labels = []
        
        print(f"🔄 Creating combined candidate pool...")
        
        # Add generated grids if available
        if generated_grids is not None and generated_predictions is not None:
            all_grids.append(generated_grids)
            all_predictions.append(generated_predictions)
            source_labels.extend(['generated'] * len(generated_grids))
            print(f"   Added {len(generated_grids):,} generated grids")
        
        # Add top original grids if available
        if original_grids is not None and original_predictions is not None:
            # Filter original grids to only include high-scoring ones
            original_min_scores = np.min(original_predictions, axis=1)
            
            # Take top scoring original grids
            n_original_to_include = min(max_original, len(original_grids))
            top_original_indices = np.argsort(original_min_scores)[-n_original_to_include:]
            
            top_original_grids = original_grids[top_original_indices]
            top_original_predictions = original_predictions[top_original_indices]
            
            all_grids.append(top_original_grids)
            all_predictions.append(top_original_predictions)
            source_labels.extend(['original'] * len(top_original_grids))
            print(f"   Added {len(top_original_grids):,} top original grids")
        
        # Combine all candidates
        if all_grids:
            combined_grids = np.vstack(all_grids)
            combined_predictions = np.vstack(all_predictions)
            source_labels = np.array(source_labels)
            
            print(f"✅ Combined candidate pool: {len(combined_grids):,} grids total")
            
            # Show composition of candidate pool
            source_counts = Counter(source_labels)
            print(f"   Composition:")
            for source, count in source_counts.items():
                percentage = (count / len(combined_grids)) * 100
                print(f"     • {source.capitalize()}: {count:,} grids ({percentage:.1f}%)")
            
            return combined_grids, combined_predictions, source_labels
        else:
            print("❌ No candidate grids available")
            return None, None, None
    
    def create_synthetic_pool(self, n_synthetic=5000, seed=42):
        """Create synthetic data for demonstration purposes"""
        print("❌ No candidate grids available - creating synthetic example...")
        np.random.seed(seed)
        
        combined_grids = np.random.randint(0, 5, (n_synthetic, 7, 7))
        combined_predictions = np.random.beta(3, 1, (n_synthetic, 4)) * 0.4 + 0.6
        source_labels = np.array(['synthetic'] * n_synthetic)
        
        print(f"📊 Created {n_synthetic} synthetic grids for demo")
        return combined_grids, combined_predictions, source_labels


def filter_by_safety_threshold(combined_grids, combined_predictions, source_labels, 
                              safety_thresholds=[0.75, 0.80, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]):
    """Apply safety margin filtering to combined pool"""
    
    min_scores = np.min(combined_predictions, axis=1)
    
    print(f"📊 COMBINED CANDIDATE ANALYSIS:")
    print(f"Total candidate grids: {len(combined_grids):,}")
    print(f"Min score range: {np.min(min_scores):.4f} - {np.max(min_scores):.4f}")
    print(f"Mean min score: {np.mean(min_scores):.4f}")
    
    print(f"\n🛡️ SAFETY MARGIN FILTERING (COMBINED POOL):")
    print(f"{'Threshold':<12} {'Valid Grids':<15} {'Percentage':<12} {'Original':<12} {'Generated':<12}")
    print("-" * 75)
    
    valid_grids_by_threshold = {}
    for threshold in safety_thresholds:
        valid_mask = min_scores >= threshold
        n_valid = np.sum(valid_mask)
        percentage = (n_valid / len(min_scores)) * 100
        
        # Break down by source
        valid_sources = source_labels[valid_mask]
        source_breakdown = Counter(valid_sources)
        n_original = source_breakdown.get('original', 0)
        n_generated = source_breakdown.get('generated', 0)
        
        valid_grids_by_threshold[threshold] = {
            'mask': valid_mask,
            'grids': combined_grids[valid_mask],
            'predictions': combined_predictions[valid_mask],
            'sources': valid_sources,
            'count': n_valid
        }
        
        print(f"{threshold:<12.2f} {n_valid:<15,} {percentage:<12.1f}% {n_original:<12,} {n_generated:<12,}")
    
    return valid_grids_by_threshold, min_scores


def select_optimal_threshold(valid_grids_by_threshold, default_threshold=0.83):
    """Choose optimal threshold balancing safety and pool size"""
    
    optimal_threshold = default_threshold
    
    if valid_grids_by_threshold[0.87]['count'] >= 240:
        optimal_threshold = 0.87
        print(f"\n✅ Using higher threshold 0.87 (sufficient candidates: {valid_grids_by_threshold[0.87]['count']:,})")
    elif valid_grids_by_threshold[0.89]['count'] >= 130:
        optimal_threshold = 0.89
        print(f"\n✅ Using highest threshold 0.89 (sufficient candidates: {valid_grids_by_threshold[0.89]['count']:,})")
    else:
        print(f"\n✅ Using standard threshold {default_threshold} for good pool size")
    
    candidate_data = valid_grids_by_threshold[optimal_threshold]
    candidate_grids = candidate_data['grids']
    candidate_predictions = candidate_data['predictions'] 
    candidate_sources = candidate_data['sources']
    candidate_count = len(candidate_grids)
    
    print(f"\n✅ FINAL CANDIDATE SELECTION:")
    print(f"Threshold: {optimal_threshold:.2f}")
    print(f"Total candidates: {candidate_count:,} grids")
    
    # Show source breakdown of final candidates
    final_source_counts = Counter(candidate_sources)
    for source, count in final_source_counts.items():
        percentage = (count / candidate_count) * 100
        print(f"   • {source.capitalize()}: {count:,} grids ({percentage:.1f}%)")
    
    return optimal_threshold, candidate_grids, candidate_predictions, candidate_sources