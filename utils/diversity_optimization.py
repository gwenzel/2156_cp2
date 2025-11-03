import numpy as np
from itertools import combinations
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DiversityOptimizer:
    """Efficient diversity optimization using Hamming distances"""
    
    def __init__(self, grids):
        self.grids = grids
        self.n_grids = len(grids)
        self.grid_size = 7 * 7
        
        # Pre-flatten grids for efficient computation
        self.flat_grids = grids.reshape(self.n_grids, -1)
        
        print(f"🔧 Initialized optimizer for {self.n_grids:,} grids")
    
    def hamming_distance(self, grid1_idx, grid2_idx):
        """Calculate Hamming distance between two grids"""
        return np.sum(self.flat_grids[grid1_idx] != self.flat_grids[grid2_idx])
    
    def calculate_pairwise_distances(self, grid_indices=None):
        """Calculate all pairwise Hamming distances for given indices"""
        if grid_indices is None:
            grid_indices = list(range(self.n_grids))
        
        n_indices = len(grid_indices)
        distances = np.zeros((n_indices, n_indices))
        
        print(f"📊 Calculating pairwise distances for {n_indices} grids...")
        
        for i in tqdm(range(n_indices), desc="Computing distances"):
            for j in range(i + 1, n_indices):
                dist = self.hamming_distance(grid_indices[i], grid_indices[j])
                distances[i, j] = distances[j, i] = dist
        
        return distances, grid_indices
    
    def mean_pairwise_distance(self, grid_indices):
        """Calculate mean pairwise distance for a set of grids"""
        if len(grid_indices) < 2:
            return 0
        
        total_distance = 0
        n_pairs = 0
        
        for i in range(len(grid_indices)):
            for j in range(i + 1, len(grid_indices)):
                total_distance += self.hamming_distance(grid_indices[i], grid_indices[j])
                n_pairs += 1
        
        return total_distance / n_pairs if n_pairs > 0 else 0
    
    def min_distance_to_set(self, candidate_idx, selected_indices):
        """Calculate minimum distance from candidate to any grid in selected set"""
        if not selected_indices:
            return float('inf')
        
        return min(self.hamming_distance(candidate_idx, selected_idx) 
                   for selected_idx in selected_indices)
    
    def greedy_selection(self, target_size=100, initial_size=None):
        """Improved greedy selection algorithm for maximum diversity"""
        print(f"\n🧠 IMPROVED GREEDY SELECTION ALGORITHM")
        print(f"Target selection size: {target_size}")
        
        if initial_size is None:
            # Use all available grids if pool is small enough
            if self.n_grids <= 5000:
                candidate_indices = list(range(self.n_grids))
            else:
                # Random sample for efficiency
                initial_size = min(2000, self.n_grids)
                candidate_indices = random.sample(range(self.n_grids), initial_size)
        else:
            candidate_indices = random.sample(range(self.n_grids), initial_size)
        
        print(f"Candidate pool size: {len(candidate_indices)}")
        
        # Start with the most diverse pair
        print("🔍 Finding initial most diverse pair...")
        max_distance = 0
        best_pair = None
        
        # Sample pairs to find good starting point
        n_samples = min(10000, len(candidate_indices) * (len(candidate_indices) - 1) // 2)
        sampled_pairs = random.sample(list(combinations(candidate_indices, 2)), n_samples)
        
        for i, j in tqdm(sampled_pairs, desc="Finding initial pair"):
            dist = self.hamming_distance(i, j)
            if dist > max_distance:
                max_distance = dist
                best_pair = (i, j)
        
        selected_indices = list(best_pair)
        remaining_indices = [idx for idx in candidate_indices if idx not in selected_indices]
        
        print(f"✅ Initial pair distance: {max_distance}")
        print(f"🔄 Iterative selection for remaining {target_size - 2} grids...")
        
        # Iteratively add grids that maximize minimum distance to selected set
        for iteration in tqdm(range(target_size - 2), desc="Greedy selection"):
            best_candidate = None
            best_min_distance = -1
            
            # Try adding each remaining candidate - find one with max min distance
            candidates_to_try = random.sample(remaining_indices, 
                                            min(200, len(remaining_indices)))  # Limit for speed
            
            for candidate in candidates_to_try:
                # Calculate minimum distance to any selected grid
                min_dist = self.min_distance_to_set(candidate, selected_indices)
                
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_candidate = candidate
            
            # Add the best candidate (always add one, even if distance is small)
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
            else:
                # Fallback: just add a random remaining candidate
                if remaining_indices:
                    fallback = random.choice(remaining_indices)
                    selected_indices.append(fallback)
                    remaining_indices.remove(fallback)
                    print(f"⚠️  Fallback selection at iteration {iteration + 1}")
        
        final_mean_distance = self.mean_pairwise_distance(selected_indices)
        
        print(f"\n🎯 GREEDY SELECTION COMPLETE!")
        print(f"Selected {len(selected_indices)} grids")
        print(f"Final mean pairwise distance: {final_mean_distance:.2f}")
        print(f"Max possible distance: {self.grid_size} (all different)")
        print(f"Diversity score: {(final_mean_distance / self.grid_size * 100):.1f}%")

        return selected_indices, final_mean_distance
    
    def hill_climb_improve(self, selected_indices, max_iterations=1000, temperature_start=10.0):
        """Hill climbing improvement with simulated annealing to refine greedy selection"""
        print(f"\n🧗 HILL CLIMBING IMPROVEMENT")
        print(f"Starting selection size: {len(selected_indices)}")
        print(f"Max iterations: {max_iterations}")
        
        current_selection = selected_indices.copy()
        current_score = self.mean_pairwise_distance(current_selection)
        best_selection = current_selection.copy()
        best_score = current_score
        
        # Available indices not in current selection
        available_indices = [i for i in range(self.n_grids) if i not in current_selection]
        
        improvements = 0
        no_improvement_count = 0
        
        print(f"Initial score: {current_score:.2f}")
        
        for iteration in range(max_iterations):
            # Simulated annealing temperature decay
            temperature = temperature_start * (0.95 ** (iteration // 50))
            
            # Try swapping a random grid from selection with a random available grid
            if len(available_indices) == 0:
                break
                
            # Choose random indices to swap
            remove_idx = random.choice(current_selection)
            add_idx = random.choice(available_indices)
            
            # Create new selection
            new_selection = current_selection.copy()
            new_selection.remove(remove_idx)
            new_selection.append(add_idx)
            
            # Calculate new score
            new_score = self.mean_pairwise_distance(new_selection)
            
            # Accept improvement or with probability based on temperature
            score_diff = new_score - current_score
            accept = False
            
            if score_diff > 0:
                # Improvement - always accept
                accept = True
                improvements += 1
                no_improvement_count = 0
            elif temperature > 0:
                # Worse solution - accept with probability
                probability = np.exp(score_diff / temperature)
                if random.random() < probability:
                    accept = True
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            if accept:
                # Update current solution
                current_selection = new_selection
                current_score = new_score
                
                # Update available indices
                available_indices.remove(add_idx)
                available_indices.append(remove_idx)
                
                # Track best solution
                if current_score > best_score:
                    best_selection = current_selection.copy()
                    best_score = current_score
            
            # Early stopping if no improvements for a while
            if no_improvement_count > 200:
                print(f"   Early stopping at iteration {iteration + 1} (no improvements)")
                break
            
            # Progress reporting
            if (iteration + 1) % 100 == 0:
                print(f"   Iteration {iteration + 1}: Current={current_score:.3f}, "
                      f"Best={best_score:.3f}, Temp={temperature:.3f}")
        
        improvement = best_score - self.mean_pairwise_distance(selected_indices)
        improvement_pct = (improvement / self.mean_pairwise_distance(selected_indices)) * 100
        
        print(f"\n🎯 HILL CLIMBING COMPLETE!")
        print(f"Original score: {self.mean_pairwise_distance(selected_indices):.3f}")
        print(f"Improved score: {best_score:.3f}")
        print(f"Improvement: +{improvement:.3f} ({improvement_pct:+.2f}%)")
        print(f"Total improvements made: {improvements}")
        
        return best_selection, best_score
    
    def random_baseline(self, target_size=100, n_trials=10):
        """Random selection baseline for comparison"""
        print(f"\n📊 RANDOM BASELINE ({n_trials} trials)")
        
        best_score = 0
        best_selection = None
        
        for trial in range(n_trials):
            random_indices = random.sample(range(self.n_grids), target_size)
            score = self.mean_pairwise_distance(random_indices)
            
            if score > best_score:
                best_score = score
                best_selection = random_indices
        
        print(f"Best random score: {best_score:.2f}")
        print(f"Random diversity: {(best_score / self.grid_size * 100):.1f}%")
        
        return best_selection, best_score
