# Setup and Data Loading
import pickle
import os
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist

from autogluon.tabular import TabularPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for city grids with spatial awareness
    """
    
    def __init__(self):
        self.district_names = ['Residential', 'Industrial', 'Commercial', 'Parks', 'Office']
    
    def basic_features(self, grids):
        """Basic features: flatten grids and add district counts"""
        n_grids = len(grids)
        
        # Flatten grids (49 features)
        grids_flat = grids.reshape(n_grids, -1)
        
        # District counts (5 features)
        district_counts = np.zeros((n_grids, 5))
        for i in range(5):
            district_counts[:, i] = np.sum(grids_flat == i, axis=1)
        
        return np.hstack([grids_flat, district_counts])
    
    def spatial_features(self, grids):
        """Advanced spatial features"""
        n_grids = len(grids)
        spatial_features = []
        
        for grid in grids:
            features = []
            
            # 1. Adjacency features (how much districts cluster)
            adjacency_scores = np.zeros(5)
            for district in range(5):
                adjacent_pairs = 0
                total_pairs = 0
                for i in range(7):
                    for j in range(7):
                        if grid[i, j] == district:
                            # Check neighbors
                            neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                            for ni, nj in neighbors:
                                if 0 <= ni < 7 and 0 <= nj < 7:
                                    if grid[ni, nj] == district:
                                        adjacent_pairs += 1
                                    total_pairs += 1
                if total_pairs > 0:
                    adjacency_scores[district] = adjacent_pairs / total_pairs
            features.extend(adjacency_scores)
            
            # 2. Center vs edge preferences
            center_positions = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
            edge_positions = [(i,j) for i in range(7) for j in range(7) if (i,j) not in center_positions]
            
            center_counts = np.zeros(5)
            edge_counts = np.zeros(5)
            
            for i, j in center_positions:
                center_counts[int(grid[i, j])] += 1
            for i, j in edge_positions:
                edge_counts[int(grid[i, j])] += 1
                
            # Normalize by number of positions
            center_ratios = center_counts / len(center_positions)
            edge_ratios = edge_counts / len(edge_positions)
            
            features.extend(center_ratios)
            features.extend(edge_ratios)
            
            # 3. Corner features
            corners = [(0,0), (0,6), (6,0), (6,6)]
            corner_counts = np.zeros(5)
            for i, j in corners:
                corner_counts[int(grid[i, j])] += 1
            features.extend(corner_counts)
            
            # 4. Distance-based features (average distance between same districts)
            for district in range(5):
                positions = np.argwhere(grid == district)
                if len(positions) > 1:
                    distances = cdist(positions, positions)
                    avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                else:
                    avg_distance = 0
                features.append(avg_distance)
            
            # 5. Diversity metrics
            unique_districts = len(np.unique(grid))
            features.append(unique_districts)
            
            # Shannon diversity
            counts = np.bincount(grid.flatten().astype(int), minlength=5)
            probs = counts / counts.sum()
            shannon_div = -np.sum(probs * np.log(probs + 1e-10))
            features.append(shannon_div)
            
            spatial_features.append(features)
        
        return np.array(spatial_features)
    
    def interaction_features(self, grids):
        """District interaction features"""
        n_grids = len(grids)
        interaction_features = []
        
        for grid in grids:
            features = []
            
            # Pairwise district adjacencies
            for d1 in range(5):
                for d2 in range(d1, 5):
                    adjacency_count = 0
                    total_possible = 0
                    
                    for i in range(7):
                        for j in range(7):
                            if grid[i, j] == d1:
                                # Check neighbors for d2
                                neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                                for ni, nj in neighbors:
                                    if 0 <= ni < 7 and 0 <= nj < 7:
                                        total_possible += 1
                                        if grid[ni, nj] == d2:
                                            adjacency_count += 1
                    
                    if total_possible > 0:
                        features.append(adjacency_count / total_possible)
                    else:
                        features.append(0)
            
            interaction_features.append(features)
        
        return np.array(interaction_features)
    
    def economic_features(self, grids):
        """Features that might matter for economic advisors"""
        n_grids = len(grids)
        economic_features = []
        
        for grid in grids:
            features = []
            
            # Commercial-Residential proximity (good for business)
            comm_positions = np.argwhere(grid == 2)  # Commercial
            res_positions = np.argwhere(grid == 0)   # Residential (fixed from 1)
            
            if len(comm_positions) > 0 and len(res_positions) > 0:
                distances = cdist(comm_positions, res_positions)
                avg_comm_res_distance = np.mean(distances.min(axis=1))
            else:
                avg_comm_res_distance = 10  # Large penalty
            features.append(avg_comm_res_distance)
            
            # Industrial-Residential distance (bad if too close)
            ind_positions = np.argwhere(grid == 1)   # Industrial
            if len(ind_positions) > 0 and len(res_positions) > 0:
                distances = cdist(ind_positions, res_positions)
                avg_ind_res_distance = np.mean(distances.min(axis=1))
            else:
                avg_ind_res_distance = 0
            features.append(avg_ind_res_distance)
            
            # Office-Commercial synergy
            office_positions = np.argwhere(grid == 4)  # Office
            if len(office_positions) > 0 and len(comm_positions) > 0:
                distances = cdist(office_positions, comm_positions)
                avg_office_comm_distance = np.mean(distances.min(axis=1))
            else:
                avg_office_comm_distance = 10
            features.append(avg_office_comm_distance)
            
            # Parks accessibility (average distance from residential to parks)
            park_positions = np.argwhere(grid == 3)   # Parks
            if len(park_positions) > 0 and len(res_positions) > 0:
                distances = cdist(res_positions, park_positions)
                avg_res_park_distance = np.mean(distances.min(axis=1))
            else:
                avg_res_park_distance = 10
            features.append(avg_res_park_distance)
            
            economic_features.append(features)
        
        return np.array(economic_features)
    
    def create_all_features(self, grids):
        """Combine all feature types"""
        print("Creating basic features...")
        basic = self.basic_features(grids)
        
        print("Creating spatial features...")
        spatial = self.spatial_features(grids)
        
        print("Creating interaction features...")
        interaction = self.interaction_features(grids)
        
        print("Creating economic features...")
        economic = self.economic_features(grids)
        
        all_features = np.hstack([basic, spatial, interaction, economic])
        
        print(f"Total features created: {all_features.shape[1]}")
        print(f"  - Basic: {basic.shape[1]}")
        print(f"  - Spatial: {spatial.shape[1]}")
        print(f"  - Interaction: {interaction.shape[1]}")
        print(f"  - Economic: {economic.shape[1]}")
        
        return all_features


class BaseOracle:
    """
    Business-focused Oracle with specialized features and models
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.type = ""

    def create_features(self, grids):
        """Create features specifically for advisor"""
        print(f"Creating features for {self.type} Advisor...")
        all_features = self.feature_engineer.create_all_features(grids)
        return all_features

    def fit_model(self, grids, ratings):
        """ Fit model for business advisor using autogluon """

        print(f"\n{'='*50}")
        print(f"Training AutoGluon model for {self.type} Advisor")
        print(f"{'='*50}")

        # Get labeled data

        print(f"Available training samples: {len(grids)}")

        # Create features
        print("Creating features...")
        features = self.create_features(grids)

        # Prepare DataFrame for AutoGluon
        df = pd.DataFrame(features)
        df['target'] = ratings

        # Train test split (keep original 3D grids for feature creation)
        grids_train, grids_test, ratings_train, ratings_test = train_test_split(
            grids, ratings, test_size=0.2, random_state=42)

        # Create features for train and test sets
        train_features = self.create_features(grids_train)
        test_features = self.create_features(grids_test)
        
        train_data = pd.DataFrame(train_features)
        train_data['target'] = ratings_train
        test_data = pd.DataFrame(test_features)
        test_data['target'] = ratings_test

        # Train AutoGluon predictor
        predictor = TabularPredictor(label='target', eval_metric='r2').fit(
            train_data,
            presets='best_quality',
            time_limit=180  # Limit training time to 3 minutes
        )

        # Evaluate on test data
        performance = predictor.evaluate(test_data)

        self.model = predictor

        return performance['r2'], train_data, test_data

    def predict(self, grids):
        """Predict scores for all grids using the trained AutoGluon model"""
        predictor = self.model

        print(f"Creating features for all {len(grids)} grids...")
        features = self.create_features(grids)

        # Prepare DataFrame for prediction
        df = pd.DataFrame(features)

        print(f"Making predictions with AutoGluon model...")
        predictions = predictor.predict(df)

        return predictions.values

    def save_model(self, filename='business_oracle_model.pkl'):
        """ Save the trained model to a file """
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        """ Load the trained model from a file """
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)


class BusinessOracle(BaseOracle):
    """
    Business-focused Oracle wit specialized features and models
    """
    def __init__(self):
        super().__init__()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.type = "Business"


class TaxOracle(BaseOracle):
    """
    Tax-focused Oracle with specialized features and models
    """
    def __init__(self):
        super().__init__()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.type = "Tax"


class WellnessOracle(BaseOracle):
    """
    Wellness-focused Oracle with specialized features and models
    """
    def __init__(self):
        super().__init__()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.type = "Wellness"


class TransportationOracle(BaseOracle):
    """
    Transportation-focused Oracle with specialized features and models
    based on BusinessOracle, but including the specialized features for transportation
    """
    def __init__(self):
        super().__init__()
        self.feature_engineer = AdvancedTransportationFeatures()
        self.type = "Transportation"


# Analyze Oracle Results
def analyze_oracle_results(predictions, threshold=0.75):

    advisor_names = ["Wellness", "Tax", "Transportation", "Business"]

    """Analyze the Oracle's predictions"""
    
    print(f"\n{'='*60}")
    print("ORACLE PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate minimum scores per grid
    min_scores = np.min(predictions, axis=1)
    
    # Find valid grids
    valid_mask = min_scores >= threshold
    n_valid = np.sum(valid_mask)
    
    print(f"\n📊 VALIDITY ANALYSIS:")
    print(f"   • Total grids analyzed: {len(predictions):,}")
    print(f"   • Valid grids (min score ≥ {threshold}): {n_valid:,}")
    print(f"   • Validity rate: {n_valid/len(predictions)*100:.2f}%")
    
    print(f"\n📈 SCORE DISTRIBUTION:")
    for i, advisor in enumerate(advisor_names):
        advisor_scores = predictions[:, i]
        print(f"   • {advisor:15}: Mean={np.mean(advisor_scores):.3f}, "
              f"Std={np.std(advisor_scores):.3f}, "
              f"Valid={np.sum(advisor_scores >= threshold):,}")
    
    print(f"\n🎯 MINIMUM SCORE STATS:")
    print(f"   • Mean minimum score: {np.mean(min_scores):.3f}")
    print(f"   • Std minimum score: {np.std(min_scores):.3f}")
    print(f"   • Max minimum score: {np.max(min_scores):.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Individual advisor score distributions
    axes[0,0].hist([predictions[:, i] for i in range(4)], 
                   bins=50, alpha=0.6, label=advisor_names, density=True)
    axes[0,0].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    axes[0,0].set_xlabel('Predicted Score')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Score Distributions by Advisor')
    axes[0,0].legend()
    
    # Subplot 2: Minimum score distribution
    axes[0,1].hist(min_scores, bins=50, alpha=0.7, color='purple')
    axes[0,1].axvline(threshold, color='red', linestyle='--', label=f'Validity Threshold')
    axes[0,1].axvline(np.mean(min_scores), color='orange', linestyle=':', label='Mean')
    axes[0,1].set_xlabel('Minimum Score Across Advisors')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Minimum Scores')
    axes[0,1].legend()
    
    # Subplot 3: Score correlation matrix
    corr_matrix = np.corrcoef(predictions.T)
    im = axes[1,0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,0].set_xticks(range(4))
    axes[1,0].set_yticks(range(4))
    axes[1,0].set_xticklabels(advisor_names, rotation=45)
    axes[1,0].set_yticklabels(advisor_names)
    axes[1,0].set_title('Advisor Score Correlations')
    
    # Add correlation values to heatmap
    for i in range(4):
        for j in range(4):
            axes[1,0].text(j, i, f'{corr_matrix[i,j]:.3f}', 
                          ha='center', va='center', color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
    
    # Subplot 4: Valid vs Invalid comparison
    if n_valid > 0:
        valid_scores = predictions[valid_mask]
        invalid_scores = predictions[~valid_mask]
        
        x_pos = np.arange(4)
        valid_means = np.mean(valid_scores, axis=0)
        invalid_means = np.mean(invalid_scores, axis=0) if len(invalid_scores) > 0 else np.zeros(4)
        
        width = 0.35
        axes[1,1].bar(x_pos - width/2, valid_means, width, label='Valid Grids', alpha=0.8)
        axes[1,1].bar(x_pos + width/2, invalid_means, width, label='Invalid Grids', alpha=0.8)
        
        axes[1,1].set_xlabel('Advisor')
        axes[1,1].set_ylabel('Average Score')
        axes[1,1].set_title('Valid vs Invalid Grid Scores')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(advisor_names, rotation=45)
        axes[1,1].legend()
        axes[1,1].axhline(threshold, color='red', linestyle='--', alpha=0.7)
    else:
        axes[1,1].text(0.5, 0.5, 'No Valid Grids Found', 
                      transform=axes[1,1].transAxes, ha='center', va='center', fontsize=16)
        axes[1,1].set_title('Valid vs Invalid Grid Scores')
    
    plt.tight_layout()
    plt.show()
    
    return valid_mask, min_scores


# Save Oracle Results and Demonstrate Usage
def save_oracle_results(final_oracle_predictions, min_scores, valid_mask, top_grids, name="simple_oracle"):
    """Save Oracle predictions and models for later use"""
    
    print("Saving Oracle results...")
    
    # Save predictions
    np.save(f'{name}_predictions.npy', final_oracle_predictions)
    np.save(f'{name}_min_scores.npy', min_scores)
    np.save(f'{name}_valid_mask.npy', valid_mask)
    np.save(f'{name}_top_grids.npy', top_grids)
    
    print("✅ Oracle predictions saved")
    
    return True


def get_top_grids(grids, min_scores, oracle_predictions, valid_mask, n_grids=100, method='top_scoring'):
    """Get top grids based on different strategies"""
    
    if method == 'top_scoring':
        # Get grids with highest minimum scores
        top_indices = np.argsort(min_scores)[-n_grids:]
        return grids[top_indices], oracle_predictions[top_indices], top_indices
    
    elif method == 'valid_only' and np.any(valid_mask):
        # Get valid grids only
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) >= n_grids:
            selected_indices = np.random.choice(valid_indices, n_grids, replace=False)
        else:
            print(f"Only {len(valid_indices)} valid grids available, returning all")
            selected_indices = valid_indices
        return grids[selected_indices], oracle_predictions[selected_indices], selected_indices
    
    else:
        print(f"No valid grids found, falling back to top_scoring method")
        return get_top_grids(n_grids, 'top_scoring')

'''
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
'''


class AdvancedTransportationFeatures:
    """
    Ultra-specialized transportation features focusing on connectivity and flow
    """
    
    def __init__(self):
        self.district_names = ['Residential', 'Industrial', 'Commercial', 'Parks', 'Office']
    
    def create_connectivity_matrix_features(self, grids):
        """Create features based on connectivity matrices between district types"""
        n_grids = len(grids)
        features_list = []
        
        for grid in grids:
            features = []
            
            # Get positions for each relevant district type
            res_positions = np.argwhere(grid == 0)   # Residential (origin points)
            ind_positions = np.argwhere(grid == 1)   # Industrial (work destinations)
            comm_positions = np.argwhere(grid == 2)  # Commercial (service destinations)
            office_positions = np.argwhere(grid == 4)  # Office (work destinations)
            
            # Combine work destinations
            work_positions = np.vstack([ind_positions, office_positions]) if len(ind_positions) > 0 and len(office_positions) > 0 else np.vstack([ind_positions]) if len(ind_positions) > 0 else np.vstack([office_positions]) if len(office_positions) > 0 else np.array([]).reshape(0, 2)
            
            # 1. RESIDENTIAL-TO-WORK CONNECTIVITY
            if len(res_positions) > 0 and len(work_positions) > 0:
                res_work_distances = cdist(res_positions, work_positions)
                
                # Basic statistics
                features.extend([
                    np.mean(res_work_distances),  # Average commute distance
                    np.std(res_work_distances),   # Commute distance variation
                    np.min(res_work_distances),   # Best case commute
                    np.max(res_work_distances),   # Worst case commute
                    np.median(res_work_distances) # Median commute
                ])
                
                # Flow optimization metrics
                min_distances_per_resident = np.min(res_work_distances, axis=1)
                total_transport_cost = np.sum(min_distances_per_resident)
                max_individual_commute = np.max(min_distances_per_resident)
                
                features.extend([
                    total_transport_cost,      # Total city transport cost
                    max_individual_commute,    # Equity: worst individual commute
                    np.std(min_distances_per_resident)  # Commute equity (lower std = fairer)
                ])
                
                # Accessibility within thresholds
                accessible_1 = np.sum(res_work_distances <= 1.0) / res_work_distances.size
                accessible_2 = np.sum(res_work_distances <= 2.0) / res_work_distances.size
                accessible_3 = np.sum(res_work_distances <= 3.0) / res_work_distances.size
                
                features.extend([accessible_1, accessible_2, accessible_3])
                
            else:
                # Penalty values for missing districts
                features.extend([10.0, 5.0, 10.0, 10.0, 10.0, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0])
            
            # 2. RESIDENTIAL-TO-COMMERCIAL CONNECTIVITY  
            if len(res_positions) > 0 and len(comm_positions) > 0:
                res_comm_distances = cdist(res_positions, comm_positions)
                
                features.extend([
                    np.mean(res_comm_distances),
                    np.min(np.min(res_comm_distances, axis=1)),  # Best shopping access
                    np.max(np.min(res_comm_distances, axis=1)),  # Worst shopping access
                    np.mean(np.min(res_comm_distances, axis=1)), # Average shopping distance per resident
                ])
                
                # Shopping accessibility
                shop_accessible_1 = np.sum(res_comm_distances <= 1.0) / res_comm_distances.size
                shop_accessible_2 = np.sum(res_comm_distances <= 2.0) / res_comm_distances.size
                features.extend([shop_accessible_1, shop_accessible_2])
                
            else:
                features.extend([10.0, 10.0, 10.0, 10.0, 0.0, 0.0])
            
            # 3. WORK-TO-COMMERCIAL CONNECTIVITY (lunch/services during work)
            if len(work_positions) > 0 and len(comm_positions) > 0:
                work_comm_distances = cdist(work_positions, comm_positions)
                
                features.extend([
                    np.mean(work_comm_distances),
                    np.mean(np.min(work_comm_distances, axis=1)),  # Average work-to-service distance
                ])
            else:
                features.extend([10.0, 10.0])
            
            # 4. DISTRICT DENSITY AND DISTRIBUTION BALANCE
            total_cells = 49
            res_count = len(res_positions)
            work_count = len(work_positions) 
            comm_count = len(comm_positions)
            
            # Balance ratios (good transportation needs balanced development)
            work_to_res_ratio = work_count / max(1, res_count)  # Jobs per resident area
            comm_to_res_ratio = comm_count / max(1, res_count)  # Services per resident area
            
            features.extend([
                res_count / total_cells,    # Residential density
                work_count / total_cells,   # Work density  
                comm_count / total_cells,   # Commercial density
                work_to_res_ratio,          # Work-residence balance
                comm_to_res_ratio,          # Service-residence balance
            ])
            
            # 5. SPATIAL CLUSTERING ANALYSIS
            # Transportation works better with some clustering but not too much
            
            # Residential clustering
            if len(res_positions) > 1:
                res_internal_distances = cdist(res_positions, res_positions)
                res_clustering = np.mean(res_internal_distances[np.triu_indices_from(res_internal_distances, k=1)])
            else:
                res_clustering = 0
            
            # Work clustering  
            if len(work_positions) > 1:
                work_internal_distances = cdist(work_positions, work_positions)
                work_clustering = np.mean(work_internal_distances[np.triu_indices_from(work_internal_distances, k=1)])
            else:
                work_clustering = 0
                
            features.extend([res_clustering, work_clustering])
            
            # 6. CENTER-OF-MASS ANALYSIS (centralized vs distributed development)
            all_activity = np.vstack([res_positions, work_positions, comm_positions]) if len(res_positions) > 0 and len(work_positions) > 0 and len(comm_positions) > 0 else np.array([]).reshape(0, 2)
            
            if len(all_activity) > 0:
                center_of_mass = np.mean(all_activity, axis=0)
                distances_from_center = cdist(all_activity, [center_of_mass]).flatten()
                centralization_score = np.mean(distances_from_center)
                max_sprawl = np.max(distances_from_center)
            else:
                centralization_score = 5.0
                max_sprawl = 5.0
            
            features.extend([centralization_score, max_sprawl])
            
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"Created {features_array.shape[1]} connectivity features")
        return features_array
    
    def create_flow_optimization_features(self, grids):
        """Advanced features based on optimal flow theory"""
        n_grids = len(grids)
        features_list = []
        
        for grid in grids:
            features = []
            
            res_positions = np.argwhere(grid == 0)
            work_positions = np.vstack([np.argwhere(grid == 1), np.argwhere(grid == 4)]) if len(np.argwhere(grid == 1)) > 0 and len(np.argwhere(grid == 4)) > 0 else np.argwhere(grid == 1) if len(np.argwhere(grid == 1)) > 0 else np.argwhere(grid == 4)
            comm_positions = np.argwhere(grid == 2)
            
            # 1. MINIMUM SPANNING TREE COST (connectivity efficiency)
            if len(res_positions) > 1:
                from scipy.spatial.distance import pdist
                from scipy.cluster.hierarchy import linkage
                
                res_condensed_dist = pdist(res_positions)
                if len(res_condensed_dist) > 0:
                    mst_linkage = linkage(res_condensed_dist, method='single')
                    mst_cost = np.sum(mst_linkage[:, 2])  # Total MST cost for residential connectivity
                else:
                    mst_cost = 0
            else:
                mst_cost = 0
            
            features.append(mst_cost)
            
            # 2. TRANSPORTATION HUB POTENTIAL
            # Identify positions that minimize total distance to all other activity
            all_positions = []
            if len(res_positions) > 0:
                all_positions.extend(res_positions)
            if len(work_positions) > 0:
                all_positions.extend(work_positions)
            if len(comm_positions) > 0:
                all_positions.extend(comm_positions)
            
            if len(all_positions) > 0:
                all_positions = np.array(all_positions)
                
                # Find the position that minimizes total distance (potential hub location)
                min_total_distance = float('inf')
                best_hub_distance = 0
                
                for i in range(7):
                    for j in range(7):
                        hub_pos = np.array([[i, j]])
                        total_distance = np.sum(cdist(hub_pos, all_positions))
                        if total_distance < min_total_distance:
                            min_total_distance = total_distance
                            best_hub_distance = total_distance
                
                # Check if this optimal hub position is actually used
                optimal_hub = np.unravel_index(np.argmin([[np.sum(cdist([[i, j]], all_positions)) for j in range(7)] for i in range(7)]), (7, 7))
                hub_utilized = grid[optimal_hub] in [0, 1, 2, 4]  # Is it an activity district?
                
                features.extend([best_hub_distance, int(hub_utilized)])
            else:
                features.extend([100.0, 0])
            
            # 3. PERIMETER PENALTY (activities on edges create longer internal distances)
            edge_penalty = 0
            for i in range(7):
                for j in range(7):
                    if grid[i, j] in [0, 1, 2, 4]:  # Activity districts
                        if i == 0 or i == 6 or j == 0 or j == 6:  # On edge
                            # Distance penalty based on how far from center
                            center_distance = abs(i - 3) + abs(j - 3)
                            edge_penalty += center_distance
            
            features.append(edge_penalty)
            
            # 4. MANHATTAN VS EUCLIDEAN EFFICIENCY
            # Transportation networks often follow grid patterns (Manhattan distance)
            if len(res_positions) > 0 and len(work_positions) > 0:
                manhattan_total = 0
                euclidean_total = 0
                
                for res_pos in res_positions:
                    for work_pos in work_positions:
                        manhattan_dist = abs(res_pos[0] - work_pos[0]) + abs(res_pos[1] - work_pos[1])
                        euclidean_dist = np.sqrt((res_pos[0] - work_pos[0])**2 + (res_pos[1] - work_pos[1])**2)
                        
                        manhattan_total += manhattan_dist
                        euclidean_total += euclidean_dist
                
                if euclidean_total > 0:
                    grid_efficiency = euclidean_total / manhattan_total  # Closer to 1 = more grid-like
                else:
                    grid_efficiency = 0.7
            else:
                grid_efficiency = 0.7
            
            features.append(grid_efficiency)
            
            features_list.append(features)
        
        features_array = np.array(features_list)  
        print(f"Created {features_array.shape[1]} flow optimization features")
        return features_array
    
    def create_all_features(self, grids):
        """Combine all transportation-specific features"""
        print("🔧 Creating advanced transportation connectivity features...")
        
        connectivity_features = self.create_connectivity_matrix_features(grids)
        flow_features = self.create_flow_optimization_features(grids)
        
        all_features = np.hstack([connectivity_features, flow_features])
        
        print(f"✅ Total transportation features: {all_features.shape[1]}")
        print(f"   - Connectivity: {connectivity_features.shape[1]}")
        print(f"   - Flow optimization: {flow_features.shape[1]}")
        
        return all_features


class CNNTransportationOracle:
    """
    CNN-based Transportation Oracle for capturing spatial patterns in grid layouts.
    Uses Convolutional Neural Networks to understand transportation connectivity patterns.
    """
    
    def __init__(self):
        self.type = "Transportation"
        self.model = None
        self.scaler = None
        self.feature_engineer = AdvancedTransportationFeatures()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Check if PyTorch is available
        try:
            self.torch_available = True
            print(f"🔥 PyTorch available, using device: {self.device}")
        except ImportError:
            self.torch_available = False
            print("⚠️ PyTorch not available, falling back to feature-based approach")
    
    def create_enhanced_features(self, grids):
        """Create enhanced transportation features"""
        if self.feature_engineer is None:
            self.feature_engineer = AdvancedTransportationFeatures()
        return self.feature_engineer.create_all_features(grids)
    
    def create_cnn_model(self, model_type='standard', **kwargs):
        """Create PyTorch CNN model for transportation scoring
        
        Parameters:
        - model_type: 'standard' (CityCNN1) or 'deep' (CityCNN1Plus)
        - **kwargs: Additional parameters for the model
        """
        if not self.torch_available:
            raise ImportError("PyTorch not available for CNN implementation")
        
        if model_type == 'standard':
            model = CityCNN1(in_ch=5, p=kwargs.get('dropout', 0.25))
        elif model_type == 'deep':
            model = CityCNN1Plus(in_ch=5, 
                               p_head=kwargs.get('p_head', 0.35), 
                               p_mid=kwargs.get('p_mid', 0.10))
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'standard' or 'deep'")
        
        return model.to(self.device)
    
    def to_one_hot_5ch(self, grids_np):
        """Convert grids to one-hot encoding with 5 channels"""
        g = torch.as_tensor(grids_np, dtype=torch.long)
        oh = F.one_hot(g, num_classes=5)  # (N, 7, 7, 5)
        return oh.permute(0, 3, 1, 2).contiguous().float()  # (N, 5, 7, 7)
    
    def augment_symmetries(self, x5ch, use_aug=True):
        """Apply data augmentation via symmetries"""
        if not use_aug: 
            return x5ch
        if torch.rand(()) < 0.5: 
            x5ch = torch.flip(x5ch, dims=[-1])  # horizontal flip
        if torch.rand(()) < 0.5: 
            x5ch = torch.flip(x5ch, dims=[-2])  # vertical flip
        k = torch.randint(0, 4, (1,)).item()  # 0,90,180,270 deg
        if k: 
            x5ch = torch.rot90(x5ch, k, dims=[-2, -1])
        return x5ch
    
    def fit_model(self, grids, ratings, 
                 epochs=100, 
                 batch_size=128, 
                 test_size=0.2, 
                 learning_rate=3e-4,
                 weight_decay=7e-4,
                 early_stopping_patience=20,
                 use_augmentation=True,
                 model_type='standard',
                 **model_params):
        """Fit PyTorch CNN model for transportation advisor
        
        Parameters:
        - epochs: Number of training epochs (default: 100)
        - batch_size: Training batch size (default: 128)
        - test_size: Fraction of data for testing (default: 0.2)
        - learning_rate: Adam optimizer learning rate (default: 3e-4)
        - weight_decay: L2 regularization (default: 7e-4)
        - early_stopping_patience: Epochs to wait before early stopping (default: 20)
        - use_augmentation: Apply data augmentation (default: True)
        - model_type: 'standard' or 'deep' (default: 'standard')
        - **model_params: Additional parameters for create_cnn_model()
        """
        print(f"\n{'='*60}")
        print(f"Training PyTorch CNN Transportation Oracle")
        print(f"{'='*60}")
        
        print(f"Available training samples: {len(grids)}")
        print(f"Training parameters:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Test size: {test_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Model type: {model_type}")
        print(f"  - Device: {self.device}")
        
        # PyTorch approach
        print("🔥 Using PyTorch CNN approach for spatial pattern recognition...")
        
        # Train-test split on original grids
        X_train, X_test, y_train, y_test = train_test_split(
            grids, ratings, test_size=test_size, random_state=42)
        
        # Convert to one-hot encoding
        Xt_train = self.to_one_hot_5ch(X_train)  # (N, 5, 7, 7)
        Xt_test = self.to_one_hot_5ch(X_test)
        
        # Convert to PyTorch tensors
        yt_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        yt_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Create datasets and data loaders
        train_dataset = TensorDataset(Xt_train, yt_train)
        test_dataset = TensorDataset(Xt_test, yt_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.model = self.create_cnn_model(model_type=model_type, **model_params)
        
        print("🏗️ CNN Architecture:")
        print(self.model)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, threshold=1e-4, 
            cooldown=0, min_lr=0.0, eps=1e-8
        )
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        print("🚀 Training PyTorch CNN model...")
        
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # Apply augmentation
                if use_augmentation:
                    batch_x = self.augment_symmetries(batch_x, use_aug=True)
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
            
            train_loss /= len(train_dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_x)
                    val_loss += criterion(predictions, batch_y).item() * batch_x.size(0)
            
            val_loss /= len(test_dataset)
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            # Training predictions
            train_preds = []
            for batch_x, batch_y in DataLoader(train_dataset, batch_size=batch_size):
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu().numpy().reshape(-1)
                train_preds.append(preds)
            train_predictions = np.concatenate(train_preds)
            
            # Test predictions
            test_preds = []
            for batch_x, batch_y in DataLoader(test_dataset, batch_size=batch_size):
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu().numpy().reshape(-1)
                test_preds.append(preds)
            test_predictions = np.concatenate(test_preds)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        print(f"\n📊 PyTorch CNN Performance:")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        
        # Return data in expected format
        train_data = {'target': y_train}
        test_data = {'target': y_test}
        
        return test_r2, train_data, test_data

    def predict(self, grids, batch_size=1024):
        """Predict transportation scores for grids"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit_model first.")
        
        print(f"🚛 Creating transportation predictions for {len(grids)} grids...")
        
        if self.torch_available and hasattr(self.model, 'forward'):
            # PyTorch CNN prediction
            self.model.eval()
            predictions = []
            
            # Convert to one-hot encoding
            X_onehot = self.to_one_hot_5ch(grids)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_onehot)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            with torch.no_grad():
                for (batch_x,) in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_preds = self.model(batch_x).cpu().numpy().reshape(-1)
                    predictions.append(batch_preds)
            
            return np.concatenate(predictions)
        else:
            # Feature-based prediction fallback
            features = self.feature_engineer.create_all_features(grids)
            if self.scaler:
                features = self.scaler.transform(features)
            return self.model.predict(features)
    
    def save_model(self, filename):
        """Save the trained PyTorch model"""
        if self.torch_available and hasattr(self.model, 'state_dict'):
            # Save PyTorch model
            model_filename = filename.replace('.pkl', '_pytorch.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_class': self.model.__class__.__name__,
                'device': str(self.device)
            }, model_filename)
            
            # Save metadata
            metadata = {
                'type': self.type,
                'torch_available': self.torch_available,
                'scaler': self.scaler,
                'model_type': 'PyTorch CNN',
                'device': str(self.device)
            }
            with open(filename, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Saved PyTorch model to {model_filename} and metadata to {filename}")
        else:
            # Save feature-based model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'type': self.type,
                'torch_available': self.torch_available,
                'model_type': 'Feature-based'
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Saved feature-based model to {filename}")
    
    def load_model(self, filename):
        """Load the trained PyTorch model"""
        if self.torch_available:
            try:
                # Try to load PyTorch model
                model_filename = filename.replace('.pkl', '_pytorch.pth')
                if os.path.exists(model_filename):
                    checkpoint = torch.load(model_filename, map_location=self.device)
                    
                    # Recreate model based on saved class name
                    model_class = checkpoint['model_class']
                    if model_class == 'CityCNN1':
                        self.model = CityCNN1().to(self.device)
                    elif model_class == 'CityCNN1Plus':
                        self.model = CityCNN1Plus().to(self.device)
                    else:
                        raise ValueError(f"Unknown model class: {model_class}")
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Load metadata
                    with open(filename, 'rb') as f:
                        metadata = pickle.load(f)
                    self.scaler = metadata.get('scaler')
                    
                    print(f"📁 Loaded PyTorch model from {model_filename}")
                    return
            except Exception as e:
                print(f"⚠️ Failed to load PyTorch model: {e}")
        
        # Load feature-based model
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        
        print(f"📁 Loaded feature-based model from {filename}")
    
    def get_model_configs(self):
        """Get predefined PyTorch CNN model configurations for experimentation"""
        configs = {
            'lightweight': {
                'model_type': 'standard',
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 80,
                'use_augmentation': False
            },
            'standard': {
                'model_type': 'standard',
                'dropout': 0.25,
                'learning_rate': 3e-4,
                'weight_decay': 7e-4,
                'batch_size': 128,
                'epochs': 100,
                'use_augmentation': True
            },
            'deep': {
                'model_type': 'deep',
                'p_head': 0.35,
                'p_mid': 0.10,
                'learning_rate': 2e-4,
                'weight_decay': 1e-3,
                'batch_size': 64,
                'epochs': 150,
                'use_augmentation': True
            },
            'fast': {
                'model_type': 'standard',
                'dropout': 0.15,
                'learning_rate': 5e-4,
                'weight_decay': 5e-4,
                'batch_size': 256,
                'epochs': 50,
                'use_augmentation': False,
                'early_stopping_patience': 10
            },
            'robust': {
                'model_type': 'deep',
                'p_head': 0.4,
                'p_mid': 0.15,
                'learning_rate': 1e-4,
                'weight_decay': 2e-3,
                'batch_size': 64,
                'epochs': 200,
                'use_augmentation': True,
                'early_stopping_patience': 30
            }
        }
        return configs
    
    def fit_with_config(self, grids, ratings, config_name='standard', **training_params):
        """Fit model using a predefined configuration
        
        Parameters:
        - config_name: One of 'lightweight', 'standard', 'deep', 'fast', 'robust'
        - **training_params: Additional training parameters for fit_model()
        """
        configs = self.get_model_configs()
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        
        model_config = configs[config_name]
        print(f"🔧 Using '{config_name}' configuration:")
        for key, value in model_config.items():
            print(f"   {key}: {value}")
        
        return self.fit_model(grids, ratings, **model_config, **training_params)


class CityCNN1(nn.Module):
    """Compact CNN with two conv blocks and GAP head."""
    def __init__(self, in_ch=5, p=0.25):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),   nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.gap(x)
        return self.head(x)

class CityCNN1Plus(nn.Module):
    """Deeper/regularized CNN for tougher advisors."""
    def __init__(self, in_ch=5, p_head=0.35, p_mid=0.10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout(p_mid),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout(p_mid),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(p_mid),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p_head),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(p_head), nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.gap(x)
        return self.head(x)


class CNNWellnessOracle(CNNTransportationOracle):
    """
    Same as the transportation oracle
    """
    def __init__(self):
        super().__init__()
        self.type = "Wellness"