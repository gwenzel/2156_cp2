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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        # Check if tensorflow is available
        try:
            self.tf = tf
            self.keras = keras
            self.layers = layers
            self.tf_available = True
        except ImportError:
            self.tf_available = False
            print("⚠️ TensorFlow not available, falling back to feature-based approach")
    
    def create_enhanced_features(self, grids):
        """Create enhanced transportation features"""
        if self.feature_engineer is None:
            self.feature_engineer = AdvancedTransportationFeatures()
        return self.feature_engineer.create_all_features(grids)
    
    def create_cnn_model(self, input_shape=(7, 7, 1), 
                        conv_filters=[32, 64, 128, 64], 
                        conv_kernels=[(3,3), (3,3), (3,3), (7,7)],
                        dense_layers=[256, 128, 64],
                        dropout_rates=[0.3, 0.2],
                        learning_rate=0.001,
                        activation='relu'):
        """Create CNN architecture optimized for 7x7 transportation grids
        
        Parameters:
        - input_shape: Shape of input grids (default: (7, 7, 1))
        - conv_filters: List of filter numbers for conv layers (default: [32, 64, 128, 64])
        - conv_kernels: List of kernel sizes for conv layers (default: [(3,3), (3,3), (3,3), (7,7)])
        - dense_layers: List of neurons in dense layers (default: [256, 128, 64])
        - dropout_rates: List of dropout rates (default: [0.3, 0.2])
        - learning_rate: Adam optimizer learning rate (default: 0.001)
        - activation: Activation function for hidden layers (default: 'relu')
        """
        if not self.tf_available:
            raise ImportError("TensorFlow not available for CNN implementation")
        
        layers_list = []
        
        # Add convolutional layers
        for i, (filters, kernel) in enumerate(zip(conv_filters, conv_kernels)):
            if i == 0:
                # First layer needs input_shape
                layers_list.append(
                    self.layers.Conv2D(filters, kernel, activation=activation, 
                                     padding='same' if kernel != (7,7) else 'valid',
                                     input_shape=input_shape, 
                                     name=f'conv_layer_{i+1}')
                )
            else:
                layers_list.append(
                    self.layers.Conv2D(filters, kernel, activation=activation,
                                     padding='same' if kernel != (7,7) else 'valid',
                                     name=f'conv_layer_{i+1}')
                )
            layers_list.append(self.layers.BatchNormalization())
        
        # Flatten
        layers_list.append(self.layers.Flatten())
        
        # Add dropout after flatten
        if len(dropout_rates) > 0:
            layers_list.append(self.layers.Dropout(dropout_rates[0]))
        
        # Add dense layers
        for i, neurons in enumerate(dense_layers):
            layers_list.append(
                self.layers.Dense(neurons, activation=activation, 
                                name=f'dense_layer_{i+1}')
            )
            # Add dropout between dense layers (skip last layer)
            if i < len(dense_layers) - 1 and len(dropout_rates) > 1:
                layers_list.append(self.layers.Dropout(dropout_rates[1]))
        
        # Output layer
        layers_list.append(self.layers.Dense(1, activation='linear', name='output'))
        
        model = self.keras.Sequential(layers_list)
        
        # Compile with appropriate optimizer and loss
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def fit_model(self, grids, ratings, 
                 epochs=100, 
                 batch_size=32, 
                 test_size=0.2, 
                 early_stopping_patience=20,
                 lr_reduction_factor=0.5,
                 lr_reduction_patience=10,
                 min_lr=1e-7,
                 **model_params):
        """Fit CNN model for transportation advisor
        
        Parameters:
        - epochs: Number of training epochs (default: 100)
        - batch_size: Training batch size (default: 32)
        - test_size: Fraction of data for testing (default: 0.2)
        - early_stopping_patience: Epochs to wait before early stopping (default: 20)
        - lr_reduction_factor: Factor to reduce learning rate (default: 0.5)
        - lr_reduction_patience: Epochs to wait before reducing LR (default: 10)
        - min_lr: Minimum learning rate (default: 1e-7)
        - **model_params: Additional parameters for create_cnn_model()
        """
        print(f"\n{'='*60}")
        print(f"Training CNN Transportation Oracle")
        print(f"{'='*60}")
        
        print(f"Available training samples: {len(grids)}")
        print(f"Training parameters:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Test size: {test_size}")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        
        # CNN approach
        print("🧠 Using CNN approach for spatial pattern recognition...")
        
        # Reshape grids for CNN (add channel dimension)
        X = grids.reshape(-1, 7, 7, 1).astype('float32')
        # Normalize grid values
        X = X / 4.0  # Since district values are 0-4
        
        y = ratings.astype('float32')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Create and compile model
        self.model = self.create_cnn_model(**model_params)
        
        print("🏗️ CNN Architecture:")
        self.model.summary()
        
        # Early stopping and model checkpointing
        callbacks = [
            self.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
            self.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=lr_reduction_factor, patience=lr_reduction_patience, min_lr=min_lr)
        ]
        
        # Train the model
        print("🚀 Training CNN model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        print(f"\n📊 CNN Performance:")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")

        train_data = {}
        train_data['target'] = y_train
        test_data = {}
        test_data['target'] = y_test
        
        return test_r2, train_data, test_data

    def predict(self, grids):
        """Predict transportation scores for grids"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit_model first.")
        
        print(f"🚛 Creating transportation predictions for {len(grids)} grids...")
        
        if self.tf_available and hasattr(self.model, 'predict'):
            # CNN prediction
            X = grids.reshape(-1, 7, 7, 1).astype('float32') / 4.0
            predictions = self.model.predict(X, verbose=0)
            return predictions.flatten()
        else:
            # Feature-based prediction
            features = self.feature_engineer.create_all_features(grids)
            if self.scaler:
                features = self.scaler.transform(features)
            return self.model.predict(features)
    
    def save_model(self, filename):
        """Save the trained model"""
        if self.tf_available and hasattr(self.model, 'save'):
            # Save CNN model
            model_filename = filename.replace('.pkl', '_cnn.h5')
            self.model.save(model_filename)
            
            # Save scaler and metadata
            metadata = {
                'type': self.type,
                'tf_available': self.tf_available,
                'scaler': self.scaler,
                'model_type': 'CNN'
            }
            with open(filename, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Saved CNN model to {model_filename} and metadata to {filename}")
        else:
            # Save feature-based model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'type': self.type,
                'tf_available': self.tf_available,
                'model_type': 'Feature-based'
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Saved feature-based model to {filename}")
    
    def load_model(self, filename):
        """Load the trained model"""
        if self.tf_available:
            try:
                # Try to load CNN model
                model_filename = filename.replace('.pkl', '_cnn.h5')
                if os.path.exists(model_filename):
                    self.model = self.keras.models.load_model(model_filename)
                    
                    # Load metadata
                    with open(filename, 'rb') as f:
                        metadata = pickle.load(f)
                    self.scaler = metadata.get('scaler')
                    
                    print(f"📁 Loaded CNN model from {model_filename}")
                    return
            except:
                pass
        
        # Load feature-based model
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        
        print(f"📁 Loaded feature-based model from {filename}")
    
    def get_model_configs(self):
        """Get predefined CNN model configurations for experimentation"""
        configs = {
            'lightweight': {
                'conv_filters': [16, 32, 64],
                'conv_kernels': [(3,3), (3,3), (5,5)],
                'dense_layers': [128, 64],
                'dropout_rates': [0.2, 0.1],
                'learning_rate': 0.001
            },
            'standard': {
                'conv_filters': [32, 64, 128, 64],
                'conv_kernels': [(3,3), (3,3), (3,3), (7,7)],
                'dense_layers': [256, 128, 64],
                'dropout_rates': [0.3, 0.2],
                'learning_rate': 0.001
            },
            'deep': {
                'conv_filters': [32, 64, 128, 256, 128],
                'conv_kernels': [(3,3), (3,3), (3,3), (3,3), (5,5)],
                'dense_layers': [512, 256, 128, 64],
                'dropout_rates': [0.4, 0.3],
                'learning_rate': 0.0005
            },
            'wide': {
                'conv_filters': [64, 128, 256, 128],
                'conv_kernels': [(3,3), (3,3), (3,3), (7,7)],
                'dense_layers': [512, 256, 128],
                'dropout_rates': [0.3, 0.2],
                'learning_rate': 0.001
            },
            'minimal': {
                'conv_filters': [16, 32],
                'conv_kernels': [(3,3), (5,5)],
                'dense_layers': [64, 32],
                'dropout_rates': [0.2],
                'learning_rate': 0.002
            }
        }
        return configs
    
    def fit_with_config(self, grids, ratings, config_name='standard', **training_params):
        """Fit model using a predefined configuration
        
        Parameters:
        - config_name: One of 'lightweight', 'standard', 'deep', 'wide', 'minimal'
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

