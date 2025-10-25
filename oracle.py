# Setup and Data Loading
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.spatial.distance import cdist
import xgboost as xgb

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
                center_counts[grid[i, j]] += 1
            for i, j in edge_positions:
                edge_counts[grid[i, j]] += 1
                
            # Normalize by number of positions
            center_ratios = center_counts / len(center_positions)
            edge_ratios = edge_counts / len(edge_positions)
            
            features.extend(center_ratios)
            features.extend(edge_ratios)
            
            # 3. Corner features
            corners = [(0,0), (0,6), (6,0), (6,6)]
            corner_counts = np.zeros(5)
            for i, j in corners:
                corner_counts[grid[i, j]] += 1
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
            counts = np.bincount(grid.flatten(), minlength=5)
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


class Oracle:
    """
    Advanced ML Oracle with multiple models per advisor
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.advisor_names = ["Wellness", "Tax", "Transportation", "Business"]
        
    def get_model_ensemble(self):
        """Return a dictionary of models to ensemble"""
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
    
    def fit_advisor_models(self, grids, ratings, advisor_idx):
        """Fit ensemble of models for specific advisor"""
        advisor_name = self.advisor_names[advisor_idx]
        print(f"\n{'='*50}")
        print(f"Training models for {advisor_name} Advisor")
        print(f"{'='*50}")
        
        # Get labeled data
        mask = ~np.isnan(ratings[:, advisor_idx])
        labeled_grids = grids[mask]
        labeled_ratings = ratings[mask, advisor_idx]
        
        print(f"Available training samples: {len(labeled_grids)}")
        
        # Feature engineering
        print("Performing feature engineering...")
        features = self.feature_engineer.create_all_features(labeled_grids)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labeled_ratings, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[advisor_name] = scaler
        
        # Train ensemble models
        models = self.get_model_ensemble()
        trained_models = {}
        model_scores = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Some models work better with scaled features
            if model_name in ['ridge', 'elastic_net', 'svr']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            # Calculate scores
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            print(f"  {model_name:15} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
            
            trained_models[model_name] = model
            model_scores[model_name] = test_r2
        
        # Store models
        self.models[advisor_name] = trained_models
        
        # Create ensemble predictions (weighted by R² scores)
        ensemble_weights = np.array(list(model_scores.values()))
        ensemble_weights = np.maximum(ensemble_weights, 0)  # No negative weights
        if ensemble_weights.sum() > 0:
            ensemble_weights = ensemble_weights / ensemble_weights.sum()
        else:
            ensemble_weights = np.ones(len(ensemble_weights)) / len(ensemble_weights)
        
        print(f"\nEnsemble weights: {dict(zip(model_scores.keys(), ensemble_weights))}")
        
        # Calculate ensemble performance
        ensemble_pred = self._ensemble_predict(X_test, advisor_name, use_scaled=True)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\nEnsemble Performance:")
        print(f"  Test R²: {ensemble_r2:.4f}")
        print(f"  Test MAE: {ensemble_mae:.4f}")
        
        # Store ensemble weights
        self.models[advisor_name]['_ensemble_weights'] = dict(zip(model_scores.keys(), ensemble_weights))
        
        return ensemble_r2, ensemble_mae
    
    def _ensemble_predict(self, X, advisor_name, use_scaled=False):
        """Make ensemble prediction for given features"""
        models = self.models[advisor_name]
        weights = models['_ensemble_weights']
        
        predictions = []
        total_weight = 0
        
        for model_name, weight in weights.items():
            if model_name.startswith('_'):  # Skip metadata
                continue
                
            model = models[model_name]
            
            # Use appropriate features (scaled vs unscaled)
            if model_name in ['ridge', 'elastic_net', 'svr'] and use_scaled:
                X_input = self.scalers[advisor_name].transform(X)
            else:
                X_input = X
            
            pred = model.predict(X_input)
            predictions.append(pred * weight)
            total_weight += weight
        
        if total_weight > 0:
            return np.sum(predictions, axis=0) / total_weight
        else:
            return np.mean(predictions, axis=0)
    
    def predict_all_grids(self, grids, advisor_idx):
        """Predict scores for all grids using the ensemble"""
        advisor_name = self.advisor_names[advisor_idx]
        
        print(f"Generating features for all {len(grids)} grids...")
        features = self.feature_engineer.create_all_features(grids)
        
        print(f"Making ensemble predictions for {advisor_name}...")
        predictions = self._ensemble_predict(features, advisor_name, use_scaled=False)
        
        return predictions
    
    def fit_all_advisors(self, grids, ratings):
        """Fit models for all advisors"""
        results = {}
        
        for advisor_idx in range(4):
            r2, mae = self.fit_advisor_models(grids, ratings, advisor_idx)
            results[self.advisor_names[advisor_idx]] = {'r2': r2, 'mae': mae}
        
        print(f"\n{'='*60}")
        print("ORACLE TRAINING COMPLETE")
        print(f"{'='*60}")
        
        for advisor, metrics in results.items():
            print(f"{advisor:15} - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return results


# Let's start with a simpler version that works, then we can enhance
class SimpleOracle:
    """
    Simplified Oracle that focuses on working regression models
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.advisor_names = ["Wellness", "Tax", "Transportation", "Business"]
    
    def create_features(self, grids):
        """Create comprehensive but simple features"""
        n_grids = len(grids)
        
        # Flatten grids (49 features)
        grids_flat = grids.reshape(n_grids, -1)
        
        # District counts (5 features)
        district_counts = np.zeros((n_grids, 5))
        for i in range(5):
            district_counts[:, i] = np.sum(grids_flat == i, axis=1)
        
        # Spatial features
        spatial_features = []
        
        for grid in grids:
            features = []
            
            # Adjacency clustering for each district
            for district in range(5):
                adjacency_score = 0
                count = 0
                
                for i in range(7):
                    for j in range(7):
                        if grid[i, j] == district:
                            # Count same-district neighbors
                            neighbors = 0
                            total_neighbors = 0
                            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 7 and 0 <= nj < 7:
                                    total_neighbors += 1
                                    if grid[ni, nj] == district:
                                        neighbors += 1
                            if total_neighbors > 0:
                                adjacency_score += neighbors / total_neighbors
                                count += 1
                
                if count > 0:
                    features.append(adjacency_score / count)
                else:
                    features.append(0)
            
            # Edge vs center distribution
            center_count = np.zeros(5)
            edge_count = np.zeros(5)
            
            for i in range(7):
                for j in range(7):
                    district = int(grid[i, j])  # Convert to int for indexing
                    if i == 0 or i == 6 or j == 0 or j == 6:  # Edge
                        edge_count[district] += 1
                    else:  # Center
                        center_count[district] += 1
            
            # Add ratios
            for i in range(5):
                if center_count[i] + edge_count[i] > 0:
                    features.append(edge_count[i] / (edge_count[i] + center_count[i]))
                else:
                    features.append(0.5)  # Neutral if no districts of this type
            
            # Corner analysis
            corner_districts = [int(grid[0,0]), int(grid[0,6]), int(grid[6,0]), int(grid[6,6])]
            corner_counts = np.bincount(corner_districts, minlength=5)
            features.extend(corner_counts)
            
            # Grid diversity
            unique_districts = len(np.unique(grid))
            features.append(unique_districts)
            
            spatial_features.append(features)
        
        spatial_array = np.array(spatial_features)
        
        # Combine all features
        all_features = np.hstack([grids_flat, district_counts, spatial_array])
        
        print(f"Created {all_features.shape[1]} features:")
        print(f"  - Grid positions: 49")
        print(f"  - District counts: 5") 
        print(f"  - Spatial features: {spatial_array.shape[1]}")
        
        return all_features
    
    def fit_advisor_model(self, grids, ratings, advisor_idx):
        """Fit ensemble models for one advisor"""
        advisor_name = self.advisor_names[advisor_idx]
        print(f"\n{'='*50}")
        print(f"Training models for {advisor_name} Advisor")
        print(f"{'='*50}")
        
        # Get labeled data
        mask = ~np.isnan(ratings[:, advisor_idx])
        labeled_grids = grids[mask]
        labeled_ratings = ratings[mask, advisor_idx]
        
        print(f"Available training samples: {len(labeled_grids)}")
        
        # Create features
        print("Creating features...")
        features = self.create_features(labeled_grids)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labeled_ratings, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[advisor_name] = scaler
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['ridge', 'elastic_net']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            print(f"  {name:15} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
            
            trained_models[name] = model
            model_scores[name] = test_r2
        
        # Select best model (could enhance with ensemble later)
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
        print(f"\nBest model: {best_model_name} (R²: {model_scores[best_model_name]:.4f})")
        
        self.models[advisor_name] = {
            'best_model': trained_models[best_model_name],
            'best_model_name': best_model_name,
            'all_models': trained_models,
            'scores': model_scores
        }
        
        return model_scores[best_model_name]
    
    def predict_all_grids(self, grids, advisor_idx):
        """Predict scores for all grids"""
        advisor_name = self.advisor_names[advisor_idx]
        
        print(f"Creating features for all {len(grids)} grids...")
        features = self.create_features(grids)
        
        model_info = self.models[advisor_name]
        model = model_info['best_model']
        model_name = model_info['best_model_name']
        
        print(f"Making predictions with {model_name}...")
        
        if model_name in ['ridge', 'elastic_net']:
            features_scaled = self.scalers[advisor_name].transform(features)
            predictions = model.predict(features_scaled)
        else:
            predictions = model.predict(features)
        
        return predictions
    
    def fit_all_advisors(self, grids, ratings):
        """Fit models for all advisors"""
        results = {}
        
        for advisor_idx in range(4):
            best_r2 = self.fit_advisor_model(grids, ratings, advisor_idx)
            results[self.advisor_names[advisor_idx]] = best_r2
        
        print(f"\n{'='*60}")
        print("ORACLE TRAINING COMPLETE")
        print(f"{'='*60}")
        
        for advisor, r2 in results.items():
            print(f"{advisor:15} - Best R²: {r2:.4f}")
        
        return results

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


class AdvancedOracleV2:
    """
    Advanced Oracle V2 with specialized domain knowledge and better ML techniques
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.advisor_names = ["Wellness", "Tax", "Transportation", "Business"]
    
    def create_wellness_features(self, grids):
        """Specialized features for Wellness Advisor - Parks, Health, Accessibility"""
        n_grids = len(grids)
        features_list = []
        
        for grid in grids:
            features = []
            
            # Park-centric features (Wellness advisor cares about green spaces)
            park_positions = np.argwhere(grid == 3)  # Parks
            res_positions = np.argwhere(grid == 0)   # Residential
            
            # 1. Park accessibility from residential areas
            if len(park_positions) > 0 and len(res_positions) > 0:
                distances = cdist(res_positions, park_positions)
                avg_park_distance = np.mean(distances.min(axis=1))
                max_park_distance = np.max(distances.min(axis=1))
                features.extend([avg_park_distance, max_park_distance])
            else:
                features.extend([10.0, 10.0])  # Penalty for no parks/residential
            
            # 2. Park distribution quality
            park_count = len(park_positions)
            park_density = park_count / 49  # Normalized by grid size
            features.extend([park_count, park_density])
            
            # 3. Park clustering (wellness prefers distributed parks)
            if len(park_positions) > 1:
                park_distances = cdist(park_positions, park_positions)
                park_clustering = np.mean(park_distances[np.triu_indices_from(park_distances, k=1)])
            else:
                park_clustering = 0
            features.append(park_clustering)
            
            # 4. Residential-Industrial separation (health concern)
            ind_positions = np.argwhere(grid == 1)  # Industrial
            if len(ind_positions) > 0 and len(res_positions) > 0:
                ind_res_distances = cdist(res_positions, ind_positions)
                min_separation = np.min(ind_res_distances.min(axis=1))
                avg_separation = np.mean(ind_res_distances.min(axis=1))
                features.extend([min_separation, avg_separation])
            else:
                features.extend([7.0, 7.0])  # Good if no industrial pollution
            
            # 5. Green space connectivity (continuous park areas)
            connected_park_clusters = 0
            visited = set()
            for pos in park_positions:
                pos_tuple = tuple(pos)
                if pos_tuple not in visited:
                    # BFS to find connected park cluster
                    queue = [pos_tuple]
                    cluster_size = 0
                    while queue:
                        current = queue.pop(0)
                        if current in visited:
                            continue
                        visited.add(current)
                        cluster_size += 1
                        
                        # Check neighbors
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ni, nj = current[0] + di, current[1] + dj
                            if (0 <= ni < 7 and 0 <= nj < 7 and 
                                grid[ni, nj] == 3 and (ni, nj) not in visited):
                                queue.append((ni, nj))
                    
                    if cluster_size > 1:
                        connected_park_clusters += 1
            
            features.append(connected_park_clusters)
            
            # 6. Residential quality indicators
            res_count = len(res_positions)
            
            # Residential clustering (some clustering good for communities)
            if len(res_positions) > 1:
                res_distances = cdist(res_positions, res_positions)
                res_clustering = np.mean(res_distances[np.triu_indices_from(res_distances, k=1)])
            else:
                res_clustering = 0
            features.extend([res_count, res_clustering])
            
            # 7. Commercial accessibility from residential
            comm_positions = np.argwhere(grid == 2)  # Commercial
            if len(comm_positions) > 0 and len(res_positions) > 0:
                comm_distances = cdist(res_positions, comm_positions)
                avg_comm_access = np.mean(comm_distances.min(axis=1))
            else:
                avg_comm_access = 10.0
            features.append(avg_comm_access)
            
            # 8. Edge parks (parks on edges might be less accessible)
            edge_parks = 0
            for pos in park_positions:
                if pos[0] in [0, 6] or pos[1] in [0, 6]:
                    edge_parks += 1
            features.append(edge_parks / max(1, park_count))  # Ratio of edge parks
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def create_transportation_features(self, grids):
        """Specialized features for Transportation Advisor - Commute, Connectivity, Centralization"""
        n_grids = len(grids)
        features_list = []
        
        for grid in grids:
            features = []
            
            # Transportation focuses on minimizing commute distances
            res_positions = np.argwhere(grid == 0)   # Residential (where people live)
            office_positions = np.argwhere(grid == 4)  # Office (where people work)
            ind_positions = np.argwhere(grid == 1)   # Industrial (also work)
            comm_positions = np.argwhere(grid == 2)  # Commercial (services)
            
            work_positions = np.vstack([office_positions, ind_positions]) if len(office_positions) > 0 and len(ind_positions) > 0 else office_positions if len(office_positions) > 0 else ind_positions
            
            # 1. Commute distance optimization
            if len(res_positions) > 0 and len(work_positions) > 0:
                commute_distances = cdist(res_positions, work_positions)
                avg_commute = np.mean(commute_distances.min(axis=1))
                max_commute = np.max(commute_distances.min(axis=1))
                total_commute_cost = np.sum(commute_distances.min(axis=1))
                features.extend([avg_commute, max_commute, total_commute_cost])
            else:
                features.extend([10.0, 10.0, 100.0])  # Penalty
            
            # 2. Centralization metrics (centralized planning reduces overall travel)
            all_activity_positions = []
            for district_type in [0, 1, 2, 4]:  # Exclude parks for activity analysis
                positions = np.argwhere(grid == district_type)
                all_activity_positions.extend(positions)
            
            if len(all_activity_positions) > 0:
                all_activity_positions = np.array(all_activity_positions)
                center_of_mass = np.mean(all_activity_positions, axis=0)
                
                # Distance from center of mass (lower is better for transportation)
                distances_from_center = cdist(all_activity_positions, [center_of_mass])
                avg_centralization = np.mean(distances_from_center)
                max_centralization = np.max(distances_from_center)
                features.extend([avg_centralization, max_centralization])
            else:
                features.extend([5.0, 5.0])
            
            # 3. Network connectivity (Manhattan distance efficiency)
            # Transportation advisor likely prefers grid-like efficient layouts
            connectivity_score = 0
            total_pairs = 0
            
            for i in range(len(res_positions)):
                for j in range(len(work_positions)):
                    res_pos = res_positions[i]
                    work_pos = work_positions[j]
                    
                    # Manhattan distance (more realistic for urban transport)
                    manhattan_dist = abs(res_pos[0] - work_pos[0]) + abs(res_pos[1] - work_pos[1])
                    # Euclidean distance
                    euclidean_dist = np.sqrt((res_pos[0] - work_pos[0])**2 + (res_pos[1] - work_pos[1])**2)
                    
                    # Efficiency ratio (closer to 1 is better for grid-based transport)
                    if euclidean_dist > 0:
                        efficiency = euclidean_dist / manhattan_dist
                        connectivity_score += efficiency
                        total_pairs += 1
            
            if total_pairs > 0:
                avg_connectivity = connectivity_score / total_pairs
            else:
                avg_connectivity = 0.7  # Neutral value
            features.append(avg_connectivity)
            
            # 4. Mixed-use development (reduces travel needs)
            mixed_use_score = 0
            for i in range(7):
                for j in range(7):
                    # Count different district types in 3x3 neighborhood
                    neighborhood_types = set()
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 7 and 0 <= nj < 7:
                                neighborhood_types.add(grid[ni, nj])
                    
                    # Higher diversity in neighborhood = better mixed use
                    mixed_use_score += len(neighborhood_types)
            
            mixed_use_avg = mixed_use_score / 49
            features.append(mixed_use_avg)
            
            # 5. Commercial accessibility (reduces shopping travel)
            if len(res_positions) > 0 and len(comm_positions) > 0:
                shopping_distances = cdist(res_positions, comm_positions)
                avg_shopping_distance = np.mean(shopping_distances.min(axis=1))
                max_shopping_distance = np.max(shopping_distances.min(axis=1))
                features.extend([avg_shopping_distance, max_shopping_distance])
            else:
                features.extend([10.0, 10.0])
            
            # 6. Activity density (higher density reduces travel distances)
            activity_counts = {}
            for district_type in range(5):
                activity_counts[district_type] = np.sum(grid == district_type)
            
            # Calculate density score (more even distribution better for transport)
            total_activity = sum(activity_counts[dt] for dt in [0, 1, 2, 4])  # Exclude parks
            if total_activity > 0:
                activity_density_variance = np.var([activity_counts[dt] for dt in [0, 1, 2, 4]])
            else:
                activity_density_variance = 100
            features.append(activity_density_variance)
            
            # 7. Border inefficiency (activities on borders create longer internal distances)
            border_activity = 0
            total_activity_cells = 0
            for i in range(7):
                for j in range(7):
                    if grid[i, j] in [0, 1, 2, 4]:  # Activity districts
                        total_activity_cells += 1
                        if i in [0, 6] or j in [0, 6]:  # Border positions
                            border_activity += 1
            
            border_ratio = border_activity / max(1, total_activity_cells)
            features.append(border_ratio)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def create_comprehensive_features(self, grids, advisor_idx):
        """Create features tailored for specific advisor"""
        # Basic features for all
        n_grids = len(grids)
        grids_flat = grids.reshape(n_grids, -1)
        
        # District counts
        district_counts = np.zeros((n_grids, 5))
        for i in range(5):
            district_counts[:, i] = np.sum(grids_flat == i, axis=1)
        
        # Advisor-specific features
        if advisor_idx == 0:  # Wellness
            specialized = self.create_wellness_features(grids)
            print(f"Created {specialized.shape[1]} wellness-specific features")
        elif advisor_idx == 2:  # Transportation
            specialized = self.create_transportation_features(grids)
            print(f"Created {specialized.shape[1]} transportation-specific features")
        else:
            # For Tax and Business, use simpler but effective features
            specialized = self.create_simple_spatial_features(grids)
            print(f"Created {specialized.shape[1]} general spatial features")
        
        # Combine all features
        all_features = np.hstack([grids_flat, district_counts, specialized])
        
        print(f"Total features: {all_features.shape[1]} (grid: 49, counts: 5, specialized: {specialized.shape[1]})")
        return all_features
    
    def create_simple_spatial_features(self, grids):
        """Simple but effective spatial features for Tax and Business advisors"""
        features_list = []
        
        for grid in grids:
            features = []
            
            # Basic adjacency for all districts
            for district in range(5):
                positions = np.argwhere(grid == district)
                if len(positions) > 1:
                    distances = cdist(positions, positions)
                    avg_internal_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                else:
                    avg_internal_distance = 0
                features.append(avg_internal_distance)
            
            # Corner preferences
            corners = [(0,0), (0,6), (6,0), (6,6)]
            for district in range(5):
                corner_count = sum(1 for pos in corners if grid[pos[0], pos[1]] == district)
                features.append(corner_count)
            
            features_list.append(features)
        
        return np.array(features_list)


class SuperiorMLTrainer:
    """
    Advanced ML trainer with hyperparameter optimization and feature selection
    """
    
    def __init__(self, oracle):
        self.oracle = oracle
        self.advisor_names = ["Wellness", "Tax", "Transportation", "Business"]
    
    def get_advanced_models(self):
        """Advanced model ensemble with hyperparameter tuning"""
        from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
        from sklearn.neural_network import MLPRegressor
        
        return {
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, 
                max_depth=None, 
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'random_forest_tuned': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost_tuned': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'ridge_tuned': Ridge(alpha=0.5),
            'elastic_net_tuned': ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42)
        }
    
    def select_features(self, X, y, n_features=None):
        """Advanced feature selection using multiple methods"""
        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.ensemble import RandomForestRegressor
        
        if n_features is None:
            n_features = min(50, X.shape[1] // 2)  # Select half the features or 50, whichever is smaller
        
        # Method 1: Statistical selection
        selector_stats = SelectKBest(score_func=f_regression, k=n_features)
        X_stats = selector_stats.fit_transform(X, y)
        
        # Method 2: Tree-based feature importance
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(X, y)
        feature_importance = rf_selector.feature_importances_
        top_features = np.argsort(feature_importance)[-n_features:]
        
        # Method 3: Recursive Feature Elimination
        rfe_selector = RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=n_features)
        rfe_selector.fit(X, y)
        
        # Combine selections (union of top features from each method)
        stats_features = set(selector_stats.get_support(indices=True))
        importance_features = set(top_features)
        rfe_features = set(np.where(rfe_selector.support_)[0])
        
        # Take intersection of at least 2 methods or top importance if intersection too small
        common_features = stats_features.intersection(importance_features).union(
            stats_features.intersection(rfe_features)).union(
            importance_features.intersection(rfe_features))
        
        if len(common_features) < n_features // 2:
            # Fall back to top importance features
            selected_features = list(importance_features)
        else:
            selected_features = list(common_features)
        
        # Ensure we have enough features
        if len(selected_features) < n_features:
            remaining_needed = n_features - len(selected_features)
            all_features = set(range(X.shape[1]))
            remaining_features = list(all_features - set(selected_features))
            # Add top importance from remaining
            remaining_importance = [(i, feature_importance[i]) for i in remaining_features]
            remaining_importance.sort(key=lambda x: x[1], reverse=True)
            selected_features.extend([i for i, _ in remaining_importance[:remaining_needed]])
        
        selected_features = sorted(selected_features[:n_features])
        return selected_features, X[:, selected_features]
    
    def train_advisor_model(self, grids, ratings, advisor_idx):
        """Train advanced model for specific advisor"""
        advisor_name = self.advisor_names[advisor_idx]
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING ADVANCED MODEL FOR {advisor_name.upper()} ADVISOR")
        print(f"{'='*60}")
        
        # Get labeled data
        mask = ~np.isnan(ratings[:, advisor_idx])
        labeled_grids = grids[mask]
        labeled_ratings = ratings[mask, advisor_idx]
        
        print(f"📊 Training samples: {len(labeled_grids):,}")
        
        # Create specialized features
        print(f"🔧 Creating specialized features for {advisor_name}...")
        features = self.oracle.create_comprehensive_features(labeled_grids, advisor_idx)
        
        # Feature selection
        print(f"🎯 Performing intelligent feature selection...")
        selected_features, X_selected = self.select_features(features, labeled_ratings)
        print(f"✅ Selected {len(selected_features)} most important features from {features.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, labeled_ratings, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store preprocessing info
        self.oracle.scalers[advisor_name] = scaler
        self.oracle.feature_selectors[advisor_name] = selected_features
        
        # Train advanced models
        print(f"🤖 Training advanced ML ensemble...")
        models = self.get_advanced_models()
        
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            try:
                if name in ['ridge_tuned', 'elastic_net_tuned', 'neural_network']:
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                print(f"     ✅ {name:20} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
                
                trained_models[name] = model
                model_scores[name] = test_r2
                
            except Exception as e:
                print(f"     ❌ {name:20} - Failed: {str(e)}")
        
        # Create weighted ensemble
        valid_scores = {k: v for k, v in model_scores.items() if v > 0}
        if not valid_scores:
            print("❌ All models failed! Using simple fallback...")
            # Fallback to simple model
            simple_model = RandomForestRegressor(n_estimators=100, random_state=42)
            simple_model.fit(X_train, y_train)
            test_pred = simple_model.predict(X_test)
            fallback_r2 = r2_score(y_test, test_pred)
            
            self.oracle.models[advisor_name] = {
                'best_model': simple_model,
                'model_type': 'fallback',
                'feature_selector': selected_features,
                'scaler': scaler,
                'performance': fallback_r2
            }
            return fallback_r2
        
        # Ensemble weighting based on performance
        ensemble_weights = np.array(list(valid_scores.values()))
        ensemble_weights = np.maximum(ensemble_weights, 0) ** 2  # Square to emphasize better models
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        
        print(f"\n🎯 Ensemble Composition:")
        for (model_name, score), weight in zip(valid_scores.items(), ensemble_weights):
            print(f"   {model_name:20}: {weight:.3f} weight (R² = {score:.4f})")
        
        # Test ensemble performance
        ensemble_pred = self._ensemble_predict_test(X_test, X_test_scaled, trained_models, 
                                                   dict(zip(valid_scores.keys(), ensemble_weights)))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\n🏆 FINAL ENSEMBLE PERFORMANCE:")
        print(f"   Test R²: {ensemble_r2:.4f}")
        print(f"   Test MAE: {ensemble_mae:.4f}")
        print(f"   Improvement over simple model: {ensemble_r2 - 0.5:.4f}")
        
        # Store final model
        self.oracle.models[advisor_name] = {
            'models': trained_models,
            'ensemble_weights': dict(zip(valid_scores.keys(), ensemble_weights)),
            'feature_selector': selected_features,
            'scaler': scaler,
            'performance': ensemble_r2
        }
        
        return ensemble_r2
    
    def _ensemble_predict_test(self, X, X_scaled, models, weights):
        """Helper to test ensemble prediction"""
        predictions = []
        total_weight = 0
        
        for model_name, weight in weights.items():
            if model_name in models:
                if model_name in ['ridge_tuned', 'elastic_net_tuned', 'neural_network']:
                    pred = models[model_name].predict(X_scaled)
                else:
                    pred = models[model_name].predict(X)
                
                predictions.append(pred * weight)
                total_weight += weight
        
        return np.sum(predictions, axis=0) / total_weight if total_weight > 0 else np.zeros(len(X))


# 🔮 SUPERIOR ORACLE PREDICTION FUNCTION
class SuperiorPredictor:
    """Prediction engine for the superior models"""
    
    def __init__(self, simple_oracle, advanced_oracle, advisor_results):
        self.simple_oracle = simple_oracle
        self.advanced_oracle = advanced_oracle
        self.advisor_results = advisor_results
    
    def predict_advisor(self, grids, advisor_idx):
        """Predict using the best available model for each advisor"""
        advisor_name = self.advanced_oracle.advisor_names[advisor_idx]
        
        # Use advanced model if available and better
        if advisor_name in self.advisor_results:
            print(f"Using SUPERIOR model for {advisor_name}")
            return self._predict_advanced(grids, advisor_idx, advisor_name)
        else:
            print(f"Using simple model for {advisor_name}")
            return self.simple_oracle.predict_all_grids(grids, advisor_idx)
    
    def _predict_advanced(self, grids, advisor_idx, advisor_name):
        """Predict using advanced model"""
        # Create specialized features
        features = self.advanced_oracle.create_comprehensive_features(grids, advisor_idx)
        
        # Apply feature selection
        model_info = self.advanced_oracle.models[advisor_name]
        selected_features = model_info['feature_selector']
        X_selected = features[:, selected_features]
        
        # Scale if needed
        scaler = model_info['scaler']
        X_scaled = scaler.transform(X_selected)
        
        # Make ensemble prediction
        models = model_info['models']
        weights = model_info['ensemble_weights']
        
        predictions = []
        total_weight = 0
        
        for model_name, weight in weights.items():
            if model_name in models:
                if model_name in ['ridge_tuned', 'elastic_net_tuned', 'neural_network']:
                    pred = models[model_name].predict(X_scaled)
                else:
                    pred = models[model_name].predict(X_selected)
                
                predictions.append(pred * weight)
                total_weight += weight
        
        return np.sum(predictions, axis=0) / total_weight if total_weight > 0 else np.zeros(len(grids))


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
    
    def create_all_transportation_features(self, grids):
        """Combine all transportation-specific features"""
        print("🔧 Creating advanced transportation connectivity features...")
        
        connectivity_features = self.create_connectivity_matrix_features(grids)
        flow_features = self.create_flow_optimization_features(grids)
        
        all_features = np.hstack([connectivity_features, flow_features])
        
        print(f"✅ Total transportation features: {all_features.shape[1]}")
        print(f"   - Connectivity: {connectivity_features.shape[1]}")
        print(f"   - Flow optimization: {flow_features.shape[1]}")
        
        return all_features


class TransportationCNNModel:
    """
    Convolutional Neural Network specifically designed for 7x7 city grids
    CNNs are perfect for capturing spatial patterns in transportation networks
    """
    
    def __init__(self):
        # Check if TensorFlow/Keras is available, otherwise use a simpler approach
        try:
            self.tf_available = True
            self.tf = tf
            self.keras = keras
            self.layers = layers
            print("✅ TensorFlow available - using full CNN implementation")
        except ImportError:
            print("⚠️ TensorFlow not available - will use CNN-inspired features instead")
            self.tf_available = False
    
    def create_cnn_model(self, input_shape=(7, 7, 1)):
        """Create CNN architecture optimized for 7x7 grids"""
        if not self.tf_available:
            return None
        
        model = self.keras.Sequential([
            # First Conv layer - detect local patterns (3x3 neighborhoods)
            self.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            self.layers.BatchNormalization(),
            
            # Second Conv layer - detect larger patterns  
            self.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            self.layers.BatchNormalization(),
            
            # Third Conv layer - capture district interactions
            self.layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
            self.layers.Dropout(0.3),
            
            # Global patterns
            self.layers.GlobalAveragePooling2D(),
            
            # Dense layers for final prediction
            self.layers.Dense(256, activation='relu'),
            self.layers.Dropout(0.5),
            self.layers.Dense(128, activation='relu'),
            self.layers.Dropout(0.3),
            self.layers.Dense(64, activation='relu'),
            self.layers.Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_cnn_inspired_features(self, grids):
        """
        If CNN not available, create hand-crafted features inspired by CNN operations
        These mimic what convolutional layers would detect
        """
        n_grids = len(grids)
        features_list = []
        
        # Define conv-like kernels manually
        kernels = {
            'horizontal_edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            'vertical_edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'corner_detector': np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]]),
            'center_surround': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        
        for grid in grids:
            features = []
            
            # Apply each "kernel" to detect patterns
            for kernel_name, kernel in kernels.items():
                responses = []
                
                # Slide kernel over grid (like convolution)
                for i in range(5):  # 7-3+1 = 5
                    for j in range(5):
                        patch = grid[i:i+3, j:j+3]
                        response = np.sum(patch * kernel)
                        responses.append(response)
                
                # Aggregate responses (like pooling)
                features.extend([
                    np.mean(responses),
                    np.std(responses),  
                    np.max(responses),
                    np.min(responses)
                ])
            
            # Additional spatial pattern features
            # District transition counting (edge detection)
            transitions = 0
            for i in range(7):
                for j in range(6):
                    if grid[i, j] != grid[i, j+1]:
                        transitions += 1
            for i in range(6):
                for j in range(7):
                    if grid[i, j] != grid[i+1, j]:
                        transitions += 1
            
            features.append(transitions)
            
            # Local homogeneity (like texture analysis)
            homogeneity_scores = []
            for i in range(5):
                for j in range(5):
                    patch = grid[i:i+3, j:j+3]
                    unique_districts = len(np.unique(patch))
                    homogeneity = 1.0 / unique_districts  # More homogeneous = higher score
                    homogeneity_scores.append(homogeneity)
            
            features.extend([
                np.mean(homogeneity_scores),
                np.std(homogeneity_scores)
            ])
            
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"Created {features_array.shape[1]} CNN-inspired spatial pattern features")
        return features_array
    
    def prepare_cnn_data(self, grids, ratings, advisor_idx=2):
        """Prepare data for CNN training"""
        if not self.tf_available:
            return None, None, None, None
        
        # Get labeled data for transportation advisor
        mask = ~np.isnan(ratings[:, advisor_idx])
        labeled_grids = grids[mask]
        labeled_ratings = ratings[mask, advisor_idx]
        
        # Reshape for CNN (add channel dimension)
        X = labeled_grids.reshape(-1, 7, 7, 1).astype('float32')
        
        # Normalize grid values to 0-1 range
        X = X / 4.0  # Districts are 0-4, so divide by 4
        
        y = labeled_ratings.astype('float32')
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_cnn(self, grids, ratings, advisor_idx=2):
        """Train CNN model for transportation advisor"""
        if not self.tf_available:
            print("⚠️ TensorFlow not available, using CNN-inspired features instead")
            return self.create_cnn_inspired_features(grids[~np.isnan(ratings[:, advisor_idx])])
        
        print("🧠 Training CNN for Transportation advisor...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_cnn_data(grids, ratings, advisor_idx)
        
        # Create and train model
        model = self.create_cnn_model()
        
        # Callbacks for better training
        callbacks = [
            self.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            self.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate R² score
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"🏆 CNN Results:")
        print(f"   Train MAE: {train_mae:.4f}")
        print(f"   Test MAE: {test_mae:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        
        self.model = model
        return model, test_r2
    
    def predict_cnn(self, grids):
        """Make predictions using trained CNN"""
        if not self.tf_available or not hasattr(self, 'model'):
            return None
        
        X = grids.reshape(-1, 7, 7, 1).astype('float32') / 4.0
        predictions = self.model.predict(X)
        return predictions.flatten()


class TransportationOracleV3:
    """
    Ultimate Transportation Oracle combining connectivity features, CNN, and ensemble methods
    """
    
    def __init__(self):
        self.transport_features = AdvancedTransportationFeatures()
        self.cnn_model = TransportationCNNModel()
        self.traditional_models = {}
        self.cnn_trained_model = None
        self.scaler = None
        self.advisor_name = "Transportation"
    
    def create_comprehensive_features(self, grids):
        """Create all features: connectivity + CNN-inspired + basic"""
        # Basic grid features
        n_grids = len(grids)
        grids_flat = grids.reshape(n_grids, -1)
        
        # District counts
        district_counts = np.zeros((n_grids, 5))
        for i in range(5):
            district_counts[:, i] = np.sum(grids_flat == i, axis=1)
        
        # Advanced transportation features
        transport_features = self.transport_features.create_all_transportation_features(grids)
        
        # CNN-inspired spatial features
        spatial_features = self.cnn_model.create_cnn_inspired_features(grids)
        
        # Combine all features
        all_features = np.hstack([grids_flat, district_counts, transport_features, spatial_features])
        
        print(f"🎯 Combined feature breakdown:")
        print(f"   Grid positions: 49")
        print(f"   District counts: 5")
        print(f"   Transportation: {transport_features.shape[1]}")
        print(f"   Spatial patterns: {spatial_features.shape[1]}")
        print(f"   TOTAL: {all_features.shape[1]}")
        
        return all_features
    
    def train_ensemble_models(self, grids, ratings, advisor_idx=2):
        """Train ensemble of traditional models with new features"""
        print(f"🚗 Training Transportation Oracle V3 with advanced features...")
        
        # Get labeled data
        mask = ~np.isnan(ratings[:, advisor_idx])
        labeled_grids = grids[mask]
        labeled_ratings = ratings[mask, advisor_idx]
        
        print(f"📊 Training samples: {len(labeled_grids):,}")
        
        # Create comprehensive features
        features = self.create_comprehensive_features(labeled_grids)
        
        # Feature selection (focus on most predictive features)
        from sklearn.feature_selection import SelectKBest, f_regression
        n_features = min(80, features.shape[1])  # Select top 80 features
        selector = SelectKBest(score_func=f_regression, k=n_features)
        features_selected = selector.fit_transform(features, labeled_ratings)
        
        print(f"🎯 Selected top {n_features} features from {features.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_selected, labeled_ratings, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Advanced models ensemble
        models = {
            'extra_trees_deep': ExtraTreesRegressor(
                n_estimators=500, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'random_forest_deep': RandomForestRegressor(
                n_estimators=500, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'gradient_boost_advanced': GradientBoostingRegressor(
                n_estimators=300, max_depth=10, learning_rate=0.03,
                subsample=0.8, max_features='sqrt', random_state=42
            ),
            'ridge_alpha_tuned': Ridge(alpha=0.1),
            'elastic_net_tuned': ElasticNet(alpha=0.005, l1_ratio=0.8, random_state=42)
        }
        
        # Try to add XGBoost if available
        try:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            print("✅ Added XGBoost to ensemble")
        except ImportError:
            print("⚠️ XGBoost not available")
        
        # Train models
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            try:
                if name in ['ridge_alpha_tuned', 'elastic_net_tuned']:
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                print(f"     ✅ {name:25} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
                
                trained_models[name] = model
                model_scores[name] = test_r2
                
            except Exception as e:
                print(f"     ❌ {name:25} - Failed: {str(e)}")
        
        # Store models and selector
        self.traditional_models = trained_models
        self.feature_selector = selector
        self.model_scores = model_scores
        
        # Create ensemble weights
        valid_scores = {k: v for k, v in model_scores.items() if v > 0}
        if valid_scores:
            ensemble_weights = np.array(list(valid_scores.values()))
            ensemble_weights = np.maximum(ensemble_weights, 0) ** 3  # Cube to really emphasize best models
            ensemble_weights = ensemble_weights / ensemble_weights.sum()
            
            self.ensemble_weights = dict(zip(valid_scores.keys(), ensemble_weights))
            
            print(f"\n🎯 Ensemble weights:")
            for model_name, weight in self.ensemble_weights.items():
                print(f"   {model_name:25}: {weight:.3f}")
            
            # Test ensemble
            ensemble_pred = self._predict_ensemble(X_test, X_test_scaled)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            
            print(f"\n🏆 ENSEMBLE PERFORMANCE:")
            print(f"   Test R²: {ensemble_r2:.4f}")
            print(f"   Test MAE: {ensemble_mae:.4f}")
            
            return ensemble_r2
        else:
            print("❌ All models failed!")
            return 0.0
    
    def train_cnn_model(self, grids, ratings, advisor_idx=2):
        """Train CNN model if TensorFlow is available"""
        if self.cnn_model.tf_available:
            print("🧠 Training CNN component...")
            cnn_model, cnn_r2 = self.cnn_model.train_cnn(grids, ratings, advisor_idx)
            self.cnn_trained_model = cnn_model
            return cnn_r2
        else:
            print("⚠️ CNN training skipped (TensorFlow not available)")
            return 0.0
    
    def _predict_ensemble(self, X, X_scaled):
        """Internal ensemble prediction"""
        predictions = []
        total_weight = 0
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in self.traditional_models:
                model = self.traditional_models[model_name]
                
                if model_name in ['ridge_alpha_tuned', 'elastic_net_tuned']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions.append(pred * weight)
                total_weight += weight
        
        if total_weight > 0:
            return np.sum(predictions, axis=0) / total_weight
        else:
            return np.zeros(len(X))
    
    def predict_all_grids(self, grids):
        """Predict transportation scores for all grids"""
        print(f"🚗 Predicting transportation scores for {len(grids):,} grids...")
        
        # Create features
        features = self.create_comprehensive_features(grids)
        
        # Apply feature selection
        features_selected = self.feature_selector.transform(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features_selected)
        
        # Traditional ensemble prediction
        ensemble_pred = self._predict_ensemble(features_selected, features_scaled)
        
        # CNN prediction (if available)
        if self.cnn_trained_model is not None:
            cnn_pred = self.cnn_model.predict_cnn(grids)
            
            # Combine ensemble and CNN (weighted average)
            # Give more weight to ensemble since it's trained on more features
            final_pred = 0.7 * ensemble_pred + 0.3 * cnn_pred
            print("✅ Combined ensemble + CNN predictions")
        else:
            final_pred = ensemble_pred
            print("✅ Using ensemble predictions only")
        
        return final_pred
    
    def train_complete_model(self, grids, ratings):
        """Train the complete V3 model with all components"""
        print(f"\n{'='*70}")
        print("🚗 TRAINING TRANSPORTATION ORACLE V3 - ULTIMATE MODEL")
        print(f"{'='*70}")
        
        # Train traditional ensemble
        ensemble_r2 = self.train_ensemble_models(grids, ratings)
        
        # Train CNN component
        cnn_r2 = self.train_cnn_model(grids, ratings)
        
        print(f"\n{'='*70}")
        print("🏆 TRANSPORTATION V3 TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Traditional Ensemble R²: {ensemble_r2:.4f}")
        print(f"CNN R²: {cnn_r2:.4f}")
        print(f"Expected Combined Performance: ~{max(ensemble_r2, cnn_r2):.4f}")
        
        return max(ensemble_r2, cnn_r2)
