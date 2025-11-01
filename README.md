# Challenge Problem 2 - Oracle Training Pipeline

## Overview
This repository contains an advanced Oracle training pipeline for the 2156 Challenge Problem 2. The system uses enhanced CNN-based oracles with CP02_v10 optimizations to predict grid scores across multiple advisor domains (Business, Tax, Transportation, Wellness).

## 🔄 Step-by-Step Execution Guide

### Phase 1: Train Individual Oracles

Run each oracle training notebook sequentially to train specialized predictors for each advisor:

#### 1.1 Wellness Oracle Training
```bash
# Run the wellness oracle notebook
jupyter notebook 0_fit_oracle_wellness.ipynb
```
- **Purpose**: Trains CNN oracle for wellness advisor predictions
- **Input**: Labeled grid datasets with wellness scores  
- **Output**: Trained wellness oracle model (saved to disk)
- **Features**: Uses CityCNN1Plus architecture (deep model for advisor 0)


#### 1.2 Tax Oracle Training  
```bash
# Run the tax oracle notebook
jupyter notebook 0_fit_oracle_tax.ipynb
```
- **Purpose**: Trains CNN oracle for tax advisor predictions
- **Input**: Labeled grid datasets with tax scores
- **Output**: Trained tax oracle model (saved to disk)
- **Features**: Uses CityCNN1 architecture (standard model for advisor 1)

#### 1.3 Transportation Oracle Training
```bash
# Run the transportation oracle notebook
jupyter notebook 0_fit_oracle_transportation.ipynb
```
- **Purpose**: Trains CNN oracle for transportation advisor predictions  
- **Input**: Labeled grid datasets with transportation scores
- **Output**: Trained transportation oracle model (saved to disk)
- **Features**: Uses CityCNN1Plus architecture (deep model for difficult advisor 2)

#### 1.4 Business Oracle Training
```bash
# Run the business oracle notebook
jupyter notebook 0_fit_oracle_business.ipynb
```
- **Purpose**: Trains CNN oracle for business advisor predictions
- **Input**: Labeled grid datasets with business scores
- **Output**: Trained business oracle model (saved to disk)
- **Features**: Uses CityCNN1 architecture (standard model for advisor 3)


### Phase 2: Apply All Oracles to Full Dataset

#### 2.1 Oracle Ensemble Evaluation
```bash
# Run the comprehensive oracle evaluation
jupyter notebook 1_oracle_all.ipynb
```
- **Purpose**: Loads all trained oracles and applies them to the complete 500k grid dataset
- **Input**: 
  - All trained oracle models from Phase 1
  - Complete 500,000 grid dataset (`grids_0.npy` to `grids_4.npy`)
- **Output**: 
  - Comprehensive predictions across all advisors
  - Performance metrics and validation results
  - Oracle prediction arrays for downstream use
- **Features**:
  - Ensemble prediction combining all oracles
  - Test-Time Augmentation (TTA) for improved accuracy
  - Cross-advisor performance analysis

### Phase 3: Grid Validation Analysis

#### 3.1 Valid Grid Identification
```bash
# Run grid validity analysis
jupyter notebook 2_identify_valid.ipynb
```
- **Purpose**: Analyzes and identifies which grids are valid/invalid according to city planning rules
- **Input**: 
  - Grid datasets from Phase 2
  - Oracle predictions
  - City planning validation rules
- **Output**:
  - Valid grid mask arrays
  - Validation statistics and reports
  - Grid quality metrics
- **Features**:
  - Constraint validation (zoning, adjacency, etc.)
  - Statistical analysis of grid validity patterns
  - Identification of common invalid configurations

### Phase 4: Grid Generation

#### 4.1 New Grid Generation
```bash
# Run grid generation pipeline
jupyter notebook 3_grid_generation.ipynb
```
- **Purpose**: Generates new high-quality grid configurations using trained oracles as guidance
- **Input**:
  - Trained oracle models from Phase 1
  - Valid grid patterns from Phase 3
  - Generation parameters and constraints
- **Output**:
  - Generated grid configurations (`generated_grids.npy`)
  - Generation quality metrics
  - Diversity statistics for generated grids
- **Features**:
  - Oracle-guided grid synthesis
  - Constraint-aware generation
  - Quality filtering and validation
  - Diversity optimization during generation

### Phase 5: Submission Optimization

#### 5.1 Grid Selection and Submission Building
```bash
# Run final optimization and submission creation
jupyter notebook 4_calculate_score_build_submission.ipynb
```
- **Purpose**: Post-processes original and generated grids, optimizes selection for maximum diversity, and builds final submission
- **Input**:
  - Original 500k grid dataset
  - Generated grids from Phase 4
  - Oracle predictions for all grids
  - Valid grid masks from Phase 3
- **Output**:
  - Optimized grid selection (`submission.npy`)
  - Diversity metrics and analysis
  - Final submission statistics
- **Features**:
  - **Diversity Optimization**: Uses Hamming distance, Farthest Point Sampling (FPS)
  - **Multi-Objective Selection**: Balances high scores with maximum diversity
  - **Hill Climbing**: Local optimization for diverse grid combinations
  - **Constraint Enforcement**: Ensures all selected grids meet validity requirements
  - **Performance Analytics**: Detailed analysis of final selection quality

## 🏗️ Architecture Overview

### Enhanced Oracle Features (CP02_v10 Optimizations)

Each oracle incorporates advanced training techniques:

- **Weighted Loss Function**: Emphasizes hard examples near decision boundaries
- **Multi-Seed Ensemble**: Trains multiple models with seeds [42, 43, 44] for robust predictions
- **Test-Time Augmentation**: Uses 6 spatial augmentations for improved inference accuracy
- **Automatic Architecture Selection**: CityCNN1Plus for difficult advisors (0,2), CityCNN1 for others
- **Advanced Hyperparameters**: 250 epochs, weighted loss (γ=0.75), ensemble averaging

### Model Architecture Details

```python
# Deep Advisors (Wellness=0, Transportation=2)
CityCNN1Plus: Deeper CNN with 3 conv blocks, dropout, batch normalization

# Standard Advisors (Tax=1, Business=3)  
CityCNN1: Compact CNN with 2 conv blocks, efficient for standard tasks
```

## 📊 Expected Outputs

### After Phase 1:
- `business_oracle.pkl` - Trained business prediction model
- `tax_oracle.pkl` - Trained tax prediction model
- `transportation_oracle.pkl` - Trained transportation prediction model  
- `wellness_oracle.pkl` - Trained wellness prediction model

### After Phase 2:
- `oracle_predictions.npy` - All oracle predictions on 500k dataset
- `oracle_performance_metrics.json` - Cross-validation results
- `ensemble_predictions.npy` - Combined oracle predictions

### After Phase 3:
- `valid_grid_mask.npy` - Boolean mask of valid grids
- `validation_report.json` - Grid validity statistics
- `constraint_analysis.csv` - Detailed constraint violation analysis

### After Phase 4:
- `generated_grids.npy` - New grid configurations
- `generation_metrics.json` - Quality statistics for generated grids
- `diversity_analysis.json` - Diversity measurements

### After Phase 5:
- `submission.npy` - Final optimized grid selection
- `diversity_optimization_log.json` - Selection optimization details
- `submission_analysis.json` - Final performance and diversity metrics

## 🔧 Technical Requirements

### Dependencies:
- PyTorch (for CNN training)
- NumPy (for grid processing)
- Pandas (for data analysis)
- Scikit-learn (for metrics)
- Matplotlib/Seaborn (for visualization)

### Hardware:
- GPU recommended for CNN training (CUDA support)
- Minimum 16GB RAM for 500k grid processing
- ~50GB disk space for models and intermediate results

## 🚀 Quick Start

```bash
# 1. Train all oracles
jupyter notebook 1_oracle_business.ipynb         # Train business oracle
jupyter notebook 1_oracle_tax.ipynb             # Train tax oracle  
jupyter notebook 1_oracle_transportation.ipynb  # Train transport oracle
jupyter notebook 1_oracle_wellness.ipynb        # Train wellness oracle

# 2. Apply to full dataset
jupyter notebook 1_oracle_all.ipynb             # Evaluate all oracles on 500k grids

# 3. Validate grids
jupyter notebook 2_identify_valid.ipynb         # Analyze grid validity

# 4. Generate new grids
jupyter notebook 3_grid_generation.ipynb        # Generate new high-quality grids

# 5. Build optimized submission
jupyter notebook 4_calculate_score_build_submission.ipynb  # Optimize selection & create submission
```

## 📈 Performance Expectations

With CP02_v10 enhancements, expect:
- **Training R²**: 0.85-0.95 across advisors
- **Validation R²**: 0.75-0.90 across advisors
- **Ensemble Improvement**: +5-10% over single models
- **TTA Improvement**: +2-5% over standard inference

## 🎯 Diversity Optimization Details

### Phase 5 Advanced Features:

#### Multi-Objective Optimization:
- **High Score Selection**: Prioritizes grids with best oracle predictions
- **Diversity Maximization**: Uses Hamming distance to ensure grid variety
- **Constraint Satisfaction**: Validates all selections meet city planning rules

#### Optimization Algorithms:
```python
# Farthest Point Sampling (FPS)
diverse_indices = greedy_fps_selection(grids, scores, k=50)

# Hill Climbing Local Optimization  
optimized_selection = hill_climbing_diversity(grids, initial_selection, max_iterations=1000)

# Combined Multi-Stage Approach
final_submission = select_diverse_top_grids(
    original_grids + generated_grids, 
    all_scores, 
    k=submission_size,
    use_fps=True, 
    use_hill_climbing=True
)
```

#### Diversity Metrics:
- **Hamming Distance**: Measures grid configuration differences
- **Spatial Diversity**: Ensures variety in building placements
- **Functional Diversity**: Balances different advisor score profiles
- **Coverage Analysis**: Maximizes exploration of solution space

## 🔍 Troubleshooting

### Common Issues:
1. **GPU Memory**: Reduce batch_size if CUDA out of memory
2. **Training Time**: Each oracle takes ~10-30 minutes on GPU
3. **Disk Space**: Monitor space during 500k grid processing (Phase 2) and generation (Phase 4)
4. **Model Loading**: Ensure all Phase 1 models saved before Phase 2
5. **Generation Time**: Phase 4 can take several hours depending on generation parameters
6. **Optimization Complexity**: Phase 5 diversity optimization scales with dataset size - use sampling for large datasets