# Challenge Problem 2 - City Grid Optimization

## Overview
Train CNN-based oracles to predict city planning advisor scores and generate optimized 7x7 city grids for maximum diversity and performance.

## � Quick Start

Run the notebooks in order:

### 1. Train Oracle Models (Optional - Models Included)
```bash
# Train CNN predictors for each advisor
0_fit_oracle_wellness.ipynb      # Wellness advisor
0_fit_oracle_tax.ipynb           # Tax advisor  
0_fit_oracle_transportation.ipynb # Transportation advisor
0_fit_oracle_business.ipynb      # Business advisor
```

### 2. Evaluate All Oracles
```bash
1_oracle_all.ipynb              # Apply all oracles to 500k grid dataset
```

### 3. Generate New Grids
```bash
3_grid_generation_compact.ipynb  # Generate bias-free grids with constraints
```

### 4. Build Submission
```bash
4_calculate_score_build_submission.ipynb  # Select diverse, high-scoring grids
```

### 5. Analysis (Optional)
```bash
2_identify_valid.ipynb          # Grid validation analysis
5_heatmap_analysis.ipynb        # Frequency heatmaps
```

## 🏗️ What It Does

### Oracle Models
- **CNN Predictors**: Trained on 500k city grids to predict advisor scores
- **4 Advisors**: Wellness, Tax, Transportation, Business  
- **Architecture**: CityCNN1Plus (deep) for difficult advisors, CityCNN1 (standard) for others

### Grid Generation
- **Constraint-Based**: Avoids unwanted bias patterns (corner offices, etc.)
- **Diversity Optimization**: Maximizes variety using Hamming distance
- **Quality Filtering**: Keeps only high-scoring grids (configurable thresholds)

## � Key Files

### Data
- `2155-Challenge-Problem-2/datasets/grids_*.npy` - Original 500k city grids
- `data/generated_grids/` - Generated constraint-free grids
- `data/models/` - Trained oracle models (.pkl files)

### Outputs
- `data/submissions/*_submission.npy` - Final optimized 100-grid selections

## ⚙️ Requirements

- **Python**: NumPy, PyTorch, Pandas, Matplotlib
- **Hardware**: 16GB+ RAM, GPU recommended for training
- **Data**: ~500k city grids, ~50GB disk space

## 🎛️ Configuration

### Grid Selection (Notebook 4)
```python
# Adjust thresholds per advisor
ADVISOR_THRESHOLDS = {
    'Wellness': 0.75,      # Easier advisor
    'Tax': 0.90,          # Harder advisor - raise bar  
    'Transportation': 0.85,
    'Business': 0.80
}

# Enable/disable generated grids
USE_GENERATED_GRIDS = True  # Set False for original grids only
```

### Grid Generation (Notebook 3)  
```python
# Choose constraint types to avoid
CONSTRAINTS_TO_GENERATE = [
    'corner_offices_top',     # No offices at top corners
    'unwanted_residential_left',  # Specific residential constraints
    # ... add more constraint patterns
]
```

## 🔧 Troubleshooting

- **Import Errors**: All utilities moved to `utils/` folder
- **Memory Issues**: Reduce batch sizes or grid counts
- **No Generated Grids**: Run notebook 3 first or set `USE_GENERATED_GRIDS=False`
- **Low Scores**: Adjust advisor thresholds in notebook 4