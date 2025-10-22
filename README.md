# 2156_cp2

This is a Python project containing the 2155 Challenge Problem 2 materials for urban planning optimization using machine learning.

## Project Overview

This project provides tools and a Jupyter notebook for solving an urban planning challenge problem. You'll work with:
- A large dataset of zoning configurations
- Utility functions for visualization and analysis
- A starter notebook to guide you through the problem

## Setup

### Create the environment

Using **Mamba** (recommended):

```bash
mamba env create -f env.yml
```

Or using **Conda**:

```bash
conda env create -f env.yml
```

### Activate the environment

```bash
conda activate cp2
```

> Tip: You can rename the environment in `env.yml` under the `name:` field.

## Getting Started

1. Activate the conda environment as described above
2. Start Jupyter: `jupyter notebook`
3. Open `starter_notebook.ipynb` and follow along

## Project Structure

- `starter_notebook.ipynb` - Main Jupyter notebook for the challenge problem
- `utils_public.py` - Utility functions for data loading, visualization, and analysis
- `datasets/` - Contains grid configurations and scores data
- `assets/` - Image assets for visualization
- `env.yml` - Conda environment specification

## Credits

Based on the original project: https://github.com/Lyleregenwetter/2155-Challenge-Problem-2