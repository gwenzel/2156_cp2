"""
Setup script for Challenge Problem 2 - City Grid Optimization
Easy installation: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="cp2-city-grid",
    version="1.0.0",
    description="CNN-based oracle predictors for city grid optimization",
    author="Challenge Problem 2 Team",
    python_requires=">=3.8",
    
    # Automatically find packages (utils, etc.)
    packages=find_packages(),
    
    # Core dependencies
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
    ],
    
    # Optional dependencies for deep learning
    extras_require={
        "torch": [
            "torch>=1.10.0",
            "torchvision>=0.11.0",
        ],
        "tensorflow": [
            "tensorflow>=2.8.0",
        ],
    },
    
    # Include data files
    include_package_data=True,
    
    # Development tools
    zip_safe=False,
)
