# SOM-CSC: Self-Organizing Maps for Chandra Source Catalog

A web application for exploring and analyzing the Chandra Source Catalog (CSC) using Self-Organizing Maps (SOM). This application provides a user-friendly interface for training, visualizing, and interpreting SOMs with astronomical data.

## Overview

SOM-CSC is a Streamlit-based web application that allows astronomers and researchers to:

1. Explore CSC data using the powerful Self-Organizing Maps technique
2. Train custom SOMs with different features and hyperparameters
3. Select different datasets
4. Visualize data patterns and clusters in an interactive interface
5. Classify astronomical sources based on their characteristics

## Features

### Data Handling
- Loading and preprocessing of CSC datasets
- Support for pre-normalized log data
- Feature selection for SOM training
- Data import and export in CSV format

### SOM Training
- Configurable SOM parameters:
  - Map dimensions (square grids)
  - Sigma (neighborhood spread)
  - Learning rate
  - Number of iterations
  - Topology (hexagonal or rectangular)
  - Random seed for reproducibility
- Training progress monitoring
- Error tracking (quantization and topographic error)

### Visualization
- **U-Matrix**: Visualize distances between neurons to identify clusters
- **Activation Response**: Display regions of SOM grid activated by different data points
- **Training Feature Space Map**: Visualize feature space distribution
- **Source Name Visualization**: Map named sources in the SOM grid
- **Source Dispersion Visualization**: Visualize the dispersion of sources across the map
- **Main Type Visualization**: Display the distribution of source types (as classified by SIMBAD)
- **Feature Visualization**: Show distribution of specific features not used during training across the map
- Multiple color schemes and scaling options (linear/logarithmic)
- Support for both hexagonal and rectangular topologies

### Model Management
- Save and load trained SOM models
- Download models for offline analysis

### Source Classification
- Source type classification based on SOM mapping
- Configurable classification parameters
- 2 type of classification methods

## Technical Details

### Dependencies
- Streamlit (web application framework)
- MiniSom (Self-Organizing Map implementation)
- NumPy and Pandas (data handling)
- Matplotlib and Altair (visualization)
- Scikit-learn (for additional data processing)

## Data Sources

The application uses data from the Chandra Source Catalog (CSC), which contains information about X-ray sources detected by the Chandra X-ray Observatory. The data has been preprocessed and cross-matched with the SIMBAD astronomical database for source classification. 