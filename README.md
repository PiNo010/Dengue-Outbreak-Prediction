# Dengue Outbreak Prediction

# Epidemiological Forecasting: Stacking & Neural Networks

This repository implements a machine learning pipeline to predict disease incidence (expressed as log-incidence per 100,000 population) by integrating epidemiological data with economic and environmental indicators. The project compares advanced strategies including Stacking Regressor model and Neural networks.

## Project Overview
The core objective is to perform time-series forecasting across multiple countries. The workflow consists of:
1. **Pre-processing**: Incidence calculation, logarithmic normalization, and feature engineering (including 1-year lagged variables).
2. **Ensemble Modeling**: A custom `Stacking Regressor` that combines multiple CatBoost models using a meta-learner approach.
3. **Neural Network**: A PyTorch-based Multi-Layer Perceptron (MLP) designed to capture complex non-linear relationships.
4. **Evaluation**: Performance analysis using standard metrics (MSE, MAE, R², Explained Variance) and automated visualization for each country.

## Repository Structure
*   `prediction_clean.py`: The main execution script. It handles data loading, model training, and result generation.
*   `CustomStackRegressor.py`: Implements the stacking logic using K-Fold cross-validation to train the meta-model.
*   `Model.py`: Defines the MLP neural network architecture using PyTorch.
*   `train.py`: Utility functions for the Deep Learning training loop (optimizer, loss functions, and hardware acceleration).


## How it Works
The system automatically splits the data into two periods:
*   **Training Set**: Data prior to the year 2024.
*   **Test Set**: Data from 2024 onwards (used to validate predictive accuracy on "future" scenarios).

