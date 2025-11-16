"""
Script to train and save the ML model for house price prediction.

This script:
1. Loads the California housing dataset
2. Splits data into training and test sets
3. Trains a RandomForestRegressor model
4. Evaluates the model performance
5. Saves the trained model to disk
"""

# Standard library imports
import os
from pathlib import Path

# Third-party imports for ML
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_data():
    """
    Load the California housing dataset.

    Returns:
        tuple: (X, y, feature_names) where:
            - X: Feature matrix (20,640 samples × 8 features)
            - y: Target values (house prices in $100,000s)
            - feature_names: Names of the 8 features
    """
    print("Loading California housing dataset...")

    # Fetch the dataset from scikit-learn
    # This downloads data the first time, then caches it locally
    housing = fetch_california_housing()

    # Extract features (X) and target (y)
    # X is a 2D array: rows = houses, columns = features
    X = housing.data

    # y is a 1D array: median house values for each district
    y = housing.target

    # Get the feature names for reference
    feature_names = housing.feature_names

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        X: Feature matrix
        y: Target values
        test_size: Fraction of data to use for testing (default: 0.2 = 20%)
        random_state: Random seed for reproducibility (like using a fixed seed in C#)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")

    # Split the data randomly
    # random_state=42 ensures we get the same split every time (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test
