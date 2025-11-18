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


def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        Trained model
    """
    print("\nTraining RandomForestRegressor model...")

    # Create the model with specific parameters
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees in the forest
        max_depth=15,          # Maximum depth of each tree
        random_state=42,       # For reproducibility
        n_jobs=-1,             # Use all CPU cores for parallel training
        verbose=1              # Print progress during training
    )

    # Train the model (this is where the "learning" happens!)
    # The model analyzes the relationship between X_train and y_train
    model.fit(X_train, y_train)

    print("Model training complete!")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True target values for test set

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\nEvaluating model on test set...")

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    # These tell us how good our model is

    # R² Score: How much variance in the data the model explains
    # Range: 0 to 1, closer to 1 is better
    # 0.8 means model explains 80% of the variance
    r2 = r2_score(y_test, y_pred)

    # Mean Absolute Error: Average absolute difference between prediction and actual
    # In dollars (since target is in $100,000s, we multiply by 100,000)
    mae = mean_absolute_error(y_test, y_pred)

    # Root Mean Squared Error: Like MAE but penalizes large errors more
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print the results
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"R² Score:              {r2:.4f}")
    print(f"Mean Absolute Error:   ${mae * 100000:,.2f}")
    print(f"Root Mean Squared Error: ${rmse * 100000:,.2f}")
    print(f"{'='*50}\n")

    # Return metrics as a dictionary
    metrics = {
        "r2_score": r2,
        "mae": mae,
        "rmse": rmse
    }

    return metrics


def save_model(model, model_dir="../models", model_name="house_price_model.joblib"):
    """
    Save the trained model to disk.

    Args:
        model: Trained model to save
        model_dir: Directory to save the model (relative to script location)
        model_name: Name of the model file

    Returns:
        str: Path where model was saved
    """
    print("\nSaving model to disk...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Build the full path to the models directory
    # __file__ is the current script path
    # .parent goes up one level
    # / model_dir goes to the models folder
    models_path = script_dir / model_dir

    # Create the directory if it doesn't exist
    # exist_ok=True means "don't error if it already exists"
    models_path.mkdir(parents=True, exist_ok=True)

    # Full path to the model file
    model_file_path = models_path / model_name

    # Save the model using joblib
    # joblib is optimized for saving large numpy arrays (better than pickle)
    joblib.dump(model, model_file_path)

    print(f"Model saved successfully to: {model_file_path}")

    return str(model_file_path)


def main():
    """
    Main function that orchestrates the entire training pipeline.
    """
    print("="*60)
    print("HOUSE PRICE PREDICTION MODEL - TRAINING PIPELINE")
    print("="*60)

    # Step 1: Load the data
    X, y, feature_names = load_data()

    # Step 2: Split into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Train the model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Save the model
    model_path = save_model(model)

    # Print final summary
    print("="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved at: {model_path}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print("="*60)


# This is Python's entry point (like Main() in C#)
# Runs only when script is executed directly, not when imported
if __name__ == "__main__":
    main()
