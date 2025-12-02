"""
Linear Regression Model for NSI Prediction

Predicts next month's Neighborhood Safety Index (NSI_next) using 
spatial, temporal, and lagged crime features.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from data_format.format import load_formatted_data

def prepare_features(df):
    """
    Prepare feature matrix X and target vector y.
    Converts neighborhood_id to numeric.
    """
    df = df.copy()
    
    # Convert neighborhood_id to numeric (they are numeric strings like '002', '013')
    df['neighborhood_id'] = pd.to_numeric(df['neighborhood_id'], errors='coerce')
    
    # Drop rows with any NaN values
    df = df.dropna()
    print(f"Rows after dropping NaN: {len(df)}")
    
    # Separate features and target
    X = df.drop(columns=['NSI_next'])
    y = df['NSI_next']
    
    return X, y


def train_linear_regression(X_train, y_train):
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # from sklearn docs
    # # the value when some element of the y_true is zero is arbitrarily high because
    # of the division by epsilon
    # remove 0 from y_true to avoid this issue
    y_test_non_zero = y_test[y_test != 0]
    y_pred_non_zero = y_pred[y_test != 0]
    mape = mean_absolute_percentage_error(y_test_non_zero, y_pred_non_zero) * 100  # Convert to percentage
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'y_pred': y_pred,
        'MAPE': mape
    }


def print_coefficients(model, feature_names):
    """Print model coefficients."""
    print("\nModel Coefficients:")
    print("-" * 40)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {model.intercept_:.6f}")


def main():
    # Load formatted data
    print("Loading formatted data...")
    df = load_formatted_data()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features
    X, y = prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures used: {list(X.columns)}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining Linear Regression model...")
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Training performance
    train_results = evaluate_model(model, X_train, y_train)
    print(f"\nTraining Performance:")
    print(f"  R² Score: {train_results['R2']:.4f}")
    print(f"  RMSE: {train_results['RMSE']:.4f}")
    print(f"  MAE: {train_results['MAE']:.4f}")
    print(f"  MAPE: {train_results['MAPE']:.4f}")
    
    # Test performance
    test_results = evaluate_model(model, X_test, y_test)
    print(f"\nTest Performance:")
    print(f"  R² Score: {test_results['R2']:.4f}")
    print(f"  RMSE: {test_results['RMSE']:.4f}")
    print(f"  MAE: {test_results['MAE']:.4f}")
    print(f"  MAPE: {test_results['MAPE']:.4f}")
    
    # Print coefficients
    print_coefficients(model, X.columns)
    
    # Sample predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS (first 10 test samples)")
    print("=" * 50)
    sample_df = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': test_results['y_pred'][:10],
        'Error': y_test.values[:10] - test_results['y_pred'][:10]
    })
    print(sample_df.to_string(index=False))
    
    return model, test_results


if __name__ == "__main__":
    model, results = main()
