import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# --- CONFIGURATION ---
DATA_FILE = 'data/mci_formatted.pkl'
TARGET_COLUMN = 'NSI_next'
ID_COLUMN = 'neighborhood_id'
TIME_COLUMN_MONTH = 'report_month'
COLUMNS_TO_DROP = ['NSI_next'] 
RANDOM_STATE = 42
N_JOBS = -1 # Use all available cores for parallel processing

def load_and_preprocess_data(filepath):
    """
    Loads data, cleans, and One-Hot Encodes the neighborhood ID for use in the GBR model, 
    while dropping rows where neighborhood_id is 'NSA'. Keeps report_month as a linear numeric feature.
    """
    try:
        df = pd.read_pickle(filepath) 
        print(f"Original data shape: {df.shape}")

        y = df[TARGET_COLUMN]
        X = df.drop(columns=COLUMNS_TO_DROP)

        # 5. One-Hot Encoding for the ID_COLUMN (Spatial Context)
        X_encoded = pd.get_dummies(X, columns=[ID_COLUMN], prefix=ID_COLUMN)

        # The report_month column is now included in X_encoded as a numeric feature (1 to 12).
        print(f"Final data shape for training: X={X_encoded.shape}, y={y.shape}")
        return X_encoded, y, df.index, df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it exists.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during data loading and preprocessing: {e}")
        return None, None, None, None


def perform_grid_search(X_train, y_train):
    """
    Performs Grid Search Cross-Validation to find the optimal hyperparameters 
    for the GradientBoostingRegressor. 
    """
    print("\n--- Starting Hyperparameter Grid Search (5-Fold CV) ---")
    
    # Define a smaller, faster grid for initial tuning.
    param_grid = {
        'n_estimators': [50, 100, 200],         # Number of trees
        'learning_rate': [0.05, 0.1, 0.2],      # Shrinkage factor
        'max_depth': [3, 4, 5],                 # Depth of each tree
    }
    
    # Initialize the base model
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    
    # We use 'neg_mean_squared_error' as the scoring metric for regression.
    grid_search = GridSearchCV(
        estimator=gbr, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error',
        cv=5, # 5-fold cross-validation
        verbose=2, 
        n_jobs=N_JOBS # Use all cores
    )
    
    # Execute the search
    grid_search.fit(X_train, y_train)
    
    # Report the best parameters
    print("\n--- Grid Search Complete ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    # Convert best negative MSE back to positive RMSE
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best Cross-Validation RMSE: {best_rmse:.4f}")
    
    return grid_search.best_params_


def train_and_predict_nsi(X, y, data_index, df_cleaned, best_params=None):
    """
    Trains the GBR model (optionally with best_params), makes predictions, and evaluates.
    """
    # 6. Split data into training and testing sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data_index, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 7. Initialize the Gradient Boosting Regressor
    if best_params:
        print("\nInitializing GBR with optimized parameters.")
        gbr = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    else:
        print("\nInitializing GBR with default parameters (n_estimators=100, learning_rate=0.1, max_depth=3).")
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    
    # 8. Train the model
    print("Starting model training...")
    gbr.fit(X_train, y_train)
    print("Training complete.")

    # 9. Make predictions on the test set
    y_pred = gbr.predict(X_test)

    # 10. Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Overall Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

    # 11. Create a results DataFrame
    results_df = pd.DataFrame({
        'Actual_NSI_next': y_test,
        'Predicted_NSI_next': y_pred
    }, index=idx_test)
    
    original_df_index_map = df_cleaned.loc[df_cleaned.index.intersection(idx_test), ID_COLUMN].to_dict()
    results_df[ID_COLUMN] = results_df.index.map(original_df_index_map.get)
    
    # Group results by neighborhood_id to show average performance for each area
    neighborhood_results = results_df.groupby(ID_COLUMN).agg(
        Count=('Actual_NSI_next', 'size'),
        Avg_Actual_NSI=('Actual_NSI_next', 'mean'),
        Avg_Predicted_NSI=('Predicted_NSI_next', 'mean')
    ).sort_values(by='Count', ascending=False)

    print("\n--- Sample Predictions per Neighborhood (Test Set) ---")
    print("Shows the average predicted NSI_next for each neighborhood in the test set.")
    # NOTE: The .to_markdown() function requires the optional dependency 'tabulate'.
    # If an ImportError occurs, please run: pip install tabulate
    print(neighborhood_results.head(10).to_markdown(index=True, floatfmt=".4f"))
    
    return neighborhood_results


if __name__ == "__main__":
    X_encoded, y, data_index, df_cleaned = load_and_preprocess_data(DATA_FILE)
    
    if X_encoded is not None and X_encoded.shape[0] > 0:
        # 1. Split data once for tuning and final evaluation
        X_full_train, X_final_test, y_full_train, y_final_test, idx_full_train, idx_final_test = train_test_split(
            X_encoded, y, data_index, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # 2. Perform Hyperparameter Tuning on the training data
        optimal_params = perform_grid_search(X_full_train, y_full_train)

        # 3. Retrain the model using the optimal parameters and evaluate on the final test set
        print("\n--- Retraining with Optimal Parameters and Final Evaluation ---")
        train_and_predict_nsi(X_encoded, y, data_index, df_cleaned, best_params=optimal_params)
    else:
        print("\nCould not proceed with training due to data loading or cleaning errors.")