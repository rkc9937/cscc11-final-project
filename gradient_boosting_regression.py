import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


DATA_FILE = 'data/mci_formatted.pkl'
TARGET_COLUMN = 'NSI_next'
ID_COLUMN = 'neighborhood_id'
TIME_COLUMN_MONTH = 'report_month'
COLUMNS_TO_DROP = ['NSI_next'] 
RANDOM_STATE = 42
N_JOBS = -1 

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

        X_encoded = pd.get_dummies(X, columns=[ID_COLUMN], prefix=ID_COLUMN)

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
    
    param_grid = {
        'n_estimators': [50, 100, 200, 1000],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],        
    }
    
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=gbr, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2, 
        n_jobs=N_JOBS 
    )
    grid_search.fit(X_train, y_train)
    print("\n--- Grid Search Complete ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best Cross-Validation RMSE: {best_rmse:.4f}")
    
    return grid_search.best_params_


def train_and_predict_nsi(X, y, data_index, df_cleaned, best_params=None):
    """
    Trains the GBR model (optionally with best_params), makes predictions, and evaluates.
    """
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data_index, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
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
    print("Starting model training...")
    gbr.fit(X_train, y_train)
    print("Training complete.")
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Overall Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    results_df = pd.DataFrame({
        'Actual_NSI_next': y_test,
        'Predicted_NSI_next': y_pred
    }, index=idx_test)
    
    original_df_index_map = df_cleaned.loc[df_cleaned.index.intersection(idx_test), ID_COLUMN].to_dict()
    results_df[ID_COLUMN] = results_df.index.map(original_df_index_map.get)    
    neighborhood_results = results_df.groupby(ID_COLUMN).agg(
        Count=('Actual_NSI_next', 'size'),
        Avg_Actual_NSI=('Actual_NSI_next', 'mean'),
        Avg_Predicted_NSI=('Predicted_NSI_next', 'mean')
    ).sort_values(by='Count', ascending=False)

    print("\n--- Sample Predictions per Neighborhood (Test Set) ---")
    print("Shows the average predicted NSI_next for each neighborhood in the test set.")
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