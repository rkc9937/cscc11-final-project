import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from data_format.format import load_formatted_data

def prepare_features(df):
    """
    Prepare feature matrix X and target vector y.
    Drops non-numeric columns and separates target.
    """
    # Drop non-numeric column (neighborhood_id is categorical)
    numeric_df = df.drop(columns=['neighborhood_id'])
    # Drop rows with any NaN values
    numeric_df = numeric_df.dropna()
    print(f"Rows after dropping NaN: {len(numeric_df)}")

    # Separate features and target
    X = numeric_df.drop(columns=['NSI_next'])
    y = numeric_df['NSI_next']
    return X, y

def train_ridge_regression(X_train, y_train, alpha=1.0):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Filter out near-zero values before calculating MAPE
    y_true_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    non_zero_mask = np.abs(y_true_arr) >= 0.01
    
    if non_zero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(
            y_true_arr[non_zero_mask], 
            y_pred_arr[non_zero_mask]
        ) * 100
    else:
        mape = np.nan

    return {
        'MSE': mse, 
        'RMSE': rmse, 
        'MAE': mae, 
        'R2': r2,
        'MAPE': mape,
        'y_pred': y_pred
    }

def print_coefficients(model, feature_names):
    """Print model coefficients."""
    # If using Pipeline with step name "ridge"
    if hasattr(model, "named_steps"):
        ridge = model.named_steps["ridge"]
    else:
        ridge = model

    print("\nModel Coefficients (Ridge):")
    print("-" * 40)
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": ridge.coef_
    }).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {ridge.intercept_:.6f}")

def tune_ridge_alpha(X_train, y_train):
    """
    Tune Ridge alpha using 5-fold cross-validation on the training set.
    Returns best_alpha and a DataFrame with all alpha results.
    """
    alphas = np.logspace(-3, 3, 20)  # 0.001 ... 1000
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    print("\nAlpha tuning (5-fold CV, scoring = RMSE):")
    print("alpha\tmean_RMSE\tstd_RMSE")
    print("-" * 40)

    for alpha in alphas:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="neg_root_mean_squared_error"
        )
        rmse_scores = -scores  # scores are negative
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()

        results.append({
            "alpha": alpha,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse
        })

        print(f"{alpha:.5f}\t{mean_rmse:.4f}\t\t{std_rmse:.4f}")

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["mean_rmse"].idxmin()]
    best_alpha = float(best_row["alpha"])

    print("\nBest alpha based on CV RMSE:")
    print(best_row)

    return best_alpha, results_df


def main():
    # Load formatted data (same as linear)
    print("Loading formatted data...")
    df = load_formatted_data()
    print(f"Dataset shape: {df.shape}")

    # Prepare features
    X, y = prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures used: {list(X.columns)}")

    # Train-test split (80-20, same as linear)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Tune alpha with K-fold CV on the training set
    print("\nTuning Ridge Regression alpha with 5-fold cross-validation...")
    best_alpha, alpha_df = tune_ridge_alpha(X_train, y_train)

    # Train Ridge
    print(f"\nTraining final Ridge Regression model (alpha={best_alpha:.6f})...")
    model = train_ridge_regression(X_train, y_train, alpha=best_alpha)


    # Evaluate
    print("\n" + "=" * 50)
    print("MODEL EVALUATION (Ridge)")
    print("=" * 50)

    train_results = evaluate_model(model, X_train, y_train)
    print(f"\nTraining Performance:")
    print(f"  R^2 Score: {train_results['R2']:.4f}")
    print(f"  RMSE: {train_results['RMSE']:.4f}")
    print(f"  MAE: {train_results['MAE']:.4f}")
    print(f"  MAPE: {train_results['MAPE']:.2f}%")

    test_results = evaluate_model(model, X_test, y_test)
    print(f"\nTest Performance:")
    print(f"  R^2 Score: {test_results['R2']:.4f}")
    print(f"  RMSE: {test_results['RMSE']:.4f}")
    print(f"  MAE: {test_results['MAE']:.4f}")
    print(f"  MAPE: {test_results['MAPE']:.2f}%")


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

    # Plots
    y_test_arr = y_test.values
    y_pred_arr = test_results['y_pred']

    plt.figure(figsize=(6,6))
    plt.scatter(y_test_arr, y_pred_arr, s=5)
    plt.xlabel("Actual NSI_next")
    plt.ylabel("Predicted NSI_next")
    plt.title("Ridge Regression: Predicted vs Actual")
    plt.plot([y_test_arr.min(), y_test_arr.max()],
            [y_test_arr.min(), y_test_arr.max()],
            linewidth=1)
    plt.tight_layout()
    plt.savefig("ridge_pred_vs_actual.png", dpi=300)

    # Plot alpha tuning results
    plt.figure(figsize=(6,4))
    plt.semilogx(alpha_df["alpha"], alpha_df["mean_rmse"])
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Mean CV RMSE")
    plt.title("Ridge Regression Hyperparameter Tuning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ridge_alpha_tuning.png", dpi=300)

    # Plot Coefficient Magnitude Bar
    ridge = model.named_steps["ridge"]
    coefs = ridge.coef_
    names = X.columns

    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(coefs)), coefs[sorted_idx])
    plt.xticks(range(len(coefs)), names[sorted_idx], rotation=90)
    plt.title("Ridge Regression Coefficient Magnitudes")
    plt.tight_layout()
    plt.savefig("ridge_coeffs.png", dpi=300)


    return model, test_results

if __name__ == "__main__":
    model, results = main()