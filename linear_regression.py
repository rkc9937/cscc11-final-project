from data_format.format import load_formatted_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def prepare_features(df):
    df = df.copy()
    df['neighborhood_id'] = pd.to_numeric(df['neighborhood_id'], errors='coerce')
    df = df.dropna()
    print(f"Rows after dropping NaN: {len(df)}")

    X = df.drop(columns=["NSI_next"])
    y = df["NSI_next"]
    return X, y


def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def analyze_model_performance(y_true, y_pred, model_name, verbose=True):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_test_non_zero = y_true[y_true != 0]
    y_pred_non_zero = y_pred[y_true != 0]
    mape = mean_absolute_percentage_error(y_test_non_zero, y_pred_non_zero) * 100  # Convert to percentage

    if verbose:
        print(f"\n{model_name} Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"r2:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


def print_coefficients(model, feature_names):
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {model.intercept_:.6f}")


def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal fit")

    plt.xlabel("Actual NSI_next")
    plt.ylabel("Predicted NSI_next")
    plt.title("Actual vs Predicted NSI (Linear Regression)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_coefficients(model, feature_names):
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Linear Regression Feature Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()


df = load_formatted_data()

print(df)
print(f"Dataset shape: {df.shape}")
print(f"Feature names: {df.columns}")

X, y = prepare_features(df)
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = process_data(X, y)

model = train_linear_model(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_metrics = analyze_model_performance(y_train, y_train_pred, "Linear Regression (train)", True)
test_metrics = analyze_model_performance(y_test, y_test_pred, "Linear Regression (test)", True)

print_coefficients(model, X.columns)

plot_actual_vs_predicted(y_test, y_test_pred)
plot_feature_coefficients(model, X.columns)
