from data_format.format import load_formatted_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def process_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) 
    
    # Verify standardization
    print(f"\nBefore scaling - Training set mean: {X_train.mean(axis=0)[:3]}")
    print(f"Before scaling - Training set std: {X_train.std(axis=0)[:3]}")
    print(f"After scaling - Training set mean: {X_train_scaled.mean(axis=0)[:3]}")
    print(f"After scaling - Training set std: {X_train_scaled.std(axis=0)[:3]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_knn_model(X_train, y_train, k_values, cv_folds):
    cv_scores = []
    cv_std = []
    
    print("Cross-validation results:")
    print("k\tMean Score\tStd Score")
    print("-" * 30)
    
    for k in k_values:
        # Create KNN regressor with current k
        knn = KNeighborsRegressor(n_neighbors=k)
        
        # Perform cross-validation
        scores = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=cv_folds, scoring='neg_root_mean_squared_error')
        
        # Get RMSE scores
        rmse_scores = scores*-1
        
        cv_scores.append(rmse_scores.mean())
        cv_std.append(rmse_scores.std())
        
        print(f"{k}\t{rmse_scores.mean():.4f}\t\t{rmse_scores.std():.4f}")
    
    # Find optimal k
    scatter = plt.scatter(k_values, cv_scores)
    plt.show()

    optimal_k = 10
    
    print(f"\nOptimal k: {optimal_k}")
    print(f"Best CV RMSE: {rmse_scores[3]}")
    
    # Train final model with optimal k
    final_model = KNeighborsRegressor(10)
    final_model.fit(X_train,y_train)
    
    # Store training results
    training_results = {
        'final_model': final_model,
        'cv_std': cv_std   
    }
    
    return optimal_k, cv_scores, training_results

def analyze_model_performance(y_true, y_pred, model_name, verbose):
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true,y_pred)
    
    if verbose:
        print(f"\n{model_name} Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")  
        print(f"r2:   {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

df = load_formatted_data()

print(df)

#X_df = pd.DataFrame(df.data, columns = df.feature_names)
#y_df = pd.DataFrame(df.target, columns = df.target_names)

#combined_df = pd.concat([X_df, y_df], axis = 1)
combined_df = df
#features_df

print(f"Dataset shape: {combined_df.shape}")
print(f"Feature names: {combined_df.columns}")
print(f"Target name: {combined_df.index}")

print(combined_df.head())

X = df.drop(columns=["NSI_next"])
y = df[["NSI_next"]]

fig, axes = plt.subplots(7, 3, figsize=(15, 12))
fig.suptitle('Distribution of Features and Target Variable', fontsize=16)

columns = combined_df.columns

for i, col in enumerate(columns):
    row, col_idx = i // 3, i % 3
    axes[row, col_idx].hist(combined_df[columns[i]])
    axes[row, col_idx].set_title(f'{col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create correlation heatmap
#plt.figure(figsize=(10, 8))
#correlation_matrix = combined_df.corr()
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
#            square=True, linewidths=0.1)
#plt.title('Correlation Heatmap')
#plt.tight_layout()
#plt.show()

X_train_scaled, X_test_scaled, y_train, y_test = process_data(X, y)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = [1, 3, 5, 10, 15, 20, 30, 45, 65]
optimal_k, cv_scores, training_results = train_knn_model(X_train_scaled, y_train, k_values, cv_folds=kfold)

final_model = training_results['final_model']
y_pred = final_model.predict(X_test_scaled)
    
# Analyze performance
metrics = analyze_model_performance(y_test, y_pred, "KNN", True)