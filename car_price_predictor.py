import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Setup directories
    os.makedirs('plots', exist_ok=True)
    
    # 1. Download and Load Dataset
    url = "https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/master/car%20data.csv"
    DATA_PATH = "car_data.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from: {url}...")
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please place 'car_data.csv' in the project directory.")
            return
    else:
        print("Dataset found locally.")
        
    # Read the data
    df = pd.read_csv(DATA_PATH)
    
    print("\n[INFO] Dataset Dimensions:", df.shape)
    print("\n[INFO] First 5 rows of raw dataset:")
    print(df.head())
    
    # Standardize column names to prevent errors across different Kaggle versions
    # Mapping commonly found synonyms:
    rename_dict = {
        'Kms_Driven': 'Driven_kms',
        'Seller_Type': 'Selling_type'
    }
    df = df.rename(columns=rename_dict)
    
    print("\n[INFO] Cleaned Columns:", list(df.columns))
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # 2. Feature Engineering
    # Calculate Age of the car (using 2026 as current year based on local time)
    CURRENT_YEAR = 2026
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    
    # Drop columns that are not predictive (like Car_Name and purchase Year)
    # Car_Name has too many unique values for a small dataset, but we will print some statistics first
    print("\nUnique Car Models:", df['Car_Name'].nunique())
    model_df = df.drop(['Car_Name', 'Year'], axis=1)
    
    # One-Hot Encoding for Categorical columns
    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
    
    print("\n[INFO] Dataset after Feature Engineering (first 5 rows):")
    print(model_df.head())
    
    # 3. Exploratory Data Analysis (EDA)
    print("\nGenerating EDA Plots...")
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_df.corr(), annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Car Features', fontsize=12, fontweight='bold', pad=10)
    plt.savefig('plots/car_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scatter plot of Selling Price vs Present Price
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='Present_Price', y='Selling_Price', hue='Fuel_Type', palette='Set1', s=70)
    plt.title('Selling Price vs. Present Price (Showroom Price)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Present Price (Lakhs)', fontsize=10)
    plt.ylabel('Selling Price (Lakhs)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/car_price_vs_present.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Boxplot of Selling Price by Transmission type
    plt.figure(figsize=(6, 4.5))
    sns.boxplot(data=df, x='Transmission', y='Selling_Price', hue='Transmission', palette='Pastel2', legend=False)
    plt.title('Selling Price Distribution by Transmission Type', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Transmission', fontsize=10)
    plt.ylabel('Selling Price (Lakhs)', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('plots/car_price_by_transmission.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Data Splitting & Feature Scaling
    X = model_df.drop('Selling_Price', axis=1)
    y = model_df['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale continuous numerical features
    scaler = StandardScaler()
    scale_cols = ['Present_Price', 'Driven_kms', 'Car_Age']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # 5. Model Training & Comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'ModelObj': model
        }
        print(f"| {name:<23} | R2 Score: {r2:.4f} | RMSE: {rmse:.4f} |")
        
    summary_df = pd.DataFrame(results).T.drop('ModelObj', axis=1)
    print("\nSummary Table:")
    print(summary_df.to_string())
    
    # Identify the best model based on R2 Score
    best_model_name = summary_df['R2'].idxmax()
    best_model = results[best_model_name]['ModelObj']
    print(f"\n[INFO] Best Performing Model: {best_model_name} (R2: {results[best_model_name]['R2']:.4f})")
    
    # 6. Detailed Evaluation of the Best Model
    y_pred_best = best_model.predict(X_test_scaled)
    
    # Plot 4: Actual vs Predicted Prices Scatter Plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred_best, color='#1A365D', s=60, alpha=0.8)
    # Plot ideal line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs. Predicted Prices ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Actual Price (Lakhs)', fontsize=10)
    plt.ylabel('Predicted Price (Lakhs)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/car_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Feature Importance for Tree-based models (or Coefficients for Linear Models)
    plt.figure(figsize=(8, 5))
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis', hue=X.columns[indices], legend=False)
        plt.title(f'Feature Importance in Price Prediction ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    else:
        coefs = best_model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        sns.barplot(x=coefs[indices], y=X.columns[indices], palette='coolwarm', hue=X.columns[indices], legend=False)
        plt.title(f'Model Coefficients in Price Prediction ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
        
    plt.xlabel('Relative Importance / Coefficient Value', fontsize=10)
    plt.ylabel('Feature', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('plots/car_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[INFO] Analysis complete. Plots saved successfully in plots/ directory.")

if __name__ == "__main__":
    main()
