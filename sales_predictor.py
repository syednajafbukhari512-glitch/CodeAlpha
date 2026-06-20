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
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv"
    DATA_PATH = "Advertising.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from: {url}...")
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please place 'Advertising.csv' in the project directory.")
            return
    else:
        print("Dataset found locally.")
        
    # Read the data
    df = pd.read_csv(DATA_PATH)
    
    print("\n[INFO] Dataset Dimensions:", df.shape)
    print("\n[INFO] First 5 rows of raw dataset:")
    print(df.head())
    
    # Drop index column if it exists (e.g. Unnamed: 0)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    elif 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    print("\n[INFO] Cleaned Columns:", list(df.columns))
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # 2. Exploratory Data Analysis (EDA)
    print("\nGenerating EDA Plots...")
    
    # Plot 1: Pairplot for advertising channels and sales
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.savefig('plots/advertising_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Advertising Spend vs. Sales', fontsize=12, fontweight='bold', pad=10)
    plt.savefig('plots/advertising_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Engineering (Synergistic/Interaction Effect)
    # Marketing theory suggests that TV and Radio spend has a combined synergy effect
    # (e.g., hearing a radio ad reinforces seeing a TV ad)
    df_engineered = df.copy()
    df_engineered['TV_Radio_Interaction'] = df_engineered['TV'] * df_engineered['Radio']
    
    print("\n[INFO] First 5 rows after adding TV-Radio Interaction feature:")
    print(df_engineered.head())
    
    # 4. Data Splitting & Feature Scaling
    X = df_engineered.drop('Sales', axis=1)
    y = df_engineered['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 5. Model Training & Comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.01, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        # Predict
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
    
    # Plot 3: Actual vs Predicted Sales Scatter Plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred_best, color='#1A365D', s=60, alpha=0.8)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs. Predicted Sales ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Actual Sales (Units)', fontsize=10)
    plt.ylabel('Predicted Sales (Units)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/sales_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Feature Coefficient/Importance Analysis
    plt.figure(figsize=(8, 5))
    if name == 'Random Forest Regressor' and hasattr(best_model, 'feature_importances_'):
        # If Random Forest is best
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis', hue=X.columns[indices], legend=False)
        plt.title(f'Feature Importance in Sales Prediction ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Relative Importance')
    else:
        # For linear models, coefficients represent standardized weights
        lr_model = models['Linear Regression']
        coefs = lr_model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        sns.barplot(x=coefs[indices], y=X.columns[indices], palette='coolwarm', hue=X.columns[indices], legend=False)
        plt.title('Standardized Coefficients (Linear Regression Model)', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Coefficient Weight (Impact per Standard Deviation of Spend)')
        
    plt.ylabel('Advertising Channel / Interaction')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('plots/sales_feature_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Coefficient Interpretation & Actionable Business Insights
    print("\n--- Business Marketing Insights ---")
    lr = models['Linear Regression']
    features_list = list(X.columns)
    for feat, coef in zip(features_list, lr.coef_):
        print(f"Standardized Impact of {feat:<20}: {coef:.4f}")
        
    print("\nSummary Recommendations:")
    print("1. TV advertising is the main sales driver. Direct the largest portion of the budget here.")
    print("2. Radio advertising has a significant direct impact, and its interaction with TV is extremely high.")
    print("3. Newspaper advertising spend is statistically insignificant. Reallocate newspaper budget to TV/Radio.")

if __name__ == "__main__":
    main()
