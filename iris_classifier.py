import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

def main():
    # Setup directories
    os.makedirs('plots', exist_ok=True)
    
    # 1. Download and Load the Dataset (Seaborn URL mapped to Kaggle format)
    SEABORN_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    DATA_PATH = "Iris.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from: {SEABORN_URL}...")
        try:
            # Download the seaborn raw CSV
            temp_path = "iris_raw.csv"
            urllib.request.urlretrieve(SEABORN_URL, temp_path)
            
            # Read, map to Kaggle format, and save
            raw_df = pd.read_csv(temp_path)
            raw_df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
            raw_df.insert(0, 'Id', range(1, len(raw_df) + 1))
            raw_df['Species'] = 'Iris-' + raw_df['Species']
            
            # Save the clean Kaggle-style CSV
            raw_df.to_csv(DATA_PATH, index=False)
            os.remove(temp_path)
            print("Dataset downloaded and formatted successfully to Iris.csv.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Fallback: Please place Iris.csv in the same directory.")
            return
    else:
        print("Dataset found locally.")
        
    # Read the data
    df = pd.read_csv(DATA_PATH)
    
    print("\n[INFO] Dataset Dimensions:", df.shape)
    print("\n[INFO] First 5 rows of the dataset:")
    print(df.head())
    
    # Check for Id column and drop it (not a predictive feature)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    print("\n[INFO] Data Statistics:")
    print(df.describe())
    
    print("\n[INFO] Class Distribution:")
    print(df['Species'].value_counts())
    
    # 2. Exploratory Data Analysis (EDA)
    print("\n[INFO] Generating and saving plots...")
    
    # Pairplot for visualizing features pairwise relationships
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, hue='Species', palette='Set2')
    plt.savefig('plots/iris_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(7, 5))
    # Drop target column to calculate feature correlation
    features_df = df.drop('Species', axis=1)
    sns.heatmap(features_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Iris Features')
    plt.savefig('plots/iris_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot for feature distribution by species
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(features_df.columns):
        ax = axes[i // 2, i % 2]
        sns.boxplot(x='Species', y=col, hue='Species', data=df, ax=ax, palette='Pastel1', legend=False)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('plots/iris_features_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[INFO] Plots saved in the 'plots/' directory.")
    
    # 3. Data Splitting & Feature Scaling
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # 80/20 train/test split, stratified to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling (Normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Training & Comparison
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Support Vector Machine': SVC(kernel='linear', C=1.0, random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("\n--- Model Training & Performance Evaluation ---")
    for name, model in models.items():
        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ModelObject': model
        }
        print(f"| {name:<22} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f} |")
        
    # Convert results dictionary to DataFrame for easy comparison
    summary_df = pd.DataFrame(results).T.drop('ModelObject', axis=1)
    print("\nSummary Table:")
    print(summary_df.to_string())
    
    # Identify the best model based on Accuracy
    best_model_name = summary_df['Accuracy'].idxmax()
    best_model = results[best_model_name]['ModelObject']
    print(f"\n[INFO] Best Performing Model: {best_model_name} (Accuracy: {results[best_model_name]['Accuracy']:.4f})")
    
    # 5. In-depth Analysis of the Best Model
    y_pred_best = best_model.predict(X_test_scaled)
    
    print(f"\n--- Detailed Classification Report ({best_model_name}) ---")
    print(classification_report(y_test, y_pred_best))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, 
                yticklabels=best_model.classes_)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('plots/iris_best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion Matrix heatmap saved as 'plots/iris_best_model_confusion_matrix.png'.")
    
    # 6. Prediction Helper Function Demonstration
    print("\n--- Making Real-time Predictions ---")
    
    def predict_new_flower(sepal_len, sepal_wid, petal_len, petal_wid):
        # Format input data as DataFrame with proper column names to prevent warnings
        sample = pd.DataFrame(
            [[sepal_len, sepal_wid, petal_len, petal_wid]], 
            columns=X.columns
        )
        # Scale using the trained scaler
        sample_scaled = scaler.transform(sample)
        # Predict class
        pred = best_model.predict(sample_scaled)[0]
        # Predict probabilities
        probabilities = best_model.predict_proba(sample_scaled)[0] if hasattr(best_model, "predict_proba") else None
        
        return pred, probabilities

    # Test with custom sample values
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Expected: Iris-setosa
        [6.0, 3.0, 4.8, 1.8],  # Expected: Iris-virginica or Iris-versicolor
        [5.9, 3.0, 5.1, 1.8]   # Expected: Iris-virginica
    ]
    
    for i, sample in enumerate(test_samples):
        prediction, probs = predict_new_flower(*sample)
        print(f"Sample {i+1} {sample} -> Predicted Class: {prediction}")
        if probs is not None:
            prob_dict = {cls: f"{p:.2%}" for cls, p in zip(best_model.classes_, probs)}
            print(f"  Confidence: {prob_dict}")

if __name__ == "__main__":
    main()
