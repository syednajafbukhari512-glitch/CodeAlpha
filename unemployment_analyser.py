import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Setup directories
    os.makedirs('plots', exist_ok=True)
    
    # 1. Download and Load Dataset
    urls = [
        "https://raw.githubusercontent.com/proteendas/OIBSIP-DS-03/main/Unemployment_Rate_upto_11_2020.csv",
        "https://raw.githubusercontent.com/Tosa9/CodeAlpha_UnemploymentAnalysis/main/data/Unemployment_Rate_upto_11_2020.csv"
    ]
    
    DATA_PATH = "Unemployment_Rate_upto_11_2020.csv"
    downloaded = False
    
    if not os.path.exists(DATA_PATH):
        for url in urls:
            print(f"Trying to download dataset from: {url}...")
            try:
                urllib.request.urlretrieve(url, DATA_PATH)
                print("Dataset downloaded successfully.")
                downloaded = True
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
        if not downloaded:
            print("[ERROR] Could not download dataset from any URL.")
            print("Please place 'Unemployment_Rate_upto_11_2020.csv' in the project directory.")
            return
    else:
        print("Dataset found locally.")
        
    # Read the data
    df = pd.read_csv(DATA_PATH)
    
    # 2. Data Cleaning
    print("\n--- Data Cleaning & Preprocessing ---")
    print("[INFO] Raw Columns:", list(df.columns))
    
    # Strip whitespaces from column names
    df.columns = df.columns.str.strip()
    print("[INFO] Cleaned Columns:", list(df.columns))
    
    # Show missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Drop rows where critical info is missing
    df = df.dropna(subset=['Region', 'Date', 'Estimated Unemployment Rate (%)'])
    
    # Convert 'Date' to datetime format
    # The date is formatted as ' 31-05-2020' or '31-05-2020', we strip first
    df['Date'] = df['Date'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Add month and year features
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Month_Num'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    print("\n[INFO] Cleaned Dataset Sample:")
    print(df.head())
    
    print("\n[INFO] Basic Statistics of Numeric Features:")
    print(df.describe())
    
    # 3. Exploratory Data Analysis & Visualizations
    print("\n--- Generating Visualizations ---")
    
    # Plot 1: Overall Monthly Unemployment Trend in India (2020)
    monthly_trend = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_trend, x='Date', y='Estimated Unemployment Rate (%)', marker='o', linewidth=2.5, color='#1A365D')
    plt.title('Average Monthly Unemployment Rate in India (2020)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Unemployment Rate (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    # Highlight the lockdown period (April - June 2020)
    plt.axvspan('2020-04-01', '2020-06-30', color='red', alpha=0.1, label='Severe Lockdown Period')
    plt.legend()
    plt.savefig('plots/unemployment_monthly_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Region-wise Average Unemployment Rate
    region_data = df.groupby('Region.1')['Estimated Unemployment Rate (%)'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=region_data.sort_values(by='Estimated Unemployment Rate (%)', ascending=False), 
                x='Region.1', y='Estimated Unemployment Rate (%)', hue='Region.1', palette='Blues_r', legend=False)
    plt.title('Average Unemployment Rate by Geographical Region', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Region', fontsize=11)
    plt.ylabel('Average Unemployment Rate (%)', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/unemployment_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: State-wise Average Unemployment Rate (Top 15 States with Highest Unemployment)
    state_data = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
    top_states = state_data.sort_values(by='Estimated Unemployment Rate (%)', ascending=False).head(15)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_states, x='Estimated Unemployment Rate (%)', y='Region', hue='Region', palette='Reds_r', legend=False)
    plt.title('Top 15 States/UTs with Highest Average Unemployment Rate (2020)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Unemployment Rate (%)', fontsize=11)
    plt.ylabel('State / Union Territory', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('plots/unemployment_top_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Correlation Matrix
    plt.figure(figsize=(6, 4.5))
    corr_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Labour Metrics', fontsize=12, fontweight='bold', pad=10)
    plt.savefig('plots/unemployment_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Covid-19 Impact Analysis (Pre- vs. Post-Lockdown)
    print("\n--- COVID-19 Impact Analysis ---")
    
    def get_phase(date):
        if date < pd.Timestamp('2020-04-01'):
            return 'Pre-Lockdown (Jan-Mar)'
        elif date <= pd.Timestamp('2020-06-30'):
            return 'Lockdown Peak (Apr-Jun)'
        else:
            return 'Post-Lockdown (Jul-Nov)'
            
    df['Lockdown_Phase'] = df['Date'].apply(get_phase)
    phase_analysis = df.groupby('Lockdown_Phase')[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].mean().reset_index()
    print(phase_analysis.to_string(index=False))
    
    # Plot 5: Boxplot showing distributions of unemployment rates during phases
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='Lockdown_Phase', y='Estimated Unemployment Rate (%)', hue='Lockdown_Phase', palette='Set2', legend=False)
    plt.title('Unemployment Rate Distribution across COVID-19 Phases', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Lockdown Phase', fontsize=11)
    plt.ylabel('Unemployment Rate (%)', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('plots/unemployment_phase_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[INFO] All plots successfully generated and saved in 'plots/' directory.")

if __name__ == "__main__":
    main()
