import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

def perform_eda():
    file_path = "gold_price_10yr.csv"
    if not pd.io.common.file_exists(file_path):
        print(f"Error: {file_path} not found. Please run fetch_gold_price.py first.")
        return

    # Load data - skip the second row which contains ticker names
    df = pd.read_csv(file_path, header=[0], skiprows=[1])
    
    # Preprocessing
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Check available columns to avoid KeyError
    available_cols = df.columns.tolist()
    print(f"Available columns: {available_cols}")
    
    # Use 'Close' if 'Adj Close' is missing (common with some yfinance exports)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    print(f"Using {price_col} for analysis.")
    
    # Ensure numeric columns are actually numeric
    numeric_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values that might have been introduced by coerce
    df = df.dropna(subset=[price_col])
    
    df = df.sort_values('Date')
    
    print("--- Descriptive Statistics ---")
    print(df.describe())
    
    # 1. Price Trend with Moving Averages
    plt.figure(figsize=(15, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df['Date'], df['Close'].rolling(window=50).mean(), label='50-day MA', color='orange')
    plt.plot(df['Date'], df['Close'].rolling(window=200).mean(), label='200-day MA', color='red')
    
    plt.title('Gold Price Trend (10 Years)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('gold_price_trend.png')
    plt.close()
    
    # 2. Daily Return Distribution
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Daily_Return'].dropna(), kde=True, color='gold')
    plt.title('Distribution of Daily Gold Price Returns (%)', fontsize=14)
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.savefig('gold_returns_dist.png')
    plt.close()
    
    # 3. Monthly Returns Heatmap
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    monthly_data = df.groupby(['Year', 'Month'])['Close'].last().unstack()
    monthly_returns = monthly_data.pct_change(axis=1) * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_returns, annot=True, fmt=".1f", cmap='RdYlGn', center=0)
    plt.title('Monthly Returns Heatmap (%)', fontsize=14)
    plt.savefig('gold_monthly_heatmap.png')
    plt.close()
    
    print("\nEDA completed. Plots saved as:")
    print("- gold_price_trend.png")
    print("- gold_returns_dist.png")
    print("- gold_monthly_heatmap.png")

if __name__ == "__main__":
    perform_eda()
