import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    filename = "samsung_stock_10yr.csv"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    # Load data
    # The CSV has multi-headers. Price, Ticker, Date.
    # We skip rows 1 and 2 to get clean columns, or handle multi-index.
    # Based on previous view_file:
    # 1: Price,Close,High,Low,Open,Volume
    # 2: Ticker,005930.KS,005930.KS,005930.KS,005930.KS,005930.KS
    # 3: Date,,,,,
    # 4: 2016-03-21,19820.7...
    
    # Let's read with multi-index header
    try:
        df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading with multi-index: {e}. Trying simple read.")
        df = pd.read_csv(filename, skiprows=2, index_col=0, parse_dates=True)
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean data
    df = df.sort_index()
    # Handle NaN values (especially for the last day)
    df = df.ffill() 

    print("Data Statistics:")
    print(df.describe())

    # Set style
    sns.set_theme(style="darkgrid")
    
    # 1. Price Trend with Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='royalblue', alpha=0.8)
    plt.plot(df.index, df['Close'].rolling(window=50).mean(), label='50-day SMA', color='orange', linestyle='--')
    plt.plot(df.index, df['Close'].rolling(window=200).mean(), label='200-day SMA', color='red', linestyle='--')
    plt.title('Samsung Electronics (005930.KS) Price Trend & SMA')
    plt.xlabel('Date')
    plt.ylabel('Price (KRW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('price_trend.png')
    print("Saved price_trend.png")

    # 2. Trading Volume
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df['Volume'], color='mediumseagreen', alpha=0.6)
    plt.title('Samsung Electronics (005930.KS) Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.tight_layout()
    plt.savefig('volume_trend.png')
    print("Saved volume_trend.png")

    # 3. Daily Returns Distribution
    df['Daily_Return'] = df['Close'].pct_change()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Daily_Return'].dropna(), bins=100, kde=True, color='purple')
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('daily_returns.png')
    print("Saved daily_returns.png")

    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation_heatmap.png")

if __name__ == "__main__":
    perform_eda()
