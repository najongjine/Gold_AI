import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_samsung_data():
    # Samsung Electronics Ticker
    ticker = "005930.KS"
    
    # Calculate start date (10 years ago from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)
    
    print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Fetch data
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if df.empty:
        print("No data found.")
        return
    
    # Save to CSV
    filename = "samsung_stock_10yr.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    
    # Display summary
    print("\nData Summary:")
    print(df.tail())
    print(f"\nTotal rows: {len(df)}")

if __name__ == "__main__":
    fetch_samsung_data()
