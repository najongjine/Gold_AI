import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os

def fetch_gold_price():
    # Define the ticker for Gold Futures (GC=F)
    ticker = "GC=F"
    
    # Calculate start and end dates (10 years ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365) # Approximation for 10 years
    
    print(f"Fetching gold price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Download data
        gold_data = yf.download(ticker, start=start_date, end=end_date)
        
        if gold_data.empty:
            print("No data found for the specified period.")
            return
            
        # Reset index to make Date a column
        gold_data.reset_index(inplace=True)
        
        # Save to CSV
        output_file = "gold_price_10yr.csv"
        gold_data.to_csv(output_file, index=False)
        
        print(f"Successfully saved gold price data to {output_file}")
        print(f"Data shape: {gold_data.shape}")
        print("\nLast 5 rows:")
        print(gold_data.tail())
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_gold_price()
