from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -----------------------------
# Basic settings
# -----------------------------
# GLD is a popular ETF that closely tracks the gold price.
TICKER = "GC=F"

# Download the last 10 years of daily data.
HISTORY_PERIOD = "10y"
INTERVAL = "1d"

# Simulate 60 future trading days, 1000 times.
FUTURE_DAYS = 60
NUM_SIMULATIONS = 1000

# Use recent market behavior instead of full 10-year statistics.
SIMULATION_WINDOW = 252
COMPARISON_WINDOWS = [60, 120, 252]

# Fix the random seed so the result is reproducible.
RANDOM_SEED = 42


def configure_yfinance_cache():
    """
    Configure a local cache folder for yfinance.
    This helps avoid SQLite cache-path issues on some environments.
    """
    cache_dir = Path(__file__).resolve().parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # yfinance exposes a helper for timezone cache location.
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


def download_gold_data():
    """
    Download gold-related price data from yfinance.
    Return a clean DataFrame with Date and Close columns.
    """
    print(f"[1] Downloading {TICKER} price data...")

    df = pd.DataFrame()

    # First try yf.download(), which is concise and common.
    try:
        # progress=False keeps the console output clean.
        df = yf.download(
            TICKER,
            period=HISTORY_PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    # If the first attempt fails, fall back to Ticker.history().
    if df.empty:
        try:
            ticker = yf.Ticker(TICKER)
            df = ticker.history(period=HISTORY_PERIOD, interval=INTERVAL, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        raise ValueError("Failed to download data. Check your internet connection or ticker symbol.")

    # Some yfinance versions return MultiIndex columns.
    # This block safely extracts the Close column in both cases.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", TICKER) in df.columns:
            close_series = df[("Close", TICKER)]
        elif ("Adj Close", TICKER) in df.columns:
            close_series = df[("Adj Close", TICKER)]
        else:
            raise KeyError("Could not find a Close column.")
    else:
        # The request asks for closing-price-based returns,
        # so Close is used first.
        if "Close" in df.columns:
            close_series = df["Close"]
        elif "Adj Close" in df.columns:
            close_series = df["Adj Close"]
        else:
            raise KeyError("Could not find a Close column.")

    # Convert the date index into a normal column first.
    # This avoids ambiguity when the index itself is also named "Date".
    price_df = pd.DataFrame({"Close": pd.to_numeric(close_series, errors="coerce")})
    price_df = price_df.reset_index()

    # Rename the first column to Date in a version-safe way.
    # Depending on the pandas/yfinance version, it may already be named Date.
    first_col = price_df.columns[0]
    if first_col != "Date":
        price_df = price_df.rename(columns={first_col: "Date"})

    price_df = price_df.dropna(subset=["Date", "Close"])
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.sort_values("Date").reset_index(drop=True)

    return price_df


def calculate_daily_returns(price_df):
    """
    Calculate daily log returns from closing prices.
    Formula:
        ln(today's close / yesterday's close)
    """
    price_df = price_df.copy()
    price_df["Log_Return"] = np.log(price_df["Close"] / price_df["Close"].shift(1))
    price_df = price_df.dropna().reset_index(drop=True)
    return price_df


def calculate_recent_statistics(price_with_returns, window):
    """
    Calculate mean log return and volatility from a recent window.
    If the requested window is longer than the available data,
    use all available rows.
    """
    recent_data = price_with_returns.tail(window).copy()

    if recent_data.empty:
        raise ValueError("Not enough return data to calculate recent statistics.")

    return {
        "window": len(recent_data),
        "mean_log_return": recent_data["Log_Return"].mean(),
        "volatility": recent_data["Log_Return"].std(),
    }


def calculate_window_comparison(price_with_returns, windows):
    """
    Build comparable statistics for multiple recent windows.
    """
    comparison = []

    for window in windows:
        stats = calculate_recent_statistics(price_with_returns, window)
        comparison.append(stats)

    return comparison


def run_monte_carlo(last_price, mean_log_return, volatility, days=60, simulations=1000):
    """
    Run a simple Monte Carlo simulation.

    Idea:
    - Measure historical daily mean log return and volatility
    - Assume future daily log returns behave similarly
    - Draw random daily log returns from a normal distribution
    - Update price with: next_price = current_price * exp(daily_log_return)
    """
    np.random.seed(RANDOM_SEED)

    # Shape:
    # rows    -> future trading days
    # columns -> simulation number
    simulated_prices = np.zeros((days, simulations))

    for sim in range(simulations):
        prices = [last_price]

        for _ in range(days):
            # Create one random daily log return.
            random_log_return = np.random.normal(loc=mean_log_return, scale=volatility)
            next_price = prices[-1] * np.exp(random_log_return)
            prices.append(next_price)

        # Skip the first value because it is the current price.
        simulated_prices[:, sim] = prices[1:]

    return simulated_prices


def summarize_simulation(simulated_prices):
    """
    Calculate the mean path and percentile paths.
    """
    mean_path = simulated_prices.mean(axis=1)
    p5_path = np.percentile(simulated_prices, 5, axis=1)
    p50_path = np.percentile(simulated_prices, 50, axis=1)
    p95_path = np.percentile(simulated_prices, 95, axis=1)

    return mean_path, p5_path, p50_path, p95_path


def plot_results(price_df, simulated_prices, mean_path, p5_path, p50_path, p95_path):
    """
    Visualize historical prices and simulation results.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Historical closing price chart
    axes[0].plot(price_df["Date"], price_df["Close"], color="goldenrod", linewidth=2)
    axes[0].set_title(f"{TICKER} Closing Price - Last 10 Years", fontsize=14)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (USD)")

    # 2. Future 60-day simulation chart
    future_x = np.arange(1, FUTURE_DAYS + 1)

    # Plot all simulation paths with very low opacity.
    axes[1].plot(future_x, simulated_prices, color="skyblue", alpha=0.03)

    # Highlight the summary paths.
    axes[1].plot(future_x, mean_path, color="red", linewidth=2, label="Mean Path")
    axes[1].plot(future_x, p5_path, color="green", linestyle="--", linewidth=2, label="5th Percentile")
    axes[1].plot(future_x, p50_path, color="blue", linestyle="-.", linewidth=2, label="50th Percentile")
    axes[1].plot(future_x, p95_path, color="purple", linestyle="--", linewidth=2, label="95th Percentile")

    axes[1].set_title(
        f"{TICKER} Monte Carlo Simulation for Next {FUTURE_DAYS} Trading Days ({NUM_SIMULATIONS} runs)",
        fontsize=14,
    )
    axes[1].set_xlabel("Future Trading Day")
    axes[1].set_ylabel("Simulated Price (USD)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    # Prepare yfinance cache before any download happens.
    configure_yfinance_cache()

    # 1. Download data
    price_df = download_gold_data()

    # 2. Calculate daily returns
    price_with_returns = calculate_daily_returns(price_df)

    # 3. Calculate recent-window statistics
    recent_stats = calculate_recent_statistics(price_with_returns, SIMULATION_WINDOW)
    comparison_stats = calculate_window_comparison(price_with_returns, COMPARISON_WINDOWS)
    mean_log_return = recent_stats["mean_log_return"]
    volatility = recent_stats["volatility"]
    last_price = price_with_returns["Close"].iloc[-1]

    # 4. Run simulation
    print(f"[2] Running Monte Carlo simulation... ({NUM_SIMULATIONS} runs)")
    simulated_prices = run_monte_carlo(
        last_price=last_price,
        mean_log_return=mean_log_return,
        volatility=volatility,
        days=FUTURE_DAYS,
        simulations=NUM_SIMULATIONS,
    )

    # 5. 평균 경로와 분위값 계산
    mean_path, p5_path, p50_path, p95_path = summarize_simulation(simulated_prices)

    # 6. Print summary to the console
    print("\n===== Gold Price Monte Carlo Simulation Result =====")
    print(f"Ticker: {TICKER}")
    print(f"Latest close price: ${last_price:.2f}")
    print(f"Simulation statistics window: recent {recent_stats['window']} trading days")
    print(f"Average daily log return: {mean_log_return:.6f}")
    print(f"Daily volatility: {volatility:.6f} ({volatility * 100:.4f}%)")
    print("\nRecent window comparison:")
    for stats in comparison_stats:
        print(
            f"- {stats['window']:>3} days | mean log return: {stats['mean_log_return']:.6f} | "
            f"volatility: {stats['volatility']:.6f} ({stats['volatility'] * 100:.4f}%)"
        )
    print(f"Expected mean price after {FUTURE_DAYS} days: ${mean_path[-1]:.2f}")
    print(f"5th percentile after {FUTURE_DAYS} days: ${p5_path[-1]:.2f}")
    print(f"50th percentile after {FUTURE_DAYS} days: ${p50_path[-1]:.2f}")
    print(f"95th percentile after {FUTURE_DAYS} days: ${p95_path[-1]:.2f}")
    print(f"Expected price range after {FUTURE_DAYS} days (5% to 95%): ${p5_path[-1]:.2f} ~ ${p95_path[-1]:.2f}")

    # 7. Show charts
    plot_results(price_df, simulated_prices, mean_path, p5_path, p50_path, p95_path)


if __name__ == "__main__":
    main()
