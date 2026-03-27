from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Basic settings
# -----------------------------
# GC=F is a popular Yahoo Finance symbol for gold futures.
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
SIMULATION_METHODS = ["normal", "bootstrap", "regime"]

# Fix the random seed so the result is reproducible.
RANDOM_SEED = 42

# Trend signal windows
SHORT_MA_WINDOW = 5
MID_MA_WINDOW = 20
LONG_MA_WINDOW = 60
RSI_WINDOW = 14
SHORT_RETURN_WINDOW = 20
LONG_RETURN_WINDOW = 60

# Small coefficients so the model keeps randomness
# while leaning slightly toward the current trend.
MA_DRIFT_ADJUSTMENT = 0.0003
RSI_DRIFT_ADJUSTMENT = 0.0002
MACD_DRIFT_SCALE = 0.05
RETURN_20_DRIFT_SCALE = 0.03
RETURN_60_DRIFT_SCALE = 0.02

# Exogenous variable settings
EXOGENOUS_WINDOW = 20
FRED_SERIES = {
    "DGS10": "Treasury_10Y",
    "DFII10": "Real_Rate_10Y",
    "CPIAUCSL": "CPI",
}
YF_EXOGENOUS_TICKERS = {
    "DX-Y.NYB": "Dollar_Index",
    "^VIX": "VIX",
    "SPY": "SPY",
    "CL=F": "Crude_Oil",
    "GLD": "GLD",
    "TLT": "TLT",
}

# Small drift adjustments from macro and cross-asset signals.
DOLLAR_DRIFT_SCALE = 0.10
REAL_RATE_DRIFT_SCALE = 0.12
RATE_DRIFT_SCALE = 0.05
VIX_DRIFT_SCALE = 0.03
SPY_DRIFT_SCALE = 0.04
OIL_DRIFT_SCALE = 0.02
GLD_DRIFT_SCALE = 0.05
TLT_DRIFT_SCALE = 0.03
INFLATION_DRIFT_SCALE = 0.02

# Regime model settings
REGIME_VOL_WINDOW = 20
HIGH_VOL_QUANTILE = 0.65


def configure_yfinance_cache():
    """
    Configure a local cache folder for yfinance.
    This helps avoid SQLite cache-path issues on some environments.
    """
    cache_dir = Path(__file__).resolve().parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


def download_gold_data():
    """
    Download gold-related price data from yfinance.
    Return a clean DataFrame with Date and Close columns.
    """
    print(f"[1] Downloading {TICKER} price data...")

    df = pd.DataFrame()

    try:
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

    if df.empty:
        try:
            ticker = yf.Ticker(TICKER)
            df = ticker.history(period=HISTORY_PERIOD, interval=INTERVAL, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        raise ValueError("Failed to download data. Check your internet connection or ticker symbol.")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", TICKER) in df.columns:
            close_series = df[("Close", TICKER)]
        elif ("Adj Close", TICKER) in df.columns:
            close_series = df[("Adj Close", TICKER)]
        else:
            raise KeyError("Could not find a Close column.")
    else:
        if "Close" in df.columns:
            close_series = df["Close"]
        elif "Adj Close" in df.columns:
            close_series = df["Adj Close"]
        else:
            raise KeyError("Could not find a Close column.")

    price_df = pd.DataFrame({"Close": pd.to_numeric(close_series, errors="coerce")})
    price_df = price_df.reset_index()

    first_col = price_df.columns[0]
    if first_col != "Date":
        price_df = price_df.rename(columns={first_col: "Date"})

    price_df = price_df.dropna(subset=["Date", "Close"])
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.sort_values("Date").reset_index(drop=True)

    return price_df


def download_yfinance_close_series(tickers, period=HISTORY_PERIOD, interval=INTERVAL):
    """
    Download multiple close-price series from yfinance and align them by date.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            list(tickers),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close_df = df["Close"].copy()
        elif "Adj Close" in df.columns.get_level_values(0):
            close_df = df["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        close_col = "Close" if "Close" in df.columns else "Adj Close"
        if close_col not in df.columns:
            return pd.DataFrame()
        close_df = df[[close_col]].copy()
        close_df.columns = [list(tickers)[0]]

    close_df = close_df.apply(pd.to_numeric, errors="coerce")
    close_df = close_df.reset_index()
    first_col = close_df.columns[0]
    if first_col != "Date":
        close_df = close_df.rename(columns={first_col: "Date"})

    renamed_columns = {"Date": "Date"}
    for ticker, alias in tickers.items():
        if ticker in close_df.columns:
            renamed_columns[ticker] = alias

    close_df = close_df.rename(columns=renamed_columns)
    close_df["Date"] = pd.to_datetime(close_df["Date"])
    close_df = close_df.sort_values("Date").reset_index(drop=True)
    return close_df


def download_fred_series(series_map):
    """
    Download FRED series via the public CSV endpoint.

    This avoids requiring an API key while still letting the model
    use macro variables as drift-adjustment inputs.
    """
    fred_frames = []

    for series_id, alias in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            series_df = pd.read_csv(url)
        except Exception:
            continue

        if series_df.empty or "DATE" not in series_df.columns or series_id not in series_df.columns:
            continue

        series_df = series_df.rename(columns={"DATE": "Date", series_id: alias})
        series_df["Date"] = pd.to_datetime(series_df["Date"], errors="coerce")
        series_df[alias] = pd.to_numeric(series_df[alias], errors="coerce")
        series_df = series_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        fred_frames.append(series_df[["Date", alias]])

    if not fred_frames:
        return pd.DataFrame()

    fred_df = fred_frames[0]
    for frame in fred_frames[1:]:
        fred_df = fred_df.merge(frame, on="Date", how="outer")

    fred_df = fred_df.sort_values("Date").reset_index(drop=True)
    return fred_df


def build_exogenous_feature_set(price_df):
    """
    Build a merged daily feature table from market and macro series.
    """
    market_df = download_yfinance_close_series(YF_EXOGENOUS_TICKERS)
    fred_df = download_fred_series(FRED_SERIES)

    feature_df = price_df[["Date", "Close"]].copy()

    if not market_df.empty:
        feature_df = feature_df.merge(market_df, on="Date", how="left")
    if not fred_df.empty:
        feature_df = feature_df.merge(fred_df, on="Date", how="left")

    feature_df = feature_df.sort_values("Date").reset_index(drop=True)
    feature_df = feature_df.ffill()

    return feature_df


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


def calculate_trend_indicators(price_df):
    """
    Calculate simple trend and momentum indicators from closing prices.
    These signals are later used to slightly adjust the Monte Carlo drift.
    """
    df = price_df.copy()

    df["MA5"] = df["Close"].rolling(SHORT_MA_WINDOW).mean()
    df["MA20"] = df["Close"].rolling(MID_MA_WINDOW).mean()
    df["MA60"] = df["Close"].rolling(LONG_MA_WINDOW).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Momentum"] = (ema12 - ema26) / df["Close"]

    df["Return_20D"] = df["Close"] / df["Close"].shift(SHORT_RETURN_WINDOW) - 1
    df["Return_60D"] = df["Close"] / df["Close"].shift(LONG_RETURN_WINDOW) - 1

    return df


def calculate_trend_adjusted_drift(price_df, base_mean_log_return):
    """
    Start from the recent average log return and add small adjustments
    from moving-average trend, RSI, MACD-like momentum, and recent returns.
    """
    indicators_df = calculate_trend_indicators(price_df)
    latest = indicators_df.iloc[-1]

    drift_adjustment = 0.0

    if pd.notna(latest["MA20"]) and pd.notna(latest["MA60"]):
        if latest["MA20"] > latest["MA60"]:
            drift_adjustment += MA_DRIFT_ADJUSTMENT
        elif latest["MA20"] < latest["MA60"]:
            drift_adjustment -= MA_DRIFT_ADJUSTMENT

    if pd.notna(latest["MA5"]) and pd.notna(latest["MA20"]):
        if latest["MA5"] > latest["MA20"]:
            drift_adjustment += MA_DRIFT_ADJUSTMENT / 2
        elif latest["MA5"] < latest["MA20"]:
            drift_adjustment -= MA_DRIFT_ADJUSTMENT / 2

    if latest["RSI"] >= 70:
        drift_adjustment -= RSI_DRIFT_ADJUSTMENT
    elif latest["RSI"] <= 30:
        drift_adjustment += RSI_DRIFT_ADJUSTMENT

    if pd.notna(latest["MACD_Momentum"]):
        drift_adjustment += latest["MACD_Momentum"] * MACD_DRIFT_SCALE

    if pd.notna(latest["Return_20D"]):
        drift_adjustment += latest["Return_20D"] * RETURN_20_DRIFT_SCALE / SHORT_RETURN_WINDOW

    if pd.notna(latest["Return_60D"]):
        drift_adjustment += latest["Return_60D"] * RETURN_60_DRIFT_SCALE / LONG_RETURN_WINDOW

    adjusted_drift = base_mean_log_return + drift_adjustment

    trend_snapshot = {
        "MA5": latest["MA5"],
        "MA20": latest["MA20"],
        "MA60": latest["MA60"],
        "RSI": latest["RSI"],
        "MACD_Momentum": latest["MACD_Momentum"],
        "Return_20D": latest["Return_20D"],
        "Return_60D": latest["Return_60D"],
        "base_mean_log_return": base_mean_log_return,
        "drift_adjustment": drift_adjustment,
        "adjusted_drift": adjusted_drift,
    }

    return adjusted_drift, trend_snapshot


def safe_window_return(series, window=EXOGENOUS_WINDOW):
    """
    Return a simple percentage change over the given window if possible.
    """
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) <= window:
        return np.nan
    start_value = valid.iloc[-window - 1]
    end_value = valid.iloc[-1]
    if start_value == 0 or pd.isna(start_value) or pd.isna(end_value):
        return np.nan
    return end_value / start_value - 1


def safe_window_change(series, window=EXOGENOUS_WINDOW):
    """
    Return a simple arithmetic change over the given window if possible.
    """
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) <= window:
        return np.nan
    return valid.iloc[-1] - valid.iloc[-window - 1]


def calculate_exogenous_drift_adjustment(feature_df):
    """
    Use macro and cross-asset variables to gently adjust gold drift.

    Heuristics:
    - Dollar weakness tends to support gold.
    - Higher nominal and real yields tend to pressure gold.
    - Higher risk aversion can support gold.
    - Oil, GLD, TLT, and inflation changes are used as softer side signals.
    """
    drift_adjustment = 0.0
    snapshot = {}

    signal_map = {
        "Dollar_Index_20D": safe_window_return(feature_df.get("Dollar_Index", pd.Series(dtype=float))),
        "VIX_20D": safe_window_return(feature_df.get("VIX", pd.Series(dtype=float))),
        "SPY_20D": safe_window_return(feature_df.get("SPY", pd.Series(dtype=float))),
        "Crude_Oil_20D": safe_window_return(feature_df.get("Crude_Oil", pd.Series(dtype=float))),
        "GLD_20D": safe_window_return(feature_df.get("GLD", pd.Series(dtype=float))),
        "TLT_20D": safe_window_return(feature_df.get("TLT", pd.Series(dtype=float))),
        "Treasury_10Y_20D_Change": safe_window_change(feature_df.get("Treasury_10Y", pd.Series(dtype=float))),
        "Real_Rate_10Y_20D_Change": safe_window_change(feature_df.get("Real_Rate_10Y", pd.Series(dtype=float))),
        "CPI_20D_Change": safe_window_change(feature_df.get("CPI", pd.Series(dtype=float))),
    }

    if pd.notna(signal_map["Dollar_Index_20D"]):
        drift_adjustment -= signal_map["Dollar_Index_20D"] * DOLLAR_DRIFT_SCALE
    if pd.notna(signal_map["Real_Rate_10Y_20D_Change"]):
        drift_adjustment -= signal_map["Real_Rate_10Y_20D_Change"] * REAL_RATE_DRIFT_SCALE
    if pd.notna(signal_map["Treasury_10Y_20D_Change"]):
        drift_adjustment -= signal_map["Treasury_10Y_20D_Change"] * RATE_DRIFT_SCALE
    if pd.notna(signal_map["VIX_20D"]):
        drift_adjustment += signal_map["VIX_20D"] * VIX_DRIFT_SCALE
    if pd.notna(signal_map["SPY_20D"]):
        drift_adjustment -= signal_map["SPY_20D"] * SPY_DRIFT_SCALE
    if pd.notna(signal_map["Crude_Oil_20D"]):
        drift_adjustment += signal_map["Crude_Oil_20D"] * OIL_DRIFT_SCALE
    if pd.notna(signal_map["GLD_20D"]):
        drift_adjustment += signal_map["GLD_20D"] * GLD_DRIFT_SCALE
    if pd.notna(signal_map["TLT_20D"]):
        drift_adjustment += signal_map["TLT_20D"] * TLT_DRIFT_SCALE
    if pd.notna(signal_map["CPI_20D_Change"]):
        drift_adjustment += signal_map["CPI_20D_Change"] * INFLATION_DRIFT_SCALE

    latest_row = feature_df.iloc[-1] if not feature_df.empty else pd.Series(dtype=float)
    snapshot.update(signal_map)
    snapshot["latest_dollar_index"] = latest_row.get("Dollar_Index", np.nan)
    snapshot["latest_vix"] = latest_row.get("VIX", np.nan)
    snapshot["latest_spy"] = latest_row.get("SPY", np.nan)
    snapshot["latest_crude_oil"] = latest_row.get("Crude_Oil", np.nan)
    snapshot["latest_treasury_10y"] = latest_row.get("Treasury_10Y", np.nan)
    snapshot["latest_real_rate_10y"] = latest_row.get("Real_Rate_10Y", np.nan)
    snapshot["latest_cpi"] = latest_row.get("CPI", np.nan)
    snapshot["drift_adjustment"] = drift_adjustment

    return drift_adjustment, snapshot


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
        "historical_log_returns": recent_data["Log_Return"].to_numpy(),
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


def build_regime_model(price_with_returns, mean_log_return):
    """
    Build a simple two-state regime model from recent returns.

    States:
    - low_vol: calmer market regime
    - high_vol: stressed or fast-moving regime

    The regime is identified from rolling volatility and simulated
    later with a simple Markov transition matrix.
    """
    regime_df = price_with_returns[["Date", "Log_Return"]].copy()
    regime_df["Rolling_Vol"] = regime_df["Log_Return"].rolling(REGIME_VOL_WINDOW).std()
    regime_df = regime_df.dropna().reset_index(drop=True)

    if regime_df.empty:
        raise ValueError("Not enough data to build a regime model.")

    vol_threshold = regime_df["Rolling_Vol"].quantile(HIGH_VOL_QUANTILE)
    regime_df["Regime"] = np.where(regime_df["Rolling_Vol"] >= vol_threshold, "high_vol", "low_vol")

    if regime_df["Regime"].nunique() < 2:
        median_vol = regime_df["Rolling_Vol"].median()
        regime_df["Regime"] = np.where(regime_df["Rolling_Vol"] >= median_vol, "high_vol", "low_vol")

    regime_stats = {}
    overall_mean = regime_df["Log_Return"].mean()

    for regime_name in ["low_vol", "high_vol"]:
        regime_returns = regime_df.loc[regime_df["Regime"] == regime_name, "Log_Return"]

        if regime_returns.empty:
            regime_stats[regime_name] = {
                "mean": overall_mean,
                "volatility": regime_df["Log_Return"].std(),
                "count": 0,
            }
            continue

        regime_stats[regime_name] = {
            "mean": regime_returns.mean(),
            "volatility": regime_returns.std(),
            "count": len(regime_returns),
        }

    for regime_name in regime_stats:
        if pd.isna(regime_stats[regime_name]["volatility"]) or regime_stats[regime_name]["volatility"] == 0:
            regime_stats[regime_name]["volatility"] = regime_df["Log_Return"].std()

    transitions = pd.crosstab(
        regime_df["Regime"].shift(1),
        regime_df["Regime"],
        dropna=False,
    ).reindex(index=["low_vol", "high_vol"], columns=["low_vol", "high_vol"], fill_value=0)

    # Add a small smoothing term so no transition probability becomes exactly zero.
    transitions = transitions + 1
    transition_matrix = transitions.div(transitions.sum(axis=1), axis=0)

    mean_adjustment = mean_log_return - overall_mean
    current_regime = regime_df["Regime"].iloc[-1]

    return {
        "states": ["low_vol", "high_vol"],
        "current_regime": current_regime,
        "vol_threshold": vol_threshold,
        "mean_adjustment": mean_adjustment,
        "regime_stats": regime_stats,
        "transition_matrix": transition_matrix.to_dict(orient="index"),
    }


def normal_mc(last_price, mean_log_return, volatility, days=60, simulations=1000):
    """
    Run a Monte Carlo simulation using a normal distribution.
    """
    np.random.seed(RANDOM_SEED)
    simulated_prices = np.zeros((days, simulations))

    for sim in range(simulations):
        prices = [last_price]

        for _ in range(days):
            random_log_return = np.random.normal(loc=mean_log_return, scale=volatility)
            next_price = prices[-1] * np.exp(random_log_return)
            prices.append(next_price)

        simulated_prices[:, sim] = prices[1:]

    return simulated_prices


def bootstrap_mc(last_price, mean_log_return, historical_log_returns, days=60, simulations=1000):
    """
    Run a Monte Carlo simulation using historical bootstrap.

    Real historical daily returns are resampled with replacement so
    skewness and fat tails remain closer to the observed market data.
    """
    np.random.seed(RANDOM_SEED)

    if len(historical_log_returns) == 0:
        raise ValueError("Historical log return pool is empty.")

    bootstrap_shift = mean_log_return - np.mean(historical_log_returns)
    simulated_prices = np.zeros((days, simulations))

    for sim in range(simulations):
        prices = [last_price]

        for _ in range(days):
            sampled_log_return = np.random.choice(historical_log_returns)
            random_log_return = sampled_log_return + bootstrap_shift
            next_price = prices[-1] * np.exp(random_log_return)
            prices.append(next_price)

        simulated_prices[:, sim] = prices[1:]

    return simulated_prices


def regime_mc(last_price, regime_model, days=60, simulations=1000):
    """
    Run a Markov regime-switching Monte Carlo simulation.

    Each day belongs to either a low-volatility or high-volatility state.
    The next state is sampled from the transition probabilities estimated
    from recent history, and returns are then drawn using that state's
    mean and volatility.
    """
    np.random.seed(RANDOM_SEED)

    transition_matrix = regime_model["transition_matrix"]
    regime_stats = regime_model["regime_stats"]
    mean_adjustment = regime_model["mean_adjustment"]
    simulated_prices = np.zeros((days, simulations))

    for sim in range(simulations):
        prices = [last_price]
        current_regime = regime_model["current_regime"]

        for _ in range(days):
            transition_probs = transition_matrix[current_regime]
            next_regime = np.random.choice(
                regime_model["states"],
                p=[transition_probs["low_vol"], transition_probs["high_vol"]],
            )
            regime_mean = regime_stats[next_regime]["mean"] + mean_adjustment
            regime_volatility = regime_stats[next_regime]["volatility"]
            random_log_return = np.random.normal(loc=regime_mean, scale=regime_volatility)
            next_price = prices[-1] * np.exp(random_log_return)
            prices.append(next_price)
            current_regime = next_regime

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


def plot_results(price_df, simulation_results):
    """
    Visualize historical prices and simulation results.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1 + len(simulation_results), 1, figsize=(14, 6 + 4 * len(simulation_results)))

    if len(simulation_results) == 0:
        raise ValueError("No simulation results to plot.")

    axes = np.atleast_1d(axes)

    axes[0].plot(price_df["Date"], price_df["Close"], color="goldenrod", linewidth=2)
    axes[0].set_title(f"{TICKER} Closing Price - Last 10 Years", fontsize=14)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (USD)")

    future_x = np.arange(1, FUTURE_DAYS + 1)
    simulation_colors = {
        "normal": ("lightcoral", "red"),
        "bootstrap": ("skyblue", "navy"),
        "regime": ("khaki", "darkorange"),
    }

    for idx, (method, result) in enumerate(simulation_results.items(), start=1):
        path_color, mean_color = simulation_colors.get(method, ("lightgray", "black"))

        axes[idx].plot(future_x, result["simulated_prices"], color=path_color, alpha=0.03)
        axes[idx].plot(future_x, result["mean_path"], color=mean_color, linewidth=2, label="Mean Path")
        axes[idx].plot(
            future_x,
            result["p5_path"],
            color="green",
            linestyle="--",
            linewidth=2,
            label="5th Percentile",
        )
        axes[idx].plot(
            future_x,
            result["p50_path"],
            color="blue",
            linestyle="-.",
            linewidth=2,
            label="50th Percentile",
        )
        axes[idx].plot(
            future_x,
            result["p95_path"],
            color="purple",
            linestyle="--",
            linewidth=2,
            label="95th Percentile",
        )
        axes[idx].set_title(
            f"{TICKER} {method.capitalize()} Monte Carlo for Next {FUTURE_DAYS} Trading Days ({NUM_SIMULATIONS} runs)",
            fontsize=14,
        )
        axes[idx].set_xlabel("Future Trading Day")
        axes[idx].set_ylabel("Simulated Price (USD)")
        axes[idx].legend()

    plt.tight_layout()
    plt.show()


def main():
    configure_yfinance_cache()

    price_df = download_gold_data()
    exogenous_df = build_exogenous_feature_set(price_df)
    price_with_returns = calculate_daily_returns(price_df)

    recent_stats = calculate_recent_statistics(price_with_returns, SIMULATION_WINDOW)
    comparison_stats = calculate_window_comparison(price_with_returns, COMPARISON_WINDOWS)
    base_mean_log_return = recent_stats["mean_log_return"]
    volatility = recent_stats["volatility"]
    historical_log_returns = recent_stats["historical_log_returns"]
    last_price = price_with_returns["Close"].iloc[-1]
    trend_adjusted_drift, trend_snapshot = calculate_trend_adjusted_drift(price_df, base_mean_log_return)
    exogenous_drift_adjustment, exogenous_snapshot = calculate_exogenous_drift_adjustment(exogenous_df)
    mean_log_return = trend_adjusted_drift + exogenous_drift_adjustment
    regime_model = build_regime_model(price_with_returns.tail(SIMULATION_WINDOW + REGIME_VOL_WINDOW), mean_log_return)

    print(f"[2] Running Monte Carlo simulations... ({NUM_SIMULATIONS} runs each)")
    simulation_results = {}

    if "normal" in SIMULATION_METHODS:
        simulated_prices = normal_mc(
            last_price=last_price,
            mean_log_return=mean_log_return,
            volatility=volatility,
            days=FUTURE_DAYS,
            simulations=NUM_SIMULATIONS,
        )
        mean_path, p5_path, p50_path, p95_path = summarize_simulation(simulated_prices)
        simulation_results["normal"] = {
            "simulated_prices": simulated_prices,
            "mean_path": mean_path,
            "p5_path": p5_path,
            "p50_path": p50_path,
            "p95_path": p95_path,
        }

    if "bootstrap" in SIMULATION_METHODS:
        simulated_prices = bootstrap_mc(
            last_price=last_price,
            mean_log_return=mean_log_return,
            historical_log_returns=historical_log_returns,
            days=FUTURE_DAYS,
            simulations=NUM_SIMULATIONS,
        )
        mean_path, p5_path, p50_path, p95_path = summarize_simulation(simulated_prices)
        simulation_results["bootstrap"] = {
            "simulated_prices": simulated_prices,
            "mean_path": mean_path,
            "p5_path": p5_path,
            "p50_path": p50_path,
            "p95_path": p95_path,
        }

    if "regime" in SIMULATION_METHODS:
        simulated_prices = regime_mc(
            last_price=last_price,
            regime_model=regime_model,
            days=FUTURE_DAYS,
            simulations=NUM_SIMULATIONS,
        )
        mean_path, p5_path, p50_path, p95_path = summarize_simulation(simulated_prices)
        simulation_results["regime"] = {
            "simulated_prices": simulated_prices,
            "mean_path": mean_path,
            "p5_path": p5_path,
            "p50_path": p50_path,
            "p95_path": p95_path,
        }

    print("\n===== Gold Price Monte Carlo Simulation Result =====")
    print(f"Ticker: {TICKER}")
    print(f"Latest close price: ${last_price:.2f}")
    print(f"Simulation statistics window: recent {recent_stats['window']} trading days")
    print(f"Simulation methods: {', '.join(SIMULATION_METHODS)}")
    print(f"Base average daily log return: {base_mean_log_return:.6f}")
    print(f"Trend-adjusted daily drift: {trend_adjusted_drift:.6f}")
    print(f"Exogenous drift adjustment: {exogenous_drift_adjustment:.6f}")
    print(f"Final daily drift used in simulation: {mean_log_return:.6f}")
    print(f"Drift adjustment from trend signals: {trend_snapshot['drift_adjustment']:.6f}")
    print(f"Daily volatility: {volatility:.6f} ({volatility * 100:.4f}%)")

    print("\nTrend indicators:")
    print(
        f"- MA5: ${trend_snapshot['MA5']:.2f} | "
        f"MA20: ${trend_snapshot['MA20']:.2f} | "
        f"MA60: ${trend_snapshot['MA60']:.2f}"
    )
    print(
        f"- RSI({RSI_WINDOW}): {trend_snapshot['RSI']:.2f} | "
        f"MACD-like momentum: {trend_snapshot['MACD_Momentum']:.6f}"
    )
    print(
        f"- Recent 20-day return: {trend_snapshot['Return_20D'] * 100:.2f}% | "
        f"Recent 60-day return: {trend_snapshot['Return_60D'] * 100:.2f}%"
    )

    print("\nRecent window comparison:")
    for stats in comparison_stats:
        print(
            f"- {stats['window']:>3} days | mean log return: {stats['mean_log_return']:.6f} | "
            f"volatility: {stats['volatility']:.6f} ({stats['volatility'] * 100:.4f}%)"
        )

    print("\nExogenous signals:")
    print(
        f"- Dollar Index latest: {exogenous_snapshot['latest_dollar_index']:.2f} | "
        f"20D return: {exogenous_snapshot['Dollar_Index_20D']:.4f}"
    )
    print(
        f"- 10Y Treasury latest: {exogenous_snapshot['latest_treasury_10y']:.2f} | "
        f"20D change: {exogenous_snapshot['Treasury_10Y_20D_Change']:.4f}"
    )
    print(
        f"- 10Y Real Rate latest: {exogenous_snapshot['latest_real_rate_10y']:.2f} | "
        f"20D change: {exogenous_snapshot['Real_Rate_10Y_20D_Change']:.4f}"
    )
    print(
        f"- VIX latest: {exogenous_snapshot['latest_vix']:.2f} | "
        f"20D return: {exogenous_snapshot['VIX_20D']:.4f}"
    )
    print(
        f"- SPY latest: {exogenous_snapshot['latest_spy']:.2f} | "
        f"20D return: {exogenous_snapshot['SPY_20D']:.4f}"
    )
    print(
        f"- Crude Oil latest: {exogenous_snapshot['latest_crude_oil']:.2f} | "
        f"20D return: {exogenous_snapshot['Crude_Oil_20D']:.4f}"
    )
    print(f"- CPI latest: {exogenous_snapshot['latest_cpi']:.2f}")

    print("\nRegime model:")
    print(
        f"- Current regime: {regime_model['current_regime']} | "
        f"rolling-vol threshold: {regime_model['vol_threshold']:.6f}"
    )
    for regime_name in regime_model["states"]:
        stats = regime_model["regime_stats"][regime_name]
        print(
            f"- {regime_name}: mean {stats['mean'] + regime_model['mean_adjustment']:.6f} | "
            f"volatility {stats['volatility']:.6f} | observations {stats['count']}"
        )
    print(
        f"- Transition low_vol -> low_vol/high_vol: "
        f"{regime_model['transition_matrix']['low_vol']['low_vol']:.3f} / "
        f"{regime_model['transition_matrix']['low_vol']['high_vol']:.3f}"
    )
    print(
        f"- Transition high_vol -> low_vol/high_vol: "
        f"{regime_model['transition_matrix']['high_vol']['low_vol']:.3f} / "
        f"{regime_model['transition_matrix']['high_vol']['high_vol']:.3f}"
    )

    print("\nSimulation comparison:")
    for method, result in simulation_results.items():
        print(f"{method.capitalize()} Monte Carlo:")
        print(f"Expected mean price after {FUTURE_DAYS} days: ${result['mean_path'][-1]:.2f}")
        print(f"5th percentile after {FUTURE_DAYS} days: ${result['p5_path'][-1]:.2f}")
        print(f"50th percentile after {FUTURE_DAYS} days: ${result['p50_path'][-1]:.2f}")
        print(f"95th percentile after {FUTURE_DAYS} days: ${result['p95_path'][-1]:.2f}")
        print(
            f"Expected price range after {FUTURE_DAYS} days (5% to 95%): "
            f"${result['p5_path'][-1]:.2f} ~ ${result['p95_path'][-1]:.2f}"
        )

    plot_results(price_df, simulation_results)


if __name__ == "__main__":
    main()
