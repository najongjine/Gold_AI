import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf

from gold_monte_lightgbm import (
    PREDICTION_HORIZON as LIGHTGBM_PREDICTION_HORIZON,
    build_and_predict as build_lightgbm_prediction,
)


# -----------------------------
# Basic settings
# -----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_p2dBsFZ9Mvfx@ep-divine-sea-ahijybu5-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
TABLE_NAME = "t_gold_model"
REPORT_TIMEZONE = "Asia/Seoul"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    ml VARCHAR NOT NULL DEFAULT '',
    gru VARCHAR NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_UPDATED_AT_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

CREATE_UPDATED_AT_TRIGGER_SQL = f"""
DROP TRIGGER IF EXISTS trg_t_gold_model_updated_at ON {TABLE_NAME};
CREATE TRIGGER trg_t_gold_model_updated_at
BEFORE UPDATE ON {TABLE_NAME}
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
"""

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

# Regime model settings
REGIME_VOL_WINDOW = 20
HIGH_VOL_QUANTILE = 0.65

# Predictive model blend settings
USE_LIGHTGBM_DRIFT = True
LIGHTGBM_DRIFT_WEIGHT = 0.65


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


def calculate_combined_drift(trend_adjusted_drift, model_daily_drift):
    """
    Blend the trend-based drift and the predictive model drift.

    The LightGBM output gives a directional expectation, while the Monte Carlo
    layer still handles uncertainty by sampling many future paths around it.
    """
    model_weight = LIGHTGBM_DRIFT_WEIGHT
    trend_weight = 1.0 - model_weight
    combined_drift = trend_adjusted_drift * trend_weight + model_daily_drift * model_weight

    return combined_drift, {
        "trend_weight": trend_weight,
        "model_weight": model_weight,
        "trend_adjusted_drift": trend_adjusted_drift,
        "model_daily_drift": model_daily_drift,
        "combined_drift": combined_drift,
    }


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


def column_exists(cur, column_name):
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND column_name = %s
        )
        """,
        (TABLE_NAME, column_name),
    )
    return bool(cur.fetchone()[0])


def ensure_results_table(cur):
    cur.execute(CREATE_TABLE_SQL)

    if not column_exists(cur, "ml"):
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN ml VARCHAR NOT NULL DEFAULT ''")

    if column_exists(cur, "gru") and not column_exists(cur, "monte"):
        cur.execute(f"ALTER TABLE {TABLE_NAME} RENAME COLUMN gru TO monte")
    elif not column_exists(cur, "monte"):
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN monte VARCHAR NOT NULL DEFAULT ''")

    if not column_exists(cur, "created_at"):
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

    if not column_exists(cur, "updated_at"):
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

    cur.execute(CREATE_UPDATED_AT_FUNCTION_SQL)
    cur.execute(CREATE_UPDATED_AT_TRIGGER_SQL)


def build_monte_report(
    price_df,
    recent_stats,
    comparison_stats,
    base_mean_log_return,
    mean_log_return,
    combined_drift,
    trend_snapshot,
    predictive_snapshot,
    regime_model,
    simulation_results,
    last_price,
):
    latest_data_date = price_df["Date"].iloc[-1].strftime("%Y-%m-%d")
    lines = [
        "===== Gold Monte Carlo Result =====",
        f"Ticker: {TICKER}",
        f"Latest data date: {latest_data_date}",
        f"Latest close price: ${last_price:.2f}",
        f"Simulation window: {recent_stats['window']} trading days",
        f"Future horizon: {FUTURE_DAYS} trading days",
        f"Number of simulations: {NUM_SIMULATIONS}",
        f"Simulation methods: {', '.join(SIMULATION_METHODS)}",
        f"Base average daily log return: {base_mean_log_return:.6f}",
        f"Trend-adjusted daily drift: {mean_log_return:.6f}",
        f"Monte Carlo input daily drift: {combined_drift:.6f}",
        f"Trend drift adjustment: {trend_snapshot['drift_adjustment']:.6f}",
        f"Daily volatility: {recent_stats['volatility']:.6f}",
        "",
        "[Predictive model blend]",
        f"Status: {predictive_snapshot['message']}",
    ]

    if predictive_snapshot["enabled"]:
        lines.extend(
            [
                f"Prediction date: {predictive_snapshot['prediction_date']}",
                f"Prediction horizon: {LIGHTGBM_PREDICTION_HORIZON} trading days",
                f"Model daily drift: {predictive_snapshot['model_daily_drift']:.6f}",
                f"Predicted horizon return: {predictive_snapshot['predicted_horizon_return'] * 100:.2f}%",
                f"Blend weights trend/model: {predictive_snapshot['trend_weight']:.2f} / {predictive_snapshot['model_weight']:.2f}",
                f"CV mean MAE: {predictive_snapshot['cv_mean_mae']:.6f}",
                f"CV directional accuracy: {predictive_snapshot['cv_directional_accuracy']:.2f}%",
                "Top features:",
            ]
        )
        for feature, score in predictive_snapshot["top_features"]:
            lines.append(f"- {feature}: {score:.4f}")

    lines.extend(
        [
            "",
            "[Trend indicators]",
            f"MA5/MA20/MA60: ${trend_snapshot['MA5']:.2f} / ${trend_snapshot['MA20']:.2f} / ${trend_snapshot['MA60']:.2f}",
            f"RSI({RSI_WINDOW}): {trend_snapshot['RSI']:.2f}",
            f"MACD-like momentum: {trend_snapshot['MACD_Momentum']:.6f}",
            f"Recent 20-day return: {trend_snapshot['Return_20D'] * 100:.2f}%",
            f"Recent 60-day return: {trend_snapshot['Return_60D'] * 100:.2f}%",
            "",
            "[Recent window comparison]",
        ]
    )

    for stats in comparison_stats:
        lines.append(
            f"- {stats['window']} days | mean log return: {stats['mean_log_return']:.6f} | "
            f"volatility: {stats['volatility']:.6f}"
        )

    lines.extend(
        [
            "",
            "[Regime model]",
            f"Current regime: {regime_model['current_regime']}",
            f"Rolling-vol threshold: {regime_model['vol_threshold']:.6f}",
        ]
    )
    for regime_name in regime_model["states"]:
        stats = regime_model["regime_stats"][regime_name]
        lines.append(
            f"- {regime_name}: mean {stats['mean'] + regime_model['mean_adjustment']:.6f} | "
            f"volatility {stats['volatility']:.6f} | observations {stats['count']}"
        )

    lines.extend(
        [
            "",
            "[Simulation comparison]",
        ]
    )
    for method, result in simulation_results.items():
        lines.extend(
            [
                f"{method.capitalize()} Monte Carlo:",
                f"- Expected mean price after {FUTURE_DAYS} days: ${result['mean_path'][-1]:.2f}",
                f"- 5th percentile after {FUTURE_DAYS} days: ${result['p5_path'][-1]:.2f}",
                f"- 50th percentile after {FUTURE_DAYS} days: ${result['p50_path'][-1]:.2f}",
                f"- 95th percentile after {FUTURE_DAYS} days: ${result['p95_path'][-1]:.2f}",
            ]
        )

    return "\n".join(lines)


def save_monte_result_to_postgres(monte_report):
    print(f"[3] Saving Monte Carlo result to {TABLE_NAME}...")
    report_date = datetime.now(ZoneInfo(REPORT_TIMEZONE)).date()
    conn = psycopg2.connect(DATABASE_URL)

    try:
        with conn:
            with conn.cursor() as cur:
                ensure_results_table(cur)
                cur.execute(
                    f"""
                    SELECT id
                    FROM {TABLE_NAME}
                    WHERE (created_at AT TIME ZONE %s)::date = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (REPORT_TIMEZONE, report_date),
                )
                existing_row = cur.fetchone()

                if existing_row is None:
                    cur.execute(
                        f"INSERT INTO {TABLE_NAME} (monte) VALUES (%s)",
                        (monte_report,),
                    )
                    print(f"Inserted Monte Carlo result for {report_date}.")
                else:
                    cur.execute(
                        f"UPDATE {TABLE_NAME} SET monte = %s WHERE id = %s",
                        (monte_report, existing_row[0]),
                    )
                    print(f"Updated Monte Carlo result for {report_date}. id={existing_row[0]}")
    finally:
        conn.close()


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
    price_with_returns = calculate_daily_returns(price_df)

    recent_stats = calculate_recent_statistics(price_with_returns, SIMULATION_WINDOW)
    comparison_stats = calculate_window_comparison(price_with_returns, COMPARISON_WINDOWS)
    base_mean_log_return = recent_stats["mean_log_return"]
    volatility = recent_stats["volatility"]
    historical_log_returns = recent_stats["historical_log_returns"]
    last_price = price_with_returns["Close"].iloc[-1]
    mean_log_return, trend_snapshot = calculate_trend_adjusted_drift(price_df, base_mean_log_return)

    predictive_result = None
    predictive_snapshot = {
        "enabled": False,
        "combined_drift": mean_log_return,
        "message": "Predictive model blend disabled.",
    }

    if USE_LIGHTGBM_DRIFT:
        try:
            predictive_result = build_lightgbm_prediction()
            combined_drift, blend_snapshot = calculate_combined_drift(
                mean_log_return,
                predictive_result["latest_prediction"]["predicted_daily_drift"],
            )
            predictive_snapshot = {
                "enabled": True,
                "message": "Using blended predictive drift from LightGBM + trend signals.",
                "prediction_date": predictive_result["latest_prediction"]["prediction_date"],
                "predicted_horizon_return": predictive_result["latest_prediction"]["predicted_horizon_return"],
                "cv_mean_mae": predictive_result["training_result"]["mean_mae"],
                "cv_directional_accuracy": predictive_result["training_result"]["mean_directional_accuracy"],
                "top_features": predictive_result["training_result"]["top_features"][:8],
                **blend_snapshot,
            }
        except Exception as exc:
            predictive_snapshot = {
                "enabled": False,
                "combined_drift": mean_log_return,
                "message": f"LightGBM blend unavailable, falling back to trend drift only: {exc}",
            }

    combined_drift = predictive_snapshot["combined_drift"]
    regime_model = build_regime_model(price_with_returns.tail(SIMULATION_WINDOW + REGIME_VOL_WINDOW), combined_drift)

    print(f"[2] Running Monte Carlo simulations... ({NUM_SIMULATIONS} runs each)")
    simulation_results = {}

    if "normal" in SIMULATION_METHODS:
        simulated_prices = normal_mc(
            last_price=last_price,
            mean_log_return=combined_drift,
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
            mean_log_return=combined_drift,
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
    print(f"Trend-adjusted daily drift: {mean_log_return:.6f}")
    print(f"Monte Carlo input daily drift: {combined_drift:.6f}")
    print(f"Drift adjustment from trend signals: {trend_snapshot['drift_adjustment']:.6f}")
    print(f"Daily volatility: {volatility:.6f} ({volatility * 100:.4f}%)")

    print("\nPredictive model blend:")
    print(f"- Status: {predictive_snapshot['message']}")
    if predictive_snapshot["enabled"]:
        print(
            f"- LightGBM prediction date: {predictive_snapshot['prediction_date']} | "
            f"horizon: {LIGHTGBM_PREDICTION_HORIZON} trading days"
        )
        print(
            f"- Model daily drift: {predictive_snapshot['model_daily_drift']:.6f} | "
            f"predicted {LIGHTGBM_PREDICTION_HORIZON}-day return: "
            f"{predictive_snapshot['predicted_horizon_return'] * 100:.2f}%"
        )
        print(
            f"- Blend weights trend/model: "
            f"{predictive_snapshot['trend_weight']:.2f} / {predictive_snapshot['model_weight']:.2f}"
        )
        print(
            f"- CV mean MAE: {predictive_snapshot['cv_mean_mae']:.6f} | "
            f"CV directional accuracy: {predictive_snapshot['cv_directional_accuracy']:.2f}%"
        )
        print("- Top model features:")
        for feature, score in predictive_snapshot["top_features"]:
            print(f"  {feature}: {score:.4f}")

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

    monte_report = build_monte_report(
        price_df=price_df,
        recent_stats=recent_stats,
        comparison_stats=comparison_stats,
        base_mean_log_return=base_mean_log_return,
        mean_log_return=mean_log_return,
        combined_drift=combined_drift,
        trend_snapshot=trend_snapshot,
        predictive_snapshot=predictive_snapshot,
        regime_model=regime_model,
        simulation_results=simulation_results,
        last_price=last_price,
    )
    save_monte_result_to_postgres(monte_report)

    plot_results(price_df, simulation_results)


if __name__ == "__main__":
    main()
