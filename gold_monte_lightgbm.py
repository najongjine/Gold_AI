from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# -----------------------------
# Configuration
# -----------------------------
HISTORY_PERIOD = "10y"
INTERVAL = "1d"
PREDICTION_HORIZON = 20
SMOOTHING_WINDOW = 20
TIME_SERIES_SPLITS = 5
RANDOM_SEED = 42

YF_TICKERS = {
    "Gold": "GC=F",
    "Dollar_Index": "DX-Y.NYB",
    "VIX": "^VIX",
    "Crude_Oil": "CL=F",
    "TIP": "TIP",
    "TLT": "TLT",
}


def configure_yfinance_cache() -> None:
    cache_dir = Path(__file__).resolve().parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


def download_yfinance_close_series(tickers: Dict[str, str]) -> pd.DataFrame:
    try:
        df = yf.download(
            list(tickers.values()),
            period=HISTORY_PERIOD,
            interval=INTERVAL,
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
        close_df.columns = [next(iter(tickers.values()))]

    close_df = close_df.reset_index()
    first_col = close_df.columns[0]
    if first_col != "Date":
        close_df = close_df.rename(columns={first_col: "Date"})

    rename_map = {"Date": "Date"}
    for alias, ticker in tickers.items():
        if ticker in close_df.columns:
            rename_map[ticker] = alias

    close_df = close_df.rename(columns=rename_map)
    close_df["Date"] = pd.to_datetime(close_df["Date"])
    close_df = close_df.sort_values("Date").reset_index(drop=True)

    for alias in tickers:
        if alias not in close_df.columns:
            close_df[alias] = np.nan

    return close_df[["Date"] + list(tickers.keys())]


def build_market_dataset() -> pd.DataFrame:
    configure_yfinance_cache()

    market_df = download_yfinance_close_series(YF_TICKERS)
    if market_df.empty:
        raise ValueError("Failed to download yfinance market data.")

    market_df = market_df.sort_values("Date").reset_index(drop=True)
    market_df = market_df.ffill().bfill()

    return market_df


def add_return_features(df: pd.DataFrame, column: str, windows: List[int]) -> None:
    for window in windows:
        df[f"{column}_ret_{window}"] = df[column].pct_change(window)


def add_momentum_features(df: pd.DataFrame, column: str) -> None:
    ma_mid = df[column].rolling(20).mean()
    ma_long = df[column].rolling(60).mean()

    df[f"{column}_ma20_gap"] = df[column] / ma_mid - 1
    df[f"{column}_ma20_ma60_gap"] = ma_mid / ma_long - 1


def add_volatility_features(df: pd.DataFrame, column: str, windows: List[int]) -> None:
    daily_return = df[column].pct_change()
    for window in windows:
        df[f"{column}_vol_{window}"] = daily_return.rolling(window).std()


def add_rsi_feature(df: pd.DataFrame, column: str, window: int = 14) -> None:
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"{column}_rsi_{window}"] = 100 - (100 / (1 + rs))


def add_macd_features(df: pd.DataFrame, column: str) -> None:
    ema12 = df[column].ewm(span=12, adjust=False).mean()
    ema26 = df[column].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    df[f"{column}_macd_line"] = macd_line
    df[f"{column}_macd_signal"] = signal
    df[f"{column}_macd_hist"] = macd_line - signal


def engineer_features(market_df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    df = market_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    price_columns = ["Gold", "Dollar_Index", "VIX", "Crude_Oil", "TIP", "TLT"]
    for column in price_columns:
        add_return_features(df, column, [1,20, 60])
        add_momentum_features(df, column)
        add_volatility_features(df, column, [20, 60])

    add_rsi_feature(df, "Gold")
    add_macd_features(df, "Gold")

    df["Gold_to_Dollar_ratio"] = df["Gold"] / df["Dollar_Index"]
    df["Gold_to_Oil_ratio"] = df["Gold"] / df["Crude_Oil"]
    df["Gold_to_TIP_ratio"] = df["Gold"] / df["TIP"]
    df["Gold_to_TLT_ratio"] = df["Gold"] / df["TLT"]
    df["TIP_to_TLT_ratio"] = df["TIP"] / df["TLT"]
    df["TIP_minus_TLT"] = df["TIP"] - df["TLT"]
    df["TIP_minus_TLT_change_20"] = df["TIP_minus_TLT"].diff(20)

    df["vix_gold_stress"] = df["VIX_ret_20"] - df["Gold_ret_20"]
    df["vix_tlt_stress"] = df["VIX_ret_20"] - df["TLT_ret_20"]
    df["dollar_tip_interaction"] = df["Dollar_Index_ret_20"] * df["TIP_ret_20"]
    df["dollar_tlt_interaction"] = df["Dollar_Index_ret_20"] * df["TLT_ret_20"]
    df["oil_tip_interaction"] = df["Crude_Oil_ret_20"] * df["TIP_ret_20"]
    df["oil_tlt_interaction"] = df["Crude_Oil_ret_20"] * df["TLT_ret_20"]

    smoothed_gold = df["Gold"].rolling(SMOOTHING_WINDOW).mean()
    future_smoothed_gold = smoothed_gold.shift(-PREDICTION_HORIZON)

    df["target_future_return"] = future_smoothed_gold / smoothed_gold - 1
    df["target_daily_drift"] = np.log1p(df["target_future_return"]) / PREDICTION_HORIZON

    df = df.replace([np.inf, -np.inf], np.nan)

    non_feature_columns = {"Date", "target_future_return", "target_daily_drift"}
    empty_feature_columns = [
        column for column in df.columns
        if column not in non_feature_columns and df[column].isna().all()
    ]
    if empty_feature_columns:
        df = df.drop(columns=empty_feature_columns)

    required_columns = get_feature_columns(df)
    if require_target:
        required_columns.append("target_daily_drift")
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "Date",
        "target_future_return",
        "target_daily_drift",
    }
    return [col for col in df.columns if col not in excluded]


def train_lightgbm_model(df: pd.DataFrame) -> Dict[str, object]:
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No usable feature columns were created. Check the downloaded source data.")

    if df.empty:
        raise ValueError(
            "The engineered dataset is empty. This usually means one of the data sources "
            "(often FRED macro data) did not download successfully."
        )

    if len(df) < 3:
        raise ValueError(f"Not enough rows to train the model after feature engineering: {len(df)}")

    X = df[feature_columns]
    y = df["target_daily_drift"]

    n_splits = min(TIME_SERIES_SPLITS, len(df) - 1)
    if n_splits < 2:
        raise ValueError(
            f"Not enough samples for time-series cross validation: {len(df)} rows available."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    mae_scores: List[float] = []
    directional_scores: List[float] = []
    last_model = None

    oof_predictions = np.full(len(df), np.nan)

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            importance_type="gain",
            verbose=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        preds = model.predict(X_test)
        oof_predictions[test_idx] = preds
        mae_scores.append(mean_absolute_error(y_test, preds))
        directional_scores.append(float(np.mean(np.sign(y_test.values) == np.sign(preds)) * 100))
        last_model = model

    if last_model is None:
        raise ValueError("Not enough data to train the LightGBM model.")

    final_model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        importance_type="gain",
        verbose=-1,
    )
    final_model.fit(X, y)

    importance = pd.Series(final_model.feature_importances_, index=feature_columns).sort_values(ascending=False)

    return {
        "model": final_model,
        "feature_columns": feature_columns,
        "mean_mae": float(np.mean(mae_scores)),
        "mean_directional_accuracy": float(np.mean(directional_scores)),
        "top_features": [(name, float(score)) for name, score in importance.head(15).items()],
        "oof_predictions": oof_predictions,
    }


def predict_latest(model: LGBMRegressor, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, float]:
    latest_row = df.iloc[[-1]]
    predicted_daily_drift = float(model.predict(latest_row[feature_columns])[0])
    predicted_horizon_return = float(np.expm1(predicted_daily_drift * PREDICTION_HORIZON))

    return {
        "prediction_date": latest_row["Date"].iloc[0].strftime("%Y-%m-%d"),
        "predicted_daily_drift": predicted_daily_drift,
        "predicted_horizon_return": predicted_horizon_return,
        "latest_gold_price": float(latest_row["Gold"].iloc[0]),
    }


def print_summary(training_result: Dict[str, object], latest_prediction: Dict[str, float]) -> None:
    print("\n===== Gold Monte LightGBM Summary =====")
    print(f"Prediction horizon: {PREDICTION_HORIZON} trading days")
    print(f"CV mean MAE (daily drift): {training_result['mean_mae']:.6f}")
    print(f"CV directional accuracy: {training_result['mean_directional_accuracy']:.2f}%")
    print(f"Latest prediction date: {latest_prediction['prediction_date']}")
    print(f"Latest gold price: ${latest_prediction['latest_gold_price']:.2f}")
    print(f"Predicted daily drift: {latest_prediction['predicted_daily_drift']:.6f}")
    print(
        f"Predicted {PREDICTION_HORIZON}-day return: "
        f"{latest_prediction['predicted_horizon_return'] * 100:.2f}%"
    )
    print("\nTop features:")
    for feature, score in training_result["top_features"]:
        print(f"- {feature}: {score:.4f}")


def build_and_predict() -> Dict[str, object]:
    market_df = build_market_dataset()
    feature_df = engineer_features(market_df, require_target=False)
    dataset = feature_df.dropna(subset=get_feature_columns(feature_df) + ["target_daily_drift"]).reset_index(drop=True)
    training_result = train_lightgbm_model(dataset)
    latest_prediction = predict_latest(
        training_result["model"],
        feature_df,
        training_result["feature_columns"],
    )
    return {
        "feature_df": feature_df,
        "dataset": dataset,
        "training_result": training_result,
        "latest_prediction": latest_prediction,
    }


def main() -> None:
    result = build_and_predict()
    print_summary(result["training_result"], result["latest_prediction"])


if __name__ == "__main__":
    main()
