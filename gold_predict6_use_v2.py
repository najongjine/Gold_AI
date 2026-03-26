import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 설정값
# ============================================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "",
)
TABLE_NAME = "t_gold_model"
DATA_PERIOD = "10y"
SMOOTHING_WINDOW = 60
TARGET_WINDOW = 60
CV_GAP = TARGET_WINDOW
RECENT_BACKTEST_WINDOW = 252

# 기존 코드 + GRU 코드의 자산을 합침
TICKER_CANDIDATES = {
    "Gold": ["GC=F"],
    "Dollar_Index": ["DX-Y.NYB", "UUP"],
    "US10Y_Treasury": ["^TNX"],
    "TIPS_ETF": ["TIP"],
    "VIX": ["^VIX"],
    "S&P500": ["^GSPC"],
    "Oil": ["CL=F"],
    "Silver": ["SI=F"],
    "GLD": ["GLD"],
}

YF_CACHE_DIR = os.path.join(os.getcwd(), ".yfinance_cache")
os.makedirs(YF_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YF_CACHE_DIR)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    ml TEXT NOT NULL DEFAULT '',
    gru TEXT NOT NULL DEFAULT '',
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


# ============================================================
# 데이터 수집
# ============================================================
def download_single_close(ticker: str, period: str) -> Optional[pd.Series]:
    """ticker 하나를 받아 Close 시계열만 반환"""
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.levels[0]:
            return None
        close_series = data["Close"].iloc[:, 0]
    else:
        if "Close" not in data.columns:
            return None
        close_series = data["Close"]

    return close_series


def fetch_gold_data(period: str = DATA_PERIOD) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    print("데이터 수집을 시작합니다.")

    resolved_tickers: Dict[str, str] = {}
    series_list = []

    for name, candidates in TICKER_CANDIDATES.items():
        print(f"\n{name} 후보 {candidates} 확인 중...")
        selected_series = None
        selected_ticker = None

        for ticker in candidates:
            try:
                s = download_single_close(ticker, period)
                if s is not None and not s.empty:
                    selected_series = s.rename(name)
                    selected_ticker = ticker
                    print(f"  사용 ticker: {ticker}")
                    break
                print(f"  실패 또는 빈 데이터: {ticker}")
            except Exception as exc:
                print(f"  에러 ({ticker}): {exc}")

        if selected_series is not None:
            resolved_tickers[name] = selected_ticker
            series_list.append(selected_series.to_frame())
        else:
            print(f"경고: {name}는 사용할 수 있는 ticker를 찾지 못했습니다.")

    if not series_list:
        print("수집된 데이터가 없습니다.")
        return None, resolved_tickers

    df = pd.concat(series_list, axis=1).sort_index()
    df.index.name = "Date"
    # Avoid backward fill on time-series inputs to prevent leaking future values.
    df = df.ffill()

    print("\n데이터 수집 및 병합 완료")
    print("데이터 크기:", df.shape)
    print("컬럼:", df.columns.tolist())

    df.to_csv("gold_data_raw_extended.csv")
    return df, resolved_tickers


# ============================================================
# 스무딩
# ============================================================
def preprocess_and_smooth(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    print(f"\n{window}일 이동평균 스무딩을 시작합니다.")
    df_smoothed = df.rolling(window=window).mean().dropna()
    print("스무딩 후 데이터 크기:", df_smoothed.shape)
    df_smoothed.to_csv("gold_data_smoothed_extended.csv")
    return df_smoothed


# ============================================================
# 기술적 지표 함수
# ============================================================
def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# 파생변수 생성
# ============================================================
def engineer_features(df_raw: pd.DataFrame, df_smoothed: pd.DataFrame) -> pd.DataFrame:
    print("\n파생변수 생성을 시작합니다.")

    common_index = df_smoothed.index
    df_raw = df_raw.loc[common_index].copy()
    df_smoothed = df_smoothed.loc[common_index].copy()
    feat = pd.DataFrame(index=common_index)
    return_lags = [1, 5, 20, 60]
    ma_windows = [20, 60]
    vol_windows = [20, 60]
    mom_windows = [20, 60]

    # --------------------------------------------------
    # 1. 모든 자산 공통: 수익률, 변동성, 이평, 괴리율, 모멘텀
    # --------------------------------------------------
    for col in df_raw.columns:
        price = df_raw[col]
        smooth_price = df_smoothed[col]

        for lag in return_lags:
            feat[f"{col}_ret_{lag}"] = price.pct_change(lag) * 100

        for ma_win in ma_windows:
            feat[f"{col}_ma_{ma_win}"] = price.rolling(ma_win).mean()
            feat[f"{col}_disparity_ma_{ma_win}"] = (price / feat[f"{col}_ma_{ma_win}"] - 1) * 100

        # 기존 코드의 smoothed 기준 괴리도 유지
        if col == "Gold":
            feat[f"{col}_disparity_smooth60"] = (price / smooth_price - 1) * 100

        for vol_win in vol_windows:
            feat[f"{col}_vol_{vol_win}"] = price.pct_change().rolling(vol_win).std() * 100

        for mom_win in mom_windows:
            feat[f"{col}_mom_{mom_win}"] = (price / price.shift(mom_win) - 1) * 100

    # --------------------------------------------------
    # 2. 골드 전용 feature (기존 + GRU 확장)
    # --------------------------------------------------
    gold = df_raw["Gold"]

    feat["Gold_smooth_ret_20"] = df_smoothed["Gold"].pct_change(20) * 100
    feat["Gold_smooth_ret_60"] = df_smoothed["Gold"].pct_change(60) * 100

    # MA gap
    feat["Gold_ma20_ma60_gap"] = (feat["Gold_ma_20"] - feat["Gold_ma_60"]) / feat["Gold_ma_60"] * 100

    # RSI / MACD
    feat["Gold_RSI_14"] = calc_rsi(gold, 14)
    ema12 = gold.ewm(span=12, adjust=False).mean()
    ema26 = gold.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat["Gold_MACD_Line"] = macd_line
    feat["Gold_MACD_Signal"] = signal_line
    feat["Gold_MACD_Hist"] = macd_line - signal_line

    # lag returns
    for lag in [1, 5, 10, 20]:
        feat[f"Gold_ret_1_lag_{lag}"] = feat["Gold_ret_1"].shift(lag)
    for lag in [5, 20]:
        feat[f"Gold_ret_5_lag_{lag}"] = feat["Gold_ret_5"].shift(lag)

    # --------------------------------------------------
    # 3. 자산 간 비율 feature
    # --------------------------------------------------
    if {"Gold", "Silver"}.issubset(df_raw.columns):
        feat["Gold_to_Silver_Ratio"] = df_raw["Gold"] / df_raw["Silver"]

    if {"Gold", "Dollar_Index"}.issubset(df_raw.columns):
        feat["Gold_to_Dollar_Ratio"] = df_raw["Gold"] / df_raw["Dollar_Index"]

    if {"Gold", "GLD"}.issubset(df_raw.columns):
        feat["Gold_to_GLD_Ratio"] = df_raw["Gold"] / df_raw["GLD"]

    # --------------------------------------------------
    # 4. 실질금리 대용치
    # --------------------------------------------------
    if {"US10Y_Treasury", "TIPS_ETF"}.issubset(df_raw.columns):
        feat["Real_Rate_Proxy"] = df_raw["US10Y_Treasury"] - df_raw["TIPS_ETF"].pct_change(60) * 100

    # --------------------------------------------------
    # 5. 달력 feature
    # --------------------------------------------------
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    drop_cols = [col for col in feat.columns if col.endswith(("_ma_20", "_ma_60"))]
    feat = feat.drop(columns=drop_cols)

    # --------------------------------------------------
    # 6. 정리
    # --------------------------------------------------
    feat = feat.replace([np.inf, -np.inf], np.nan)
    # Keep feature filling causal; leading NaNs are dropped instead of backfilled.
    feat = feat.ffill().dropna()

    print("파생변수 생성 완료")
    print("데이터 크기:", feat.shape)
    print("생성 컬럼 수:", len(feat.columns))

    feat.to_csv("gold_features_extended.csv")
    return feat


# ============================================================
# 타겟 생성
# ============================================================
def create_target(df_smoothed: pd.DataFrame, window: int = TARGET_WINDOW) -> pd.Series:
    print(f"\n타겟 변수 생성({window}거래일 뒤 60일 이동평균 변화율)")
    future_ma = df_smoothed["Gold"].shift(-window)
    target = (future_ma / df_smoothed["Gold"] - 1) * 100
    target.name = "Target_Return"
    return target


def make_baseline_predictions(X_part: pd.DataFrame, baseline_feature_name: str) -> Dict[str, np.ndarray]:
    return {
        "zero_return": np.zeros(len(X_part), dtype=float),
        "last_60_return": X_part["Gold_ret_60"].to_numpy(dtype=float),
        "single_feature": X_part[baseline_feature_name].to_numpy(dtype=float),
    }


def predict_with_strategy(
    strategy_key: str,
    model: LGBMRegressor,
    recent_row: pd.DataFrame,
    baseline_feature_name: str,
) -> float:
    if strategy_key == "model":
        return float(model.predict(recent_row)[0])
    if strategy_key == "zero_return":
        return 0.0
    if strategy_key == "last_60_return":
        return float(recent_row["Gold_ret_60"].iloc[0])
    if strategy_key == "single_feature":
        return float(recent_row[baseline_feature_name].iloc[0])
    raise ValueError(f"Unknown prediction strategy: {strategy_key}")


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true.values) == np.sign(y_pred)) * 100),
    }


def evaluate_recent_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_feature_name: str,
    strategy_labels: Dict[str, str],
    window: int = RECENT_BACKTEST_WINDOW,
) -> Dict[str, object]:
    backtest_size = min(window, len(X) // 3)
    split_idx = len(X) - backtest_size
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    backtest_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
        importance_type="gain",
        verbose=-1,
    )
    backtest_model.fit(X_train, y_train)

    baseline_preds = make_baseline_predictions(X_test, baseline_feature_name)
    strategy_predictions = {
        "model": backtest_model.predict(X_test),
        **baseline_preds,
    }

    strategy_metrics = []
    for key, preds in strategy_predictions.items():
        metrics = calculate_metrics(y_test, preds)
        strategy_metrics.append(
            {
                "key": key,
                "label": strategy_labels[key],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "directional_accuracy": metrics["directional_accuracy"],
            }
        )

    return {
        "window_size": int(len(X_test)),
        "start_date": X_test.index[0].strftime("%Y-%m-%d"),
        "end_date": X_test.index[-1].strftime("%Y-%m-%d"),
        "strategy_metrics": strategy_metrics,
    }


def build_reliability_assessment(
    selected_strategy_label: str,
    selected_cv_mae: float,
    selected_recent_metrics: Dict[str, float],
    baseline_recent_metrics: Dict[str, float],
) -> Tuple[str, str]:
    recent_mae = selected_recent_metrics["mae"]
    recent_r2 = selected_recent_metrics["r2"]
    baseline_recent_mae = baseline_recent_metrics["mae"]

    if recent_r2 > 0 and recent_mae <= baseline_recent_mae:
        return "보통", "최근 1년 테스트에서는 baseline 이상 성능이 확인됐지만, 오차 폭은 아직 큽니다."
    if recent_mae <= selected_cv_mae * 1.1:
        return "낮음", "최근 1년 성능은 유지됐지만 설명력은 약해서 보조 신호로만 보는 편이 안전합니다."
    return "매우 낮음", "최근 1년에서도 오차가 커서 방향 참고용 이상으로 해석하면 위험합니다."


# ============================================================
# 학습 및 검증
# ============================================================
def train_and_evaluate_model(df_final: pd.DataFrame) -> Dict[str, object]:
    print("\n모델 학습 및 검증을 시작합니다.")

    X = df_final.drop(columns=["Target_Return"])
    y = df_final["Target_Return"]
    baseline_feature_name = "Gold_disparity_smooth60" if "Gold_disparity_smooth60" in X.columns else X.columns[0]

    print(f"[모델 학습 피처 수] {len(X.columns)}개")

    tscv = TimeSeriesSplit(n_splits=5, gap=CV_GAP)
    mae_list = []
    rmse_list = []
    r2_list = []
    dir_acc_list = []
    baseline_metrics = {
        "zero_return": {"label": "Always 0%", "mae": [], "rmse": [], "r2": [], "dir_acc": []},
        "last_60_return": {"label": "Last 60d Return", "mae": [], "rmse": [], "r2": [], "dir_acc": []},
        "single_feature": {"label": f"Baseline Model ({baseline_feature_name})", "mae": [], "rmse": [], "r2": [], "dir_acc": []},
    }
    strategy_labels = {"model": "LightGBM Model"}
    strategy_labels.update({key: val["label"] for key, val in baseline_metrics.items()})

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
        importance_type="gain",
        verbose=-1,
    )

    all_preds = np.full(len(y), np.nan)
    all_test_idx = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        print(f"\n===== Fold {fold} =====")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        train_end = X.index[train_index[-1]].strftime("%Y-%m-%d")
        test_start = X.index[test_index[0]].strftime("%Y-%m-%d")
        print(f"Train end: {train_end}")
        print(f"Test start: {test_start}")
        print(f"Embargo gap: {CV_GAP} trading days")

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        baseline_preds = make_baseline_predictions(X_test, baseline_feature_name)

        metrics = calculate_metrics(y_test, preds)
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]
        dir_acc = metrics["directional_accuracy"]

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        dir_acc_list.append(dir_acc)

        all_preds[test_index] = preds
        all_test_idx.extend(test_index.tolist())

        for key, baseline_pred in baseline_preds.items():
            baseline_eval = calculate_metrics(y_test, baseline_pred)
            baseline_mae = baseline_eval["mae"]
            baseline_rmse = baseline_eval["rmse"]
            baseline_r2 = baseline_eval["r2"]
            baseline_dir_acc = baseline_eval["directional_accuracy"]
            baseline_metrics[key]["mae"].append(baseline_mae)
            baseline_metrics[key]["rmse"].append(baseline_rmse)
            baseline_metrics[key]["r2"].append(baseline_r2)
            baseline_metrics[key]["dir_acc"].append(baseline_dir_acc)

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"방향성 적중률: {dir_acc:.2f}%")

    mean_mae = float(np.mean(mae_list))
    mean_rmse = float(np.mean(rmse_list))
    mean_r2 = float(np.mean(r2_list))
    mean_directional_accuracy = float(np.mean(dir_acc_list))
    baseline_summary = []

    for key, baseline in baseline_metrics.items():
        baseline_summary.append(
            {
                "key": key,
                "label": baseline["label"],
                "mean_mae": float(np.mean(baseline["mae"])),
                "mean_rmse": float(np.mean(baseline["rmse"])),
                "mean_r2": float(np.mean(baseline["r2"])),
                "mean_directional_accuracy": float(np.mean(baseline["dir_acc"])),
            }
        )

    print("\n[검증 결과 요약]")
    print(f"평균 MAE: {mean_mae:.4f}")
    print(f"평균 RMSE: {mean_rmse:.4f}")
    print(f"평균 R2: {mean_r2:.4f}")
    print(f"평균 방향성 적중률: {mean_directional_accuracy:.2f}%")

    model_summary = {
        "key": "model",
        "label": strategy_labels["model"],
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "mean_directional_accuracy": mean_directional_accuracy,
    }
    baseline_model_summary = next(item for item in baseline_summary if item["key"] == "single_feature")
    recent_backtest = evaluate_recent_backtest(X, y, baseline_feature_name, strategy_labels)
    recent_lookup = {item["key"]: item for item in recent_backtest["strategy_metrics"]}

    use_lightgbm = (
        model_summary["mean_mae"] < baseline_model_summary["mean_mae"]
        and recent_lookup["model"]["mae"] < recent_lookup["single_feature"]["mae"]
    )
    selected_strategy = model_summary if use_lightgbm else baseline_model_summary
    reliability_level, reliability_comment = build_reliability_assessment(
        selected_strategy_label=selected_strategy["label"],
        selected_cv_mae=selected_strategy["mean_mae"],
        selected_recent_metrics=recent_lookup[selected_strategy["key"]],
        baseline_recent_metrics=recent_lookup["single_feature"],
    )

    # 전체 데이터 재학습
    model.fit(X, y)

    print("\n[Baseline Comparison]")
    for baseline in baseline_summary:
        print(
            f"{baseline['label']}: "
            f"MAE={baseline['mean_mae']:.4f}, "
            f"RMSE={baseline['mean_rmse']:.4f}, "
            f"R2={baseline['mean_r2']:.4f}, "
            f"DirAcc={baseline['mean_directional_accuracy']:.2f}%"
        )

    print(
        f"\n[최종 예측에 사용할 방식] {selected_strategy['label']} "
        f"(CV MAE={selected_strategy['mean_mae']:.4f})"
    )
    print(
        f"[최근 1년 백테스트] {recent_backtest['start_date']} ~ {recent_backtest['end_date']} "
        f"({recent_backtest['window_size']}개 샘플)"
    )
    for item in recent_backtest["strategy_metrics"]:
        print(
            f"{item['label']}: "
            f"MAE={item['mae']:.4f}, "
            f"RMSE={item['rmse']:.4f}, "
            f"R2={item['r2']:.4f}, "
            f"DirAcc={item['directional_accuracy']:.2f}%"
        )
    print(f"[신뢰도] {reliability_level} - {reliability_comment}")

    try:
        valid_mask = ~np.isnan(all_preds)
        plt.figure(figsize=(12, 6))
        plt.plot(y.index[valid_mask], y[valid_mask], label="Actual Return", alpha=0.6)
        plt.plot(y.index[valid_mask], all_preds[valid_mask], label="Predicted Return", linewidth=1.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Gold Future MA60 Return Prediction (Extended Features, MAE={mean_mae:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("prediction_result_extended.png")
        print("예측 결과 그래프 저장 완료: prediction_result_extended.png")
    except Exception as exc:
        print(f"시각화 중 에러 발생: {exc}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = [(feature, float(score)) for feature, score in importances.head(20).items()]

    print("\n[주요 피처 중요도 Top 20]")
    print(importances.head(20))

    return {
        "model": model,
        "feature_names": X.columns.tolist(),
        "baseline_feature_name": baseline_feature_name,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "mean_directional_accuracy": mean_directional_accuracy,
        "model_summary": model_summary,
        "baseline_summary": baseline_summary,
        "best_strategy_key": selected_strategy["key"],
        "best_strategy_label": selected_strategy["label"],
        "best_strategy_mae": selected_strategy["mean_mae"],
        "recent_backtest": recent_backtest,
        "reliability_level": reliability_level,
        "reliability_comment": reliability_comment,
        "top_features": top_features,
    }


# ============================================================
# 최신 시점 예측
# ============================================================
def predict_future_trend(
    model: LGBMRegressor,
    df_features: pd.DataFrame,
    df_raw: pd.DataFrame,
    feature_names: List[str],
    strategy_key: str,
    strategy_label: str,
    baseline_feature_name: str,
    target_window: int = TARGET_WINDOW,
    smoothing_window: int = SMOOTHING_WINDOW,
) -> Tuple[float, str, float, float, str]:
    print("\n" + "=" * 60)
    print("현재 시점 기준 향후 금값 추세 예측")
    print("=" * 60)

    recent_row = df_features[feature_names].iloc[[-1]].copy()
    recent_date = recent_row.index[0].strftime("%Y-%m-%d")
    prediction = predict_with_strategy(strategy_key, model, recent_row, baseline_feature_name)

    current_gold = float(df_raw.loc[recent_row.index[0], "Gold"])
    current_ma60 = float(df_raw["Gold"].rolling(smoothing_window).mean().loc[recent_row.index[0]])
    predicted_future_ma60 = current_ma60 * (1 + prediction / 100.0)

    print(f"기준일: {recent_date}")
    print(f"현재 금 가격: {current_gold:.2f}")
    print(f"현재 {smoothing_window}일 이동평균: {current_ma60:.2f}")
    print(f"예측 미래 {smoothing_window}일 이동평균: {predicted_future_ma60:.2f}")
    print(f"현재 이동평균 대비 변화율 예측: {prediction:.2f}%")

    if prediction > 0:
        print(f"결론: 상승 추세 예상 (+{prediction:.2f}%)")
    else:
        print(f"결론: 하락 추세 예상 ({prediction:.2f}%)")

    print(f"사용 예측 방식: {strategy_label}")

    return prediction, recent_date, current_ma60, predicted_future_ma60, strategy_label


# ============================================================
# 리포트 생성
# ============================================================
def build_report_text(
    feature_names: List[str],
    mean_mae: float,
    mean_rmse: float,
    mean_r2: float,
    mean_directional_accuracy: float,
    baseline_summary: List[Dict[str, float]],
    top_features: List[Tuple[str, float]],
    resolved_tickers: Dict[str, str],
    reference_date: str,
    prediction: float,
    current_ma60: float,
    predicted_future_ma60: float,
    period: str,
    smoothing_window: int,
    target_window: int,
) -> str:
    trend = "상승 추세 예상" if prediction > 0 else "하락 추세 예상"

    lines = [
        "[데이터 출처 및 입력 지표]",
        f"- 수집 기간: 최근 {period}",
        "- 수집 소스: yfinance",
    ]

    for key, val in resolved_tickers.items():
        lines.append(f"- {key}: {val}")

    lines.extend(
        [
            "",
            "[모델 및 타겟 정의]",
            "- 모델: LightGBM Regressor",
            f"- 스무딩: {smoothing_window}거래일 이동평균",
            f"- 예측 타겟: 향후 {target_window}거래일 뒤 Gold {smoothing_window}일 이동평균의 현재 대비 변화율(%)",
            f"- 검증 방식: TimeSeriesSplit 5-fold (gap={CV_GAP})",
            "- 확장 내용: 기존 거시지표 + GRU 코드의 연관자산/비율/모멘텀/달력 feature 통합",
            "",
            f"[모델 학습 피처 수] {len(feature_names)}개",
            "",
            "[검증 결과 요약]",
            f"평균 MAE: {mean_mae:.4f}",
            f"평균 RMSE: {mean_rmse:.4f}",
            f"평균 R2: {mean_r2:.4f}",
            f"평균 방향성 적중률: {mean_directional_accuracy:.2f}%",
            "",
            "[주요 피처 중요도 Top 20]",
        ]
    )

    lines.append("[Baseline Comparison]")
    for baseline in baseline_summary:
        lines.append(
            f"{baseline['label']}: "
            f"MAE={baseline['mean_mae']:.4f}, "
            f"RMSE={baseline['mean_rmse']:.4f}, "
            f"R2={baseline['mean_r2']:.4f}, "
            f"DirAcc={baseline['mean_directional_accuracy']:.2f}%"
        )

    lines.append("")

    for feature, score in top_features:
        lines.append(f"{feature}: {score:.6f}")

    lines.extend(
        [
            "",
            "[최신 시점 예측 결과]",
            f"- 기준일: {reference_date}",
            f"- 현재 MA60: {current_ma60:.2f}",
            f"- 예측 미래 MA60: {predicted_future_ma60:.2f}",
            f"- 예측 변화율: {prediction:.2f}%",
            f"- 결론: {trend}",
        ]
    )

    return "\n".join(lines)


def build_report_text_v2(
    feature_names: List[str],
    mean_mae: float,
    mean_rmse: float,
    mean_r2: float,
    mean_directional_accuracy: float,
    model_summary: Dict[str, float],
    baseline_summary: List[Dict[str, float]],
    best_strategy_label: str,
    best_strategy_mae: float,
    recent_backtest: Dict[str, object],
    reliability_level: str,
    reliability_comment: str,
    top_features: List[Tuple[str, float]],
    resolved_tickers: Dict[str, str],
    reference_date: str,
    prediction: float,
    current_ma60: float,
    predicted_future_ma60: float,
    prediction_strategy_label: str,
    period: str,
    smoothing_window: int,
    target_window: int,
) -> str:
    trend = "상승 추세 예상" if prediction > 0 else "하락 추세 예상"

    lines = [
        "[데이터 출처 및 입력 지표]",
        f"- 수집 기간: 최근 {period}",
        "- 수집 소스: yfinance",
    ]

    for key, val in resolved_tickers.items():
        lines.append(f"- {key}: {val}")

    lines.extend(
        [
            "",
            "[모델 및 타겟 정의]",
            "- 모델: LightGBM Regressor",
            f"- 스무딩: {smoothing_window}거래일 이동평균",
            f"- 예측 타겟: 향후 {target_window}거래일 뒤 Gold {smoothing_window}일 이동평균의 현재 대비 변화율(%)",
            f"- 검증 방식: TimeSeriesSplit 5-fold (gap={CV_GAP})",
            "- 확장 내용: 기존 거시지표 + GRU 코드의 연관자산/비율/모멘텀/달력 feature 통합",
            "",
            f"[모델 학습 피처 수] {len(feature_names)}개",
            "",
            "[검증 결과 요약]",
            f"평균 MAE: {mean_mae:.4f}",
            f"평균 RMSE: {mean_rmse:.4f}",
            f"평균 R2: {mean_r2:.4f}",
            f"평균 방향성 적중률: {mean_directional_accuracy:.2f}%",
            "",
            "[Baseline Comparison]",
        ]
    )

    lines.append(
        f"{model_summary['label']}: "
        f"MAE={model_summary['mean_mae']:.4f}, "
        f"RMSE={model_summary['mean_rmse']:.4f}, "
        f"R2={model_summary['mean_r2']:.4f}, "
        f"DirAcc={model_summary['mean_directional_accuracy']:.2f}%"
    )
    for baseline in baseline_summary:
        lines.append(
            f"{baseline['label']}: "
            f"MAE={baseline['mean_mae']:.4f}, "
            f"RMSE={baseline['mean_rmse']:.4f}, "
            f"R2={baseline['mean_r2']:.4f}, "
            f"DirAcc={baseline['mean_directional_accuracy']:.2f}%"
        )

    lines.extend(
        [
            "",
            "[예측 방식 선택]",
            f"- 교차검증 기준 최종 채택 방식: {best_strategy_label}",
            f"- 채택 방식 CV MAE: {best_strategy_mae:.4f}",
            (
                "- 참고: LightGBM이 baseline model보다 좋아야만 채택되는데, 조건을 통과하지 못해 baseline을 사용"
                if best_strategy_label != "LightGBM Model"
                else "- 참고: LightGBM이 baseline model보다 좋아서 최신 예측에 모델을 사용"
            ),
            "",
            "[신뢰도 판단]",
            f"- 신뢰도: {reliability_level}",
            f"- 판단 근거: {reliability_comment}",
            "",
            "[최근 1년 백테스트]",
            (
                f"- 구간: {recent_backtest['start_date']} ~ {recent_backtest['end_date']} "
                f"({recent_backtest['window_size']}개 샘플)"
            ),
        ]
    )

    for item in recent_backtest["strategy_metrics"]:
        lines.append(
            f"{item['label']}: "
            f"MAE={item['mae']:.4f}, "
            f"RMSE={item['rmse']:.4f}, "
            f"R2={item['r2']:.4f}, "
            f"DirAcc={item['directional_accuracy']:.2f}%"
        )

    lines.extend(
        [
            "",
            "[주요 피처 중요도 Top 20]",
        ]
    )

    for feature, score in top_features:
        lines.append(f"{feature}: {score:.6f}")

    lines.extend(
        [
            "",
            "[최신 시점 예측 결과]",
            f"- 기준일: {reference_date}",
            f"- 사용 방식: {prediction_strategy_label}",
            f"- 현재 MA60: {current_ma60:.2f}",
            f"- 예측 미래 MA60: {predicted_future_ma60:.2f}",
            f"- 예측 변화율: {prediction:.2f}%",
            f"- 결론: {trend}",
        ]
    )

    return "\n".join(lines)


# ============================================================
# DB 저장
# ============================================================
def save_report_to_postgres(ml_report: str, gru_report: str = "") -> None:
    if not DATABASE_URL:
        print("DATABASE_URL이 없어서 DB 저장은 건너뜁니다.")
        return

    print(f"{TABLE_NAME} 테이블에 예측 리포트를 저장합니다.")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(CREATE_UPDATED_AT_FUNCTION_SQL)
                cur.execute(CREATE_UPDATED_AT_TRIGGER_SQL)
                cur.execute(
                    f"INSERT INTO {TABLE_NAME} (ml, gru) VALUES (%s, %s)",
                    (ml_report, gru_report),
                )
        print("DB 저장 완료.")
    finally:
        conn.close()


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("메인 프로세스 시작 (LGBM + GRU feature 통합 버전)")

    df_raw, resolved_tickers = fetch_gold_data(period=DATA_PERIOD)
    if df_raw is None:
        print("데이터 수집 실패")
        raise SystemExit(1)

    df_smoothed = preprocess_and_smooth(df_raw, window=SMOOTHING_WINDOW)
    df_features = engineer_features(df_raw, df_smoothed)
    series_target = create_target(df_smoothed, window=TARGET_WINDOW)

    df_final = pd.concat([df_features, series_target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    print("\n최종 데이터셋 준비 완료:", df_final.shape)

    training_result = train_and_evaluate_model(df_final)

    prediction, reference_date, current_ma60, predicted_future_ma60, prediction_strategy_label = predict_future_trend(
        training_result["model"],
        df_features=df_features,
        df_raw=df_raw,
        feature_names=training_result["feature_names"],
        strategy_key=training_result["best_strategy_key"],
        strategy_label=training_result["best_strategy_label"],
        baseline_feature_name=training_result["baseline_feature_name"],
        target_window=TARGET_WINDOW,
        smoothing_window=SMOOTHING_WINDOW,
    )

    ml_report = build_report_text_v2(
        feature_names=training_result["feature_names"],
        mean_mae=training_result["mean_mae"],
        mean_rmse=training_result["mean_rmse"],
        mean_r2=training_result["mean_r2"],
        mean_directional_accuracy=training_result["mean_directional_accuracy"],
        model_summary=training_result["model_summary"],
        baseline_summary=training_result["baseline_summary"],
        best_strategy_label=training_result["best_strategy_label"],
        best_strategy_mae=training_result["best_strategy_mae"],
        recent_backtest=training_result["recent_backtest"],
        reliability_level=training_result["reliability_level"],
        reliability_comment=training_result["reliability_comment"],
        top_features=training_result["top_features"],
        resolved_tickers=resolved_tickers,
        reference_date=reference_date,
        prediction=prediction,
        current_ma60=current_ma60,
        predicted_future_ma60=predicted_future_ma60,
        prediction_strategy_label=prediction_strategy_label,
        period=DATA_PERIOD,
        smoothing_window=SMOOTHING_WINDOW,
        target_window=TARGET_WINDOW,
    )

    print("\n[리포트 미리보기]")
    print(ml_report)

    with open("gold_prediction_report_extended.txt", "w", encoding="utf-8") as f:
        f.write(ml_report)
    print("리포트 저장 완료: gold_prediction_report_extended.txt")

    save_report_to_postgres(ml_report=ml_report, gru_report="")
