import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_p2dBsFZ9Mvfx@ep-divine-sea-ahijybu5-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
TABLE_NAME = "t_gold_model"
DATA_PERIOD = "10y"
SMOOTHING_WINDOW = 60
TARGET_WINDOW = 60

TICKER_CANDIDATES = {
    "Gold": ["GC=F"],
    "Dollar_Index": ["UUP"],
    "US10Y_Treasury": ["^TNX"],
    "TIPS_ETF": ["TIP"],
    "VIX": ["^VIX"],
    "S&P500": ["^GSPC"],
}

YF_CACHE_DIR = os.path.join(os.getcwd(), ".yfinance_cache")
os.makedirs(YF_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YF_CACHE_DIR)


def fetch_gold_data(period: str = DATA_PERIOD) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    print("데이터 수집을 시작합니다.")

    tickers = {name: candidates[0] for name, candidates in TICKER_CANDIDATES.items()}
    resolved_tickers: Dict[str, str] = {}
    df_list = []

    for name, ticker in tickers.items():
        print(f"{name} ({ticker}) 데이터를 가져오는 중...")
        try:
            data = yf.download(ticker, period=period, auto_adjust=False, progress=False)

            if data.empty:
                print(f"경고: {name} ({ticker}) 데이터가 비어 있습니다.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                if "Close" not in data.columns.levels[0]:
                    print(f"경고: {name} 데이터에서 'Close' 컬럼을 찾을 수 없습니다.")
                    continue
                close_data = data["Close"].iloc[:, 0].to_frame(name=name)
            else:
                close_data = data[["Close"]].rename(columns={"Close": name})

            resolved_tickers[name] = ticker
            df_list.append(close_data)
        except Exception as exc:
            print(f"에러 발생 ({name}): {exc}")

    if not df_list:
        print("수집된 데이터가 없습니다.")
        return None, resolved_tickers

    df = pd.concat(df_list, axis=1)
    df.index.name = "Date"
    df = df.ffill().bfill()

    print("데이터 수집 및 병합 완료.")
    print(f"데이터 크기: {df.shape}")

    output_file = "gold_data_raw.csv"
    df.to_csv(output_file)
    print(f"파일 저장 완료: {output_file}")

    return df, resolved_tickers


def preprocess_and_smooth(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    print(f"데이터 전처리 및 스무딩({window}일 이동평균)을 시작합니다.")
    df_smoothed = df.rolling(window=window).mean().dropna()

    print(f"스무딩 완료. 데이터 크기: {df_smoothed.shape}")

    output_file = "gold_data_smoothed.csv"
    df_smoothed.to_csv(output_file)
    print(f"스무딩 데이터 저장 완료: {output_file}")

    return df_smoothed


def engineer_features(df_raw: pd.DataFrame, df_smoothed: pd.DataFrame) -> pd.DataFrame:
    print("파생변수 생성을 시작합니다.")

    common_index = df_smoothed.index
    df_features = pd.DataFrame(index=common_index)
    df_raw_aligned = df_raw.loc[common_index]

    for col in df_raw.columns:
        df_features[f"{col}_Returns"] = df_raw[col].pct_change().loc[common_index] * 100

    for col in df_raw.columns:
        df_features[f"{col}_Disparity"] = (df_raw_aligned[col] / df_smoothed[col] - 1) * 100

    df_features["Gold_to_SP500_Ratio"] = df_raw_aligned["Gold"] / df_raw_aligned["S&P500"]
    df_features["Gold_to_Dollar_Ratio"] = df_raw_aligned["Gold"] / df_raw_aligned["Dollar_Index"]
    df_features["Real_Rate_Proxy"] = (
        df_raw_aligned["US10Y_Treasury"] - df_raw_aligned["TIPS_ETF"].pct_change(60) * 100
    )

    print("Gold RSI 지표를 계산합니다.")
    window_rsi = 14
    delta = df_raw_aligned["Gold"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / loss
    df_features["Gold_RSI"] = 100 - (100 / (1 + rs))

    print("Gold MACD 지표를 계산합니다.")
    ema12 = df_raw_aligned["Gold"].ewm(span=12, adjust=False).mean()
    ema26 = df_raw_aligned["Gold"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df_features["Gold_MACD_Line"] = macd_line
    df_features["Gold_MACD_Signal"] = signal_line
    df_features["Gold_MACD_Hist"] = macd_line - signal_line

    for col in df_raw.columns:
        vol = df_raw[col].pct_change().rolling(window=60).std()
        df_features[f"{col}_Volatility"] = vol.loc[common_index]

    for col in ["Gold", "Dollar_Index"]:
        for lag in [1, 7, 15]:
            df_features[f"{col}_Returns_Lag_{lag}"] = df_features[f"{col}_Returns"].shift(lag)

    df_features = df_features.ffill().bfill()

    print(f"파생변수 생성 완료. 데이터 크기: {df_features.shape}")
    print(f"생성된 컬럼 수: {len(df_features.columns)}")

    output_file = "gold_features.csv"
    df_features.to_csv(output_file)
    print(f"파생변수 데이터 저장 완료: {output_file}")

    return df_features


def create_target(df_smoothed: pd.DataFrame, window: int = TARGET_WINDOW) -> pd.Series:
    print(f"타겟 변수 생성({window}거래일 뒤 이동평균 대비 변화율)을 시작합니다.")
    future_ma = df_smoothed["Gold"].shift(-window)
    target = (future_ma / df_smoothed["Gold"] - 1) * 100
    target.name = "Target_Return"
    return target


def train_and_evaluate_model(df_final: pd.DataFrame) -> Dict[str, object]:
    print("모델 학습 및 검증을 시작합니다.")

    X = df_final.drop(columns=["Target_Return"])
    y = df_final["Target_Return"]

    print(f"\n[모델 학습 피처 목록 - 총 {len(X.columns)}개]")
    print(X.columns.tolist())

    tscv = TimeSeriesSplit(n_splits=5)
    mae_list = []
    accuracy_list = []

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        importance_type="gain",
        verbose=-1,
    )

    all_preds = np.zeros(len(y))
    all_test_idx = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mae_list.append(mae)
        correct_direction = np.sign(y_test.values) == np.sign(preds)
        accuracy_list.append(np.mean(correct_direction) * 100)
        all_preds[test_index] = preds
        all_test_idx.extend(test_index)

    mean_mae = float(np.mean(mae_list))
    mean_directional_accuracy = float(np.mean(accuracy_list))

    print("\n[검증 결과 요약]")
    print(f"평균 MAE: {mean_mae:.4f}")
    print(f"평균 방향성 적중률: {mean_directional_accuracy:.2f}%")

    try:
        plt.figure(figsize=(12, 6))
        test_y = y.iloc[all_test_idx]
        test_preds = all_preds[all_test_idx]
        plt.plot(y.index[all_test_idx], test_y, label="Actual Return", color="gray", alpha=0.5)
        plt.plot(y.index[all_test_idx], test_preds, label="Predicted Return", color="blue", linewidth=1.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Gold Price Return Prediction v6 (MACD included, MAE: {mean_mae:.2f})")
        plt.legend()
        plt.grid(True)
        plt.savefig("prediction_result_v6.png")
        print("예측 결과 그래프 저장 완료: prediction_result_v6.png")
    except Exception as exc:
        print(f"시각화 중 에러 발생: {exc}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = [(feature, float(score)) for feature, score in importances.head(10).items()]

    print("\n[주요 피처 중요도 (Top 10)]")
    print(importances.head(10))

    return {
        "model": model,
        "feature_names": X.columns.tolist(),
        "mean_mae": mean_mae,
        "mean_directional_accuracy": mean_directional_accuracy,
        "top_features": top_features,
    }


def predict_future_trend(
    model: LGBMRegressor,
    df_features: pd.DataFrame,
    cols_to_drop: List[str],
    target_window: int = TARGET_WINDOW,
    smoothing_window: int = SMOOTHING_WINDOW,
) -> Tuple[float, str]:
    print("\n" + "=" * 50)
    print("현재 시점 기준 향후 금값 추세 예측")
    print("=" * 50)

    recent_data = df_features.iloc[[-1]].copy()
    recent_date = recent_data.index[0].strftime("%Y-%m-%d")
    recent_data = recent_data.drop(columns=[c for c in cols_to_drop if c in recent_data.columns])

    prediction = float(model.predict(recent_data)[0])

    print(f"기준일 (가장 최근 거래일): {recent_date}")
    print(
        f"예측 결과: 향후 {target_window}거래일 동안 금값 {smoothing_window}일 이동평균선은 현재 대비 "
        f"약 {prediction:.2f}% 변화할 것으로 모델은 예상합니다."
    )
    print("-" * 50)

    if prediction > 0:
        print(f"결론: 상승 추세 (Uptrend) 지속/전환 예상 (예상 수익률: +{prediction:.2f}%)")
    else:
        print(f"결론: 하락 추세 (Downtrend) 지속/전환 예상 (예상 수익률: {prediction:.2f}%)")

    return prediction, recent_date


def predict_future_trend_2(
    model: LGBMRegressor,
    df_features: pd.DataFrame,
    cols_to_drop: List[str],
    recent_days: int = 30,
) -> float:
    print("\n" + "=" * 60)
    print(f"최근 {recent_days}일 추세 기반 향후 금값 예측")
    print("=" * 60)

    recent_data_full = df_features.tail(recent_days).copy()
    dates = recent_data_full.index
    recent_data = recent_data_full.drop(columns=[c for c in cols_to_drop if c in recent_data_full.columns])
    predictions = model.predict(recent_data)

    final_prediction = float(predictions[-1])
    final_date = dates[-1].strftime("%Y-%m-%d")

    print(f"기준일 (최근 거래일): {final_date}")
    print(f"모델의 최종 예측: 향후 60거래일 동안 약 {final_prediction:.2f}% 변화 예상")

    if final_prediction > 0:
        print(f"결론: 상승 추세(Uptrend) 지속/전환 예상 (예상 수익률: +{final_prediction:.2f}%)")
    else:
        print(f"결론: 하락 추세(Downtrend) 지속/전환 예상 (예상 수익률: {final_prediction:.2f}%)")

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(dates, predictions, marker="o", linestyle="-", color="purple", label="Predicted Return (Future 60 Days)")
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.title(f"Model Prediction Trend (Last {recent_days} Days)")
        plt.xlabel("Date")
        plt.ylabel("Predicted Future Return (%)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = "recent_prediction_trend.png"
        plt.savefig(plot_filename)
        print(f"\n최근 {recent_days}일간의 예측 추세 그래프가 저장되었습니다: {plot_filename}")
    except Exception as exc:
        print(f"추세 시각화 중 에러 발생: {exc}")

    print("-" * 60)
    return final_prediction


def build_report_text(
    feature_names: List[str],
    mean_mae: float,
    mean_directional_accuracy: float,
    top_features: List[Tuple[str, float]],
    resolved_tickers: Dict[str, str],
    reference_date: str,
    prediction: float,
    period: str,
    smoothing_window: int,
    target_window: int,
) -> str:
    trend = "상승 추세 (Uptrend) 지속/전환 예상" if prediction > 0 else "하락 추세 (Downtrend) 지속/전환 예상"

    source_lines = [
        "[데이터 출처 및 입력 지표]",
        f"- 수집 기간: 최근 {period}",
        "- 수집 소스: yfinance",
        f"- Gold: {resolved_tickers.get('Gold', 'N/A')} (candidates: {TICKER_CANDIDATES['Gold']})",
        f"- Dollar_Index: {resolved_tickers.get('Dollar_Index', 'N/A')}",
        f"- US10Y_Treasury: {resolved_tickers.get('US10Y_Treasury', 'N/A')}",
        f"- TIPS_ETF: {resolved_tickers.get('TIPS_ETF', 'N/A')}",
        f"- VIX: {resolved_tickers.get('VIX', 'N/A')}",
        f"- S&P500: {resolved_tickers.get('S&P500', 'N/A')}",
        "",
        "[모델 및 타겟 정의]",
        "- 모델: LightGBM Regressor",
        f"- 스무딩: {smoothing_window}거래일 이동평균",
        f"- 예측 타겟: 향후 {target_window}거래일 뒤 Gold {smoothing_window}일 이동평균의 현재 대비 변화율(%)",
        "- 검증 방식: TimeSeriesSplit 5-fold",
        "",
        f"[모델 학습 피처 목록 - 총 {len(feature_names)}개]",
        str(feature_names),
        "",
        "[검증 결과 요약]",
        f"평균 MAE: {mean_mae:.4f}",
        f"평균 방향성 적중률: {mean_directional_accuracy:.2f}%",
        "",
        "[주요 피처 중요도 (Top 10)]",
    ]

    for feature, score in top_features:
        source_lines.append(f"{feature}: {score:.6f}")

    source_lines.extend(
        [
            "",
            "gold_predict_model module training and inference completed.",
            "",
            "=" * 50,
            "현재 시점 기준 향후 금값 추세 예측",
            "=" * 50,
            f"기준일 (가장 최근 거래일): {reference_date}",
            f"예측 결과: 향후 {target_window}거래일 동안 금값 {smoothing_window}일 이동평균선은 현재 대비 약 {prediction:.2f}% 변화할 것으로 모델은 예상합니다.",
            f"결론: {trend} (예상 수익률: {prediction:+.2f}%)",
            "",
            "[해석 가이드]",
            "- 이 값은 거시지표, 변동성, 기술적 지표를 함께 넣은 시계열 회귀 모델의 추정치입니다.",
            "- 금융시장은 외생 변수 영향이 크므로, 이 결과는 참고 지표로 해석하는 것이 적절합니다.",
        ]
    )

    return "\n".join(source_lines)


def save_report_to_postgres(ml_report: str, gru_report: str = "") -> None:
    print(f"{TABLE_NAME} 테이블에 예측 리포트를 저장합니다.")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {TABLE_NAME} (ml, gru) VALUES (%s, %s)",
                    (ml_report, gru_report),
                )
        print("DB 저장 완료.")
    finally:
        conn.close()


if __name__ == "__main__":
    print("메인 프로세스(v6 - MACD) 시작...")
    df_raw, resolved_tickers = fetch_gold_data()

    if df_raw is not None:
        df_smoothed = preprocess_and_smooth(df_raw, window=SMOOTHING_WINDOW)
        df_features = engineer_features(df_raw, df_smoothed)
        series_target = create_target(df_smoothed, window=TARGET_WINDOW)

        df_final = pd.concat([df_features, series_target], axis=1).dropna()

        cols_to_drop = ["Gold", "Dollar_Index", "US10Y_Treasury", "TIPS_ETF", "VIX", "S&P500"]
        df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

        print(f"최종 데이터셋 준비 완료. 데이터 크기: {df_final.shape}")
        training_result = train_and_evaluate_model(df_final)
        print("\ngold_predict6_use.py 학습 및 검증이 완료되었습니다.")

        prediction, reference_date = predict_future_trend(
            training_result["model"],
            df_features,
            cols_to_drop,
            target_window=TARGET_WINDOW,
            smoothing_window=SMOOTHING_WINDOW,
        )

        ml_report = build_report_text(
            feature_names=training_result["feature_names"],
            mean_mae=training_result["mean_mae"],
            mean_directional_accuracy=training_result["mean_directional_accuracy"],
            top_features=training_result["top_features"],
            resolved_tickers=resolved_tickers,
            reference_date=reference_date,
            prediction=prediction,
            period=DATA_PERIOD,
            smoothing_window=SMOOTHING_WINDOW,
            target_window=TARGET_WINDOW,
        )

        print("\n[DB 저장용 리포트 미리보기]")
        print(ml_report)
        save_report_to_postgres(ml_report=ml_report, gru_report="")
    else:
        print("데이터 수집에 실패했습니다.")
