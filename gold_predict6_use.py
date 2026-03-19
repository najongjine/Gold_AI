import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

def fetch_gold_data():
    """
    기획서의 2단계: 데이터 수집 (yfinance)
    글로벌 금값에 영향을 주는 지표들을 수집합니다.
    """
    print("데이터 수집을 시작합니다...")
    
    # 지표 설정
    tickers = {
        'Gold': 'GC=F',           # 금 선물
        'Dollar_Index': 'UUP',      # 달러 지수 프록시 (UUP ETF)
        'US10Y_Treasury': '^TNX',   # 미국채 10년물 (명목 금리)
        'TIPS_ETF': 'TIP',          # 물가연동채(TIPS) ETF (실질 금리/인플레이션 기대치 프록시)
        'VIX': '^VIX',              # 공포 지수
        'S&P500': '^GSPC'          # S&P 500
    }
    
    df_list = []
    
    # 데이터 수집 (최근 10년)
    for name, ticker in tickers.items():
        print(f"{name} ({ticker}) 데이터를 가져오는 중...")
        try:
            data = yf.download(ticker, period="10y")
            
            if data.empty:
                print(f"경고: {name} ({ticker}) 데이터가 비어있습니다.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    close_data = data['Close'].iloc[:, 0].to_frame(name=name)
                else:
                    print(f"경고: {name} 데이터에서 'Close' 컬럼을 찾을 수 없습니다.")
                    continue
            else:
                close_data = data[['Close']].rename(columns={'Close': name})
            
            df_list.append(close_data)
        except Exception as e:
            print(f"에러 발생 ({name}): {e}")
            
    if not df_list:
        print("수집된 데이터가 없습니다.")
        return None

    # 데이터 병합 (날짜 기준)
    df = pd.concat(df_list, axis=1)
    df.index.name = 'Date'
    df = df.ffill().bfill()
    
    print("데이터 수집 및 병합 완료.")
    print(f"데이터 크기: {df.shape}")
    
    # 결과 저장
    output_file = 'gold_data_raw.csv'
    df.to_csv(output_file)
    print(f"파일 저장 완료: {output_file}")
    
    return df

def preprocess_and_smooth(df, window=60):
    """
    기획서의 3단계: 데이터 전처리 및 스무딩 (Smoothing)
    """
    print(f"데이터 전처리 및 스무딩({window}일 이동평균)을 시작합니다...")
    df_smoothed = df.rolling(window=window).mean()
    df_smoothed = df_smoothed.dropna()
    
    print(f"스무딩 완료. 데이터 크기: {df_smoothed.shape}")
    
    # 결과 저장
    output_file = 'gold_data_smoothed.csv'
    df_smoothed.to_csv(output_file)
    print(f"스무딩된 데이터 저장 완료: {output_file}")
    
    return df_smoothed

def engineer_features(df_raw, df_smoothed):
    """
    기획서의 4단계: 파생변수 생성 (Feature Engineering)
    RSI, MACD 등 기술적 지표를 추가합니다.
    """
    print("파생변수 생성을 시작합니다...")
    
    common_index = df_smoothed.index
    df_features = pd.DataFrame(index=common_index)
    
    # 1. 수익률 (Returns)
    for col in df_raw.columns:
        df_features[f'{col}_Returns'] = df_raw[col].pct_change().loc[common_index] * 100
    
    # 2. 이격도 (Disparity)
    df_raw_aligned = df_raw.loc[common_index]
    for col in df_raw.columns:
        df_features[f'{col}_Disparity'] = (df_raw_aligned[col] / df_smoothed[col] - 1) * 100
    
    # 3. 자산 간 상대 지표 (Ratios)
    df_features['Gold_to_SP500_Ratio'] = df_raw_aligned['Gold'] / df_raw_aligned['S&P500']
    df_features['Gold_to_Dollar_Ratio'] = df_raw_aligned['Gold'] / df_raw_aligned['Dollar_Index']
    
    # 실질금리(Real Interest Rate) Proxy
    df_features['Real_Rate_Proxy'] = df_raw_aligned['US10Y_Treasury'] - df_raw_aligned['TIPS_ETF'].pct_change(60) * 100
    
    # 4. RSI (Relative Strength Index)
    # 금값이 과매수/과매도 상태인지 판단 (14일 기준)
    print("Gold RSI 지표를 계산합니다...")
    window_rsi = 14
    delta = df_raw_aligned['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / loss
    df_features['Gold_RSI'] = 100 - (100 / (1 + rs))
    
    # 5. MACD (Moving Average Convergence Divergence) - [신규 추가]
    # 추세의 강도와 방향 전환 포착
    print("Gold MACD 지표를 계산합니다...")
    ema12 = df_raw_aligned['Gold'].ewm(span=12, adjust=False).mean()
    ema26 = df_raw_aligned['Gold'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    df_features['Gold_MACD_Line'] = macd_line
    df_features['Gold_MACD_Signal'] = signal_line
    df_features['Gold_MACD_Hist'] = macd_hist
    
    # 6. 변동성 (Volatility)
    for col in df_raw.columns:
        vol = df_raw[col].pct_change().rolling(window=60).std()
        df_features[f'{col}_Volatility'] = vol.loc[common_index]
    
    # 7. Lag 데이터
    for col in ['Gold', 'Dollar_Index']:
        for lag in [1, 7, 15]:
            df_features[f'{col}_Returns_Lag_{lag}'] = df_features[f'{col}_Returns'].shift(lag)

    df_features = df_features.ffill().bfill()
    
    print(f"파생변수 생성 완료. 데이터 크기: {df_features.shape}")
    print(f"생성된 컬럼 수: {len(df_features.columns)}")
    
    # 결과 저장
    output_file = 'gold_features.csv'
    df_features.to_csv(output_file)
    print(f"파생변수 데이터 저장 완료: {output_file}")
    
    return df_features

def create_target(df_smoothed, window=60):
    """
    기획서의 5단계: 타겟 설정 (Labeling)
    """
    print(f"타겟 변수 생성(60일 후 이평선 대비 변화율)을 시작합니다...")
    future_ma = df_smoothed['Gold'].shift(-window)
    target = (future_ma / df_smoothed['Gold'] - 1) * 100
    target.name = 'Target_Return'
    return target

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np

def train_and_evaluate_model(df_final):
    """
    기획서의 6단계: 모델링 전략
    """
    print("모델 학습 및 검증을 시작합니다...")
    
    X = df_final.drop(columns=['Target_Return'])
    y = df_final['Target_Return']
    
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
        importance_type='gain',
        verbose=-1
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
        acc = np.mean(correct_direction) * 100
        accuracy_list.append(acc)
        all_preds[test_index] = preds
        all_test_idx.extend(test_index)
        
    print("\n[검증 결과 요약]")
    print(f"평균 MAE: {np.mean(mae_list):.4f}")
    print(f"평균 방향성 적중률: {np.mean(accuracy_list):.2f}%")
    
    # 시각화
    try:
        plt.figure(figsize=(12, 6))
        test_y = y.iloc[all_test_idx]
        test_preds = all_preds[all_test_idx]
        plt.plot(y.index[all_test_idx], test_y, label='Actual Return', color='gray', alpha=0.5)
        plt.plot(y.index[all_test_idx], test_preds, label='Predicted Return', color='blue', linewidth=1.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Gold Price Return Prediction v6 (MACD included, MAE: {np.mean(mae_list):.2f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_result_v6.png')
        print("예측 결과 그래프 저장 완료: prediction_result_v6.png")
    except Exception as e:
        print(f"시각화 중 에러 발생: {e}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[주요 피처 중요도 (Top 10)]")
    print(importances.head(10))
    
    return model

import matplotlib.pyplot as plt

def predict_future_trend(model, df_features, cols_to_drop):
    """
    학습된 모델을 사용하여 현재 시점 기준 향후 금값 추세를 예측합니다.
    """
    print("\n" + "="*50)
    print("🔮 현재 시점 기준 향후 금값 추세 예측 🔮")
    print("="*50)
    
    # 1. 가장 최근 데이터 추출 (오늘 기준 가장 마지막 행)
    # df_features는 정답(Target) 생성과 무관하게 오늘까지의 지표를 모두 가지고 있습니다.
    recent_data = df_features.iloc[[-1]].copy()
    recent_date = recent_data.index[0].strftime('%Y-%m-%d')
    
    # 2. 학습 모델과 동일한 조건으로 컬럼(절대 가격 등) 제거
    recent_data = recent_data.drop(columns=[c for c in cols_to_drop if c in recent_data.columns])
    
    # 3. 예측 수행
    prediction = model.predict(recent_data)[0]
    
    print(f"▶ 기준일 (가장 최근 거래일): {recent_date}")
    print(f"▶ 예측 결과: 향후 60거래일 동안 금값 60일 이동평균선은 현재 대비 약 {prediction:.2f}% 변화할 것으로 모델은 예상합니다.")
    print("-" * 50)
    
    if prediction > 0:
        print(f"👉 결론: 상승 추세 (Uptrend) 지속/전환 예상 📈 (예상 수익률: +{prediction:.2f}%)")
    else:
        print(f"👉 결론: 하락 추세 (Downtrend) 지속/전환 예상 📉 (예상 수익률: {prediction:.2f}%)")
        
    return prediction

def predict_future_trend_2(model, df_features, cols_to_drop, recent_days=30):
    """
    최근 N일간의 데이터를 바탕으로 모델의 예측 추세를 확인하고 최종 예측을 수행합니다.
    """
    print("\n" + "="*60)
    print(f"🔮 최근 {recent_days}일 추세 기반 향후 금값 예측 🔮")
    print("="*60)
    
    # 1. 최근 N일의 데이터 추출 (추세 확인용)
    recent_data_full = df_features.tail(recent_days).copy()
    dates = recent_data_full.index
    
    # 2. 모델 학습과 동일하게 불필요한 컬럼 제거
    recent_data = recent_data_full.drop(columns=[c for c in cols_to_drop if c in recent_data_full.columns])
    
    # 3. 최근 N일 각각에 대한 향후 60일 예측치 계산
    predictions = model.predict(recent_data)
    
    # 4. 가장 최근(오늘 기준) 예측값
    final_prediction = predictions[-1]
    final_date = dates[-1].strftime('%Y-%m-%d')
    
    print(f"▶ 기준일 (최근 거래일): {final_date}")
    print(f"▶ 모델의 최종 예측: 향후 60거래일 동안 약 {final_prediction:.2f}% 변화 예상")
    
    if final_prediction > 0:
        print(f"👉 결론: 상승 추세(Uptrend) 지속/전환 예상 📈 (예상 수익률: +{final_prediction:.2f}%)")
    else:
        print(f"👉 결론: 하락 추세(Downtrend) 지속/전환 예상 📉 (예상 수익률: {final_prediction:.2f}%)")
        
    # ==========================================
    # 📊 최근 예측 추세 시각화 (질문자님 의도 반영)
    # ==========================================
    try:
        plt.figure(figsize=(10, 5))
        
        # 최근 N일 동안 모델이 예측한 '예상 수익률 추세'
        plt.plot(dates, predictions, marker='o', linestyle='-', color='purple', label='Predicted Return (Future 60 Days)')
        
        # 0% 기준선 (상승/하락 분기점)
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'Model Prediction Trend (Last {recent_days} Days)')
        plt.xlabel('Date')
        plt.ylabel('Predicted Future Return (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = 'recent_prediction_trend.png'
        plt.savefig(plot_filename)
        print(f"\n[!] 최근 {recent_days}일간의 예측 추세 그래프가 저장되었습니다: {plot_filename}")
        print("그래프가 우상향 중이라면 모델이 점점 더 강한 상승을 점치고 있다는 뜻입니다.")
        
    except Exception as e:
        print(f"추세 시각화 중 에러 발생: {e}")

    print("-" * 60)
    return final_prediction

if __name__ == "__main__":
    print("메인 프로세스(v6 - MACD) 시작...")
    df_raw = fetch_gold_data()
    
    if df_raw is not None:
        df_smoothed = preprocess_and_smooth(df_raw)
        df_features = engineer_features(df_raw, df_smoothed)
        series_target = create_target(df_smoothed)
        
        df_final = pd.concat([df_features, series_target], axis=1).dropna()
        
        # 절대 가격 및 TIPS ETF 제거
        cols_to_drop = ['Gold', 'Dollar_Index', 'US10Y_Treasury', 'TIPS_ETF', 'VIX', 'S&P500']
        df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
        
        print(f"최종 데이터셋 준비 완료. 데이터 크기: {df_final.shape}")
        model = train_and_evaluate_model(df_final)
        print("\ngold_predict6.py 구현 및 검증이 완료되었습니다.")

        # 가장 최근 데이터를 이용한 실전 미래 예측!
        predict_future_trend(model, df_features, cols_to_drop)

        #[추세 기반 예측 실행] 최근 n일간의 추세를 바탕으로 예측
        #predict_future_trend_2(model, df_features, cols_to_drop, recent_days=60)
    else:
        print("데이터 수집에 실패하였습니다.")
