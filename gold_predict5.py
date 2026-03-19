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
    RSI (상대강도지수) 등 기술적 지표를 추가합니다.
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
    
    # 4. RSI (Relative Strength Index) - [신규 추가]
    # 금값이 과매수/과매도 상태인지 판단 (14일 기준)
    print("Gold RSI 지표를 계산합니다...")
    window_rsi = 14
    delta = df_raw_aligned['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / loss
    df_features['Gold_RSI'] = 100 - (100 / (1 + rs))
    
    # 5. 변동성 (Volatility)
    for col in df_raw.columns:
        vol = df_raw[col].pct_change().rolling(window=60).std()
        df_features[f'{col}_Volatility'] = vol.loc[common_index]
    
    # 6. Lag 데이터
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
        plt.title(f'Gold Price Return Prediction v5 (RSI included, MAE: {np.mean(mae_list):.2f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_result_v5.png')
        print("예측 결과 그래프 저장 완료: prediction_result_v5.png")
    except Exception as e:
        print(f"시각화 중 에러 발생: {e}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[주요 피처 중요도 (Top 10)]")
    print(importances.head(10))
    
    return model

if __name__ == "__main__":
    print("메인 프로세스(v5 - RSI) 시작...")
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
        print("\ngold_predict5.py 구현 및 검증이 완료되었습니다.")
    else:
        print("데이터 수집에 실패하였습니다.")
