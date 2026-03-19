import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

def fetch_gold_data():
    """
    기획서의 2단계: 데이터 수집 (yfinance)
    글로벌 금값에 영향을 주는 5대 핵심 지표를 수집합니다.
    """
    print("데이터 수집을 시작합니다...")
    
    # 지표 설정
    # DX-Y.NYB가 yfinance에서 종종 에러가 나므로, 
    # 달러 인덱스를 추종하는 ETF인 'UUP'를 대안으로 사용합니다.
    tickers = {
        'Gold': 'GC=F',           # 금 선물
        'Dollar_Index': 'UUP',      # 달러 지수 프록시 (UUP ETF)
        'US10Y_Treasury': '^TNX',   # 미국채 10년물
        'VIX': '^VIX',              # 공포 지수
        'S&P500': '^GSPC'          # S&P 500
    }
    
    df_list = []
    
    # 데이터 수집 (최근 10년)
    for name, ticker in tickers.items():
        print(f"{name} ({ticker}) 데이터를 가져오는 중...")
        try:
            # 기간을 10년으로 설정하여 충분한 학습 데이터 확보
            data = yf.download(ticker, period="10y")
            
            if data.empty:
                print(f"경고: {name} ({ticker}) 데이터가 비어있습니다.")
                continue

            # yfinance 1.0.0+ 버전의 MultiIndex 대응 및 Close 컬럼 추출
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    # 특정 티커의 Close 컬럼 선택
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
    
    # 인덱스 이름 정리 및 날짜 형식 변환
    df.index.name = 'Date'
    
    # 결측치 처리: 앞 데이터로 채우고, 그래도 없으면 뒤 데이터로 채움 (전진/후진 채우기)
    df = df.ffill().bfill()
    
    print("데이터 수집 및 병합 완료.")
    print(f"데이터 크기: {df.shape}")
    print(df.tail())
    
    # 결과 저장
    output_file = 'gold_data_raw.csv'
    df.to_csv(output_file)
    print(f"파일 저장 완료: {output_file}")
    
    # 간단한 시각화로 확인
    try:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(df.columns):
            plt.subplot(len(df.columns), 1, i+1)
            # 데이터를 0-1 사이로 정규화하여 흐름만 확인
            norm_series = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            plt.plot(norm_series, label=col)
            plt.title(f"{col} Trend (Normalized)")
            plt.legend()
        plt.tight_layout()
        plt.savefig('data_collection_verify.png')
        print("검증용 그래프 저장 완료: data_collection_verify.png")
    except Exception as e:
        print(f"그래프 생성 중 에러 발생: {e}")
    
    return df

def preprocess_and_smooth(df, window=60):
    """
    기획서의 3단계: 데이터 전처리 및 스무딩 (Smoothing)
    모든 입력 데이터에 60일 이동평균(MA60)을 적용하여 중기 추세 데이터로 변환하고 노이즈를 제거합니다.
    """
    print(f"데이터 전처리 및 스무딩({window}일 이동평균)을 시작합니다...")
    
    # 60일 이동평균 적용
    df_smoothed = df.rolling(window=window).mean()
    
    # 이동평균 계산으로 인해 발생하는 초기 결측치(NaN) 제거
    # (60일치 데이터가 쌓여야 첫 평균값이 나오므로 앞의 59행은 삭제됨)
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
    이격도, 상대적 수익률, Lag 데이터, 변동성 지표 등을 생성합니다.
    """
    print("파생변수 생성을 시작합니다...")
    
    # 공통 인덱스 설정 (스무딩된 데이터의 인덱스 기준)
    common_index = df_smoothed.index
    df_features = df_smoothed.copy()
    
    # 1. 이격도 (Disparity): 현재가 / 60일 이평선
    # df_raw를 common_index에 맞게 재색인
    df_raw_aligned = df_raw.loc[common_index]
    for col in df_raw.columns:
        df_features[f'{col}_Disparity'] = df_raw_aligned[col] / df_smoothed[col]
    
    # 2. 변동성 (Volatility): 최근 60일간의 표준편차
    for col in df_raw.columns:
        vol = df_raw[col].rolling(window=60).std()
        df_features[f'{col}_Volatility'] = vol.loc[common_index]
    
    # 3. Lag 데이터: 1일 전, 7일 전, 15일 전의 지표 상태
    # (주의: raw 데이터에서 shift를 해야 현재 시점 기준 과거 데이터를 정확히 가져옴)
    for col in ['Gold', 'Dollar_Index']:
        for lag in [1, 7, 15]:
            df_features[f'{col}_Lag_{lag}'] = df_raw[col].shift(lag).loc[common_index]
            
    # 4. 수익률 관련 (기초 데이터의 변화율)
    for col in df_raw.columns:
        df_features[f'{col}_PctChange'] = df_raw[col].pct_change().loc[common_index]

    # 결측치 처리 (Lag 등으로 인해 발생할 수 있는 NaN 채움)
    df_features = df_features.ffill().bfill()
    
    print(f"파생변수 생성 완료. 데이터 크기: {df_features.shape}")
    print(f"생성된 컬럼: {df_features.columns.tolist()}")
    
    # 결과 저장
    output_file = 'gold_features.csv'
    df_features.to_csv(output_file)
    print(f"파생변수 데이터 저장 완료: {output_file}")
    
    return df_features

    return df_features

def create_target(df_smoothed, window=60):
    """
    기획서의 5단계: 타겟 설정 (Labeling)
    현재 시점(t)의 60일 이동평균 대비 60일 후(t+60)의 이동평균의 변화율(%)을 타겟으로 설정합니다.
    """
    print(f"타겟 변수 생성(60일 후 이평선 대비 변화율)을 시작합니다...")
    
    # 1. 60일 후의 이평선 값을 현재 시점으로 가져옴
    future_ma = df_smoothed['Gold'].shift(-window)
    
    # 2. 현재 이평선 대비 변화율(%) 계산
    # 타겟 = (미래이평 / 현재이평 - 1) * 100
    target = (future_ma / df_smoothed['Gold'] - 1) * 100
    
    # 이름 설정
    target.name = 'Target_Return'
    
    return target

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np

def train_and_evaluate_model(df_final):
    """
    기획서의 6단계: 모델링 전략
    LightGBM 알고리즘과 TimeSeriesSplit을 사용하여 학습 및 검증을 진행합니다.
    """
    print("모델 학습 및 검증을 시작합니다...")
    
    # 1. Feature와 Target 분리
    X = df_final.drop(columns=['Target_Return'])
    y = df_final['Target_Return']
    
    # 2. TimeSeriesSplit 설정 (5개 폴드)
    tscv = TimeSeriesSplit(n_splits=5)
    
    mae_list = []
    accuracy_list = []
    
    # 모델 정의
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        importance_type='gain',
        verbose=-1
    )
    
    print("교차 검증 진행 중...")
    
    # 폴드별 데이터 예측 저장을 위한 배열
    all_preds = np.zeros(len(y))
    all_test_idx = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 모델 학습
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[yf.utils.get_callback("print_every", 100)] if hasattr(yf.utils, "get_callback") else [] # Dummy callback handling
        )
        
        # 예측
        preds = model.predict(X_test)
        
        # MAE 계산
        mae = mean_absolute_error(y_test, preds)
        mae_list.append(mae)
        
        # 방향성 적중률 (Directional Accuracy)
        # 실제 수익률과 예측 수익률의 부호가 같은 비율
        correct_direction = np.sign(y_test.values) == np.sign(preds)
        acc = np.mean(correct_direction) * 100
        accuracy_list.append(acc)
        
        all_preds[test_index] = preds
        all_test_idx.extend(test_index)
        
    print("\n[검증 결과 요약]")
    print(f"평균 MAE: {np.mean(mae_list):.4f}")
    print(f"평균 방향성 적중률: {np.mean(accuracy_list):.2f}%")
    
    # 3. 전체 데이터 시각화 (마지막 테스트 구간 중심)
    try:
        plt.figure(figsize=(12, 6))
        # 실제값 (검증 구간만)
        test_y = y.iloc[all_test_idx]
        test_preds = all_preds[all_test_idx]
        
        plt.plot(y.index[all_test_idx], test_y, label='Actual Return', color='gray', alpha=0.5)
        plt.plot(y.index[all_test_idx], test_preds, label='Predicted Return', color='blue', linewidth=1.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Gold Price Return Prediction (MAE: {np.mean(mae_list):.2f}, Acc: {np.mean(accuracy_list):.1f}%)')
        plt.xlabel('Date')
        plt.ylabel('Expected Return (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prediction_result.png')
        print("예측 결과 그래프 저장 완료: prediction_result.png")
    except Exception as e:
        print(f"시각화 중 에러 발생: {e}")

    # 피처 중요도 출력
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[주요 피처 중요도 (Top 5)]")
    print(importances.head(5))
    
    return model

if __name__ == "__main__":
    print("메인 프로세스 시작...")
    # 1. 데이터 수집
    df_raw = fetch_gold_data()
    
    if df_raw is not None:
        print("데이터 수집 성공. 다음 단계 진행...")
        # 2. 데이터 스무딩
        df_smoothed = preprocess_and_smooth(df_raw)
        
        # 3. 파생변수 생성
        df_features = engineer_features(df_raw, df_smoothed)
        
        # 4. 타겟 생성
        series_target = create_target(df_smoothed)
        
        # 5. 데이터 합치기 (Feature + Target)
        df_final = pd.concat([df_features, series_target], axis=1).dropna()
        print(f"최종 데이터셋 준비 완료. 데이터 크기: {df_final.shape}")
        
        # 6. 모델 학습 및 평가
        model = train_and_evaluate_model(df_final)
        
        print("\n모든 기획 단계 구현 및 검증이 완료되었습니다.")
    else:
        print("데이터 수집에 실패하여 프로세스를 중단합니다.")
