import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

def collect_data():
    """
    1단계: 데이터 수집 (yfinance)
    금 선물(Gold Futures, 티커: GC=F) 데이터를 가져옵니다.
    """
    print("Starting Stage 1: Data Collection...")
    
    # 티커 설정: 금 선물
    ticker = "GC=F"
    
    # 기간 설정: 최근 10년 데이터
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)
    
    print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # 데이터 다운로드
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print("Error: No data fetched. Please check the ticker or internet connection.")
            return None
            
        # 날짜 정렬 및 인덱스 확인
        data.sort_index(inplace=True)
        
        # yfinance 최신 버전에서는 단일 티커도 Multi-Index로 반환될 수 있음
        # 컬럼 구조를 단순화 (Ticker 레벨 제거)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 'Close' 가격이 우리가 맞출 Target
        print("Data collection successful.")
        print(f"Data shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        return data
        
    except Exception as e:
        print(f"An error occurred during data collection: {e}")
        return None

def preprocess_data(df):
    """
    2단계: 데이터 전처리
    결측치 처리 및 기본적인 통계 확인
    """
    print("\nStarting Stage 2: Data Preprocessing...")
    
    # 1. 결측치 확인 및 처리 (ffill: 직전 데이터로 채움)
    print(f"Missing values before: \n{df.isnull().sum()}")
    df.fillna(method='ffill', inplace=True)
    # 처음 데이터가 결측치일 경우 bfill도 수행
    df.fillna(method='bfill', inplace=True)
    print(f"Missing values after: \n{df.isnull().sum()}")
    
    # 2. 기초 통계량 확인 (이상치 파악용)
    print("\nSummary Statistics:")
    print(df['Close'].describe())
    
    # 트리 기반 모델(LightGBM)은 스케일링에 덜 민감하지만 관습적으로 추후를 위해 Close_scaled 생성
    scaler = MinMaxScaler()
    # Close 데이터가 2차원 배열 형태여야 하므로 reshape 필요
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])
    
    print("Data preprocessing complete.")
    return df

def visualize_data(df):
    """
    2단계: EDA (시각화)
    가격 추세 및 boxplot 확인
    """
    print("\nGenerating visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 종가 추세 그래프
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Gold Close Price', color='gold')
    plt.title('Gold Price Trend (10 Years)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # 2. Boxplot (이상치 확인용)
    plt.subplot(2, 1, 2)
    sns.boxplot(x=df['Close'], color='orange')
    plt.title('Gold Price Boxplot (Outlier Detection)')
    plt.xlabel('Price (USD)')
    
    plt.tight_layout()
    plt.savefig('gold_price_stage2_eda.png')
    print("EDA visualizations saved to 'gold_price_stage2_eda.png'")
    # plt.show() # 서버 환경 등 확인을 위해 일단 주석 처리 혹은 저장만 수행

def engineer_features(df):
    """
    3단계: 피처 엔지니어링 (변수 생성)
    이동평균, 차분, 날짜 파생 변수 생성
    """
    print("\nStarting Stage 3: Feature Engineering...")
    
    # 1. 이동평균 (MA): 5일, 20일, 60일
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 2. 차분 (Diff): 전날 대비 가격 변화
    df['Diff'] = df['Close'].diff()
    
    # 3. 날짜 파생 변수
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # 변수 생성 후 발생하는 결측치(NaN) 제거 (이동평균 등 앞부분)
    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    
    print(f"Feature engineering complete. Rows before: {before_drop}, Rows after: {after_drop}")
    print(f"New features: {['MA5', 'MA20', 'MA60', 'Diff', 'DayOfWeek', 'Month', 'Quarter']}")
    
    return df

def train_model(df):
    """
    4단계: 모델 학습 (LightGBM)
    데이터를 시간 순서대로 분할하여 학습 및 검증
    """
    print("\nStarting Stage 4: Model Training...")
    
    # 1. 피처와 타겟 설정
    # Close 가격을 맞추는 것이 목표
    features = ['MA5', 'MA20', 'MA60', 'Diff', 'DayOfWeek', 'Month', 'Quarter']
    target = 'Close'
    
    X = df[features]
    y = df[target]
    
    # 2. 데이터 분할 (시간 순서 유지)
    # 시계열 데이터이므로 무작위가 아닌 마지막 20%를 Test용으로 사용
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. 모델 설정 및 학습
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        importance_type='gain'
    )
    
    print("Training LightGBM model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[] # 최신 버전에서는 early_stopping 등을 여기서 처리 가능
    )
    
    print("Model training complete.")
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    5단계: 평가 및 시각화
    모델의 성능을 지표로 확인하고 실제값과 예측값을 비교
    """
    print("\nStarting Stage 5: Evaluation and Visualization...")
    
    # 1. 예측값 생성
    y_pred = model.predict(X_test)
    
    # 2. 평가지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 3. 시각화 (실제값 vs 예측값)
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Gold Price Prediction: Actual vs Predicted (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gold_prediction_result.png')
    print("Prediction chart saved to 'gold_prediction_result.png'")
    
    return y_pred

if __name__ == "__main__":
    gold_df = collect_data()
    if gold_df is not None:
        gold_df = preprocess_data(gold_df)
        # visualize_data(gold_df) # 완료 후에는 필요 시에만 수행
        gold_df = engineer_features(gold_df)
        
        # 모델 학습
        model, X_test, y_test = train_model(gold_df)
        
        # 모델 평가
        y_pred = evaluate_model(model, X_test, y_test)
