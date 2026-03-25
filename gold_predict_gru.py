import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# 1. 데이터 수집 (yfinance)
def fetch_gold_data():
    print("금 가격 데이터를 수집합니다...")
    gold = yf.download('GC=F', period='10y')
    
    if isinstance(gold.columns, pd.MultiIndex):
        gold = gold['Close'].iloc[:, 0].to_frame(name='Close')
    else:
        gold = gold[['Close']]
    
    gold.index.name = 'Date'
    return gold

# 2. 전처리 (이동평균 및 타겟 생성)
def preprocess_data(df, ma_window=60, horizon=60):
    print(f"{ma_window}일 이동평균 및 {horizon}일 후 타겟 수익률을 생성합니다...")
    # 이동평균 추가
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
    
    # 1. 이격도 (현재가 / 이동평균 - 1) * 100
    df['Disparity'] = (df['Close'] / df['MA'] - 1) * 100
    # 2. 이동평균 모멘텀 (현재 MA / 이전 MA - 1) * 100
    df['MA_Momentum'] = (df['MA'] / df['MA'].shift(1) - 1) * 100
    
    # 추가 feature: 수익률 4종 생성
    df['Return_1d']  = (df['Close'] / df['Close'].shift(1) - 1) * 100
    df['Return_5d']  = (df['Close'] / df['Close'].shift(5) - 1) * 100
    df['Return_20d'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
    df['Return_60d'] = (df['Close'] / df['Close'].shift(60) - 1) * 100
    
    # [주의] 데이터 누수(Data Leakage) 지점 1:
    # 타겟 설정 시 horizon(60일) 후의 데이터를 가져옵니다.
    # 즉, t시점의 y값은 t+60 시점의 정보를 포함하고 있습니다.
    # 따라서 학습 데이터의 마지막 시점 t_last와 검증 데이터의 시작 시점 t_start 사이에 
    # 최소 horizon만큼의 간격(Gap)이 없으면, 검증 데이터의 피처가 학습 데이터의 타겟에 쓰인 정보를 미리 알게 됩니다.
    df['Target_Return'] = (df['MA'].shift(-horizon) / df['MA'] - 1) * 100
    
    # 결측치 제거
    df = df.dropna()
    return df

# 3. 데이터 시퀀스 변환 (스케일링 없음)
def prepare_sequences(df, seq_len=120):
    print(f"데이터 시퀀스(길이: {seq_len})를 생성합니다 (스케일러 사용 안 함)...")
    # 추가 feature: 피처 목록을 2개에서 6개로 변경
    feature_cols = ['Disparity', 'MA_Momentum', 'Return_1d', 'Return_5d', 'Return_20d', 'Return_60d']
    features = df[feature_cols].values
    target = df['Target_Return'].values.reshape(-1, 1)
    
    # 시퀀스 생성
    X, y = [], []
    for i in range(len(features) - seq_len):
        # [주의] 데이터 누수 지점 2:
        # X[i]는 i ~ i+seq_len-1 범위의 피처를 사용합니다.
        # y[i]는 i+seq_len 시점의 타겟(즉, i+seq_len+horizon까지의 정보)을 사용합니다.
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len])
        
    return np.array(X), np.array(y)

# 4. GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    # 데이터 처리 관련 설정
    ma_window = 60    # 이동평균 윈도우
    seq_len = 120      # 입력 시퀀스 길이
    horizon = 60      # 예측 거리 (60일 후)
    
    # 데이터 수집 및 처리
    df = fetch_gold_data()
    df = preprocess_data(df, ma_window=ma_window, horizon=horizon)
    
    X, y = prepare_sequences(df, seq_len=seq_len)
    
    # 모델 하이퍼파라미터
    # 추가 feature: input_dim을 데이터프레임의 피처 수에 맞춰 자동으로 설정 (여기선 6)
    input_dim = X.shape[2] 
    hidden_dim = 64
    layer_dim = 2
    output_dim = 1
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # TimeSeriesSplit 설정 (gap 추가로 데이터 누수 방지)
    # gap=horizon(60)을 설정하여 train 종료점과 test 시작점 사이에 미래 정보 중첩을 차단합니다.
    effective_gap = seq_len + horizon - 1
    tscv = TimeSeriesSplit(n_splits=5, gap=effective_gap)
    mae_list = []
    
    print(f"\n수익률 예측 학습 및 검증을 시작합니다 (TimeSeriesSplit with gap={horizon})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 전체 예측 저장용
    all_actual = []
    all_predicted = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1} 학습 중 (Train: {len(train_index)}건, Test: {len(test_index)}건)...")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # [주의] 데이터 누수 지점 3: 스케일링
        # Scaler는 오직 Train 데이터에만 fit해야 합니다. Validation 정보가 스케일링에 섞여선 안 됩니다.
        scaler = StandardScaler()
        
        # 3D 데이터를 2D로 펼쳐서 스케일링 (batch * seq_len, features)
        N_train, L, F = X_train.shape
        X_train_reshaped = X_train.reshape(-1, F)
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N_train, L, F)
        
        N_test, L, F = X_test.shape
        X_test_reshaped = X_test.reshape(-1, F)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(N_test, L, F)
        
        # DataLoader 생성
        train_data = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
        test_data = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 학습
        model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # 모델 검증
        model.eval()
        fold_preds = []
        fold_actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                fold_preds.extend(outputs.cpu().numpy())
                fold_actuals.extend(batch_y.numpy())
        
        # 역변환 과정 생략 (이미 % 단위)
        fold_preds = np.array(fold_preds)
        fold_actuals = np.array(fold_actuals)
        
        mae = mean_absolute_error(fold_actuals, fold_preds)
        mae_list.append(mae)
        print(f"Fold {fold+1} MAE (Returns): {mae:.4f}%")
        
        # 마지막 폴드의 예측값 저장 (시각화용)
        if fold == 4:
            all_actual = fold_actuals
            all_predicted = fold_preds
    
    if len(mae_list) > 0:
        print(f"\n평균 MAE (수익률): {np.mean(mae_list):.4f}%")
    
    # 시각화 (마지막 Fold 기준)
    if len(all_actual) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(all_actual, label='Actual MA Return (%)', color='gray', alpha=0.6)
        plt.plot(all_predicted, label='Predicted MA Return (%)', color='blue')
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title(f'Gold Price MA Return Prediction (GRU) - Last Fold (MAE: {np.mean(mae_list):.4f}%)')
        plt.xlabel('Days')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('gold_gru_return_prediction.png')
        print("예측 결과 그래프가 'gold_gru_return_prediction.png'로 저장되었습니다.")
        
        # CSV 저장 (마지막 Fold 예측)
        res_df = pd.DataFrame({'Actual_Return': all_actual.flatten(), 'Predicted_Return': all_predicted.flatten()})
        res_df.to_csv('gold_gru_return_result.csv', index=False)
        print("예측 결과가 'gold_gru_return_result.csv'로 저장되었습니다.")
