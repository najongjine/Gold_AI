import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# 2. 전처리 (60일 이동평균 및 타겟 생성)
def preprocess_data(df, window=60):
    print(f"{window}일 이동평균 및 타겟 변수를 생성합니다...")
    # 60일 이동평균 추가
    df['MA60'] = df['Close'].rolling(window=window).mean()
    
    # 60일 후의 가격을 타겟으로 설정
    df['Target'] = df['Close'].shift(-window)
    
    # 결측치 제거
    df = df.dropna()
    return df

# 3. 데이터 시퀀스 변환 및 스케일링
def prepare_sequences(df, window_size=60):
    print("데이터 시퀀스 및 스케일링을 수행합니다...")
    features = df[['Close', 'MA60']].values
    target = df['Target'].values.reshape(-1, 1)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    features_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)
    
    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i+window_size])
        y.append(target_scaled[i+window_size])
        
    return np.array(X), np.array(y), scaler_x, scaler_y

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
    # 데이터 처리
    df = fetch_gold_data()
    df = preprocess_data(df)
    
    window_size = 60
    X, y, scaler_x, scaler_y = prepare_sequences(df, window_size)
    
    # 모델 하이퍼파라미터
    input_dim = 2
    hidden_dim = 64
    layer_dim = 2
    output_dim = 1
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # TimeSeriesSplit 설정
    tscv = TimeSeriesSplit(n_splits=5)
    mae_list = []
    
    print("\n학습 및 검증을 시작합니다 (TimeSeriesSplit)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 전체 예측 저장용
    all_actual = []
    all_predicted = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1} 학습 중...")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # DataLoader 생성
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
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
        
        # 역변환
        fold_preds_inv = scaler_y.inverse_transform(np.array(fold_preds))
        fold_actuals_inv = scaler_y.inverse_transform(np.array(fold_actuals))
        
        mae = mean_absolute_error(fold_actuals_inv, fold_preds_inv)
        mae_list.append(mae)
        print(f"Fold {fold+1} MAE: {mae:.2f}")
        
        # 마지막 폴드의 예측값 저장 (시각화용)
        if fold == 4:
            all_actual = fold_actuals_inv
            all_predicted = fold_preds_inv

    print(f"\n평균 MAE: {np.mean(mae_list):.2f}")

    # 시각화 (마지막 Fold 기준)
    plt.figure(figsize=(12, 6))
    plt.plot(all_actual, label='Actual Gold Price', color='gray', alpha=0.6)
    plt.plot(all_predicted, label='Predicted Gold Price (GRU)', color='blue')
    plt.title(f'Gold Price Prediction (GRU) - Last Fold (MAE: {np.mean(mae_list):.2f})')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('gold_gru_prediction.png')
    print("예측 결과 그래프가 'gold_gru_prediction.png'로 저장되었습니다.")
    
    # CSV 저장 (마지막 Fold 예측)
    res_df = pd.DataFrame({'Actual': all_actual.flatten(), 'Predicted': all_predicted.flatten()})
    res_df.to_csv('gold_gru_result.csv', index=False)
    print("예측 결과가 'gold_gru_result.csv'로 저장되었습니다.")
