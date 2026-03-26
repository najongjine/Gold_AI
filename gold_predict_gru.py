import os
import random
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# 0. 재현성 설정
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1. 데이터 수집
# =========================
def fetch_gold_data(period='10y'):
    print('금 선물 데이터(GC=F)를 수집합니다...')
    df = yf.download('GC=F', period=period, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[needed].copy()
    df.index.name = 'Date'
    print(f'수집 완료: {df.index.min().date()} ~ {df.index.max().date()}, 총 {len(df)}건')
    return df


# =========================
# 2. 보조지표 함수
# =========================
def add_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


# =========================
# 3. 전처리 / 피처 생성
# =========================
def preprocess_data(df, horizon=60):
    print('지표 및 타깃을 생성합니다...')
    df = df.copy()

    # 수익률 (%)
    df['Return_1d'] = df['Close'].pct_change(1) * 100
    df['Return_5d'] = df['Close'].pct_change(5) * 100
    df['Return_20d'] = df['Close'].pct_change(20) * 100
    df['Return_60d'] = df['Close'].pct_change(60) * 100

    # 변동성 (%) - 일간 수익률 rolling std
    df['Volatility_5d'] = df['Return_1d'].rolling(5).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()
    df['Volatility_60d'] = df['Return_1d'].rolling(60).std()

    # 이동평균
    for w in [5, 20, 60, 120]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
        df[f'Disparity_MA_{w}'] = (df['Close'] / df[f'MA_{w}'] - 1) * 100

    # MA 간 기울기/관계
    df['MA5_vs_MA20'] = (df['MA_5'] / df['MA_20'] - 1) * 100
    df['MA20_vs_MA60'] = (df['MA_20'] / df['MA_60'] - 1) * 100
    df['MA60_vs_MA120'] = (df['MA_60'] / df['MA_120'] - 1) * 100
    df['MA_60_Momentum_5d'] = (df['MA_60'] / df['MA_60'].shift(5) - 1) * 100
    df['MA_60_Momentum_20d'] = (df['MA_60'] / df['MA_60'].shift(20) - 1) * 100

    # RSI, MACD
    df = add_rsi(df, window=14)
    df = add_macd(df)

    # 고가/저가/종가 기반 범위 정보
    df['HL_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df['Close_in_DayRange'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)

    df['RollingHigh_5'] = df['High'].rolling(5).max()
    df['RollingLow_5'] = df['Low'].rolling(5).min()
    df['RollingRange_5_Pct'] = ((df['RollingHigh_5'] - df['RollingLow_5']) / df['Close']) * 100
    df['Close_in_RollingRange_5'] = (df['Close'] - df['RollingLow_5']) / (df['RollingHigh_5'] - df['RollingLow_5']).replace(0, np.nan)

    df['RollingHigh_20'] = df['High'].rolling(20).max()
    df['RollingLow_20'] = df['Low'].rolling(20).min()
    df['RollingRange_20_Pct'] = ((df['RollingHigh_20'] - df['RollingLow_20']) / df['Close']) * 100
    df['Close_in_RollingRange_20'] = (df['Close'] - df['RollingLow_20']) / (df['RollingHigh_20'] - df['RollingLow_20']).replace(0, np.nan)

    df['RollingHigh_60'] = df['High'].rolling(60).max()
    df['RollingLow_60'] = df['Low'].rolling(60).min()
    df['RollingRange_60_Pct'] = ((df['RollingHigh_60'] - df['RollingLow_60']) / df['Close']) * 100
    df['Close_in_RollingRange_60'] = (df['Close'] - df['RollingLow_60']) / (df['RollingHigh_60'] - df['RollingLow_60']).replace(0, np.nan)

    # 거래량 정보(보조)
    df['Volume_Change_5d'] = df['Volume'].pct_change(5) * 100
    df['Volume_vs_MA20'] = (df['Volume'] / df['Volume'].rolling(20).mean() - 1) * 100

    # ===== 타깃 정의 =====
    # "현 시점 MA60 대비, 향후 60일 후 MA60가 몇 % 변하는가"
    # y(t) = (MA60[t+60] / MA60[t] - 1) * 100
    df['Target_MA60_Return_60d'] = (df['MA_60'].shift(-horizon) / df['MA_60'] - 1) * 100

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f'전처리 후 데이터 개수: {len(df)}')
    return df


# =========================
# 4. 시퀀스 생성
# =========================
def prepare_sequences(df, seq_len=120):
    feature_cols = [
        'Return_1d', 'Return_5d', 'Return_20d', 'Return_60d',
        'Volatility_5d', 'Volatility_20d', 'Volatility_60d',
        'MA_5', 'MA_20', 'MA_60', 'MA_120',
        'Disparity_MA_5', 'Disparity_MA_20', 'Disparity_MA_60', 'Disparity_MA_120',
        'MA5_vs_MA20', 'MA20_vs_MA60', 'MA60_vs_MA120',
        'MA_60_Momentum_5d', 'MA_60_Momentum_20d',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'HL_Range_Pct', 'Close_in_DayRange',
        'RollingRange_5_Pct', 'Close_in_RollingRange_5',
        'RollingRange_20_Pct', 'Close_in_RollingRange_20',
        'RollingRange_60_Pct', 'Close_in_RollingRange_60',
        'Volume_Change_5d', 'Volume_vs_MA20'
    ]

    X, y, y_dates = [], [], []
    feature_values = df[feature_cols].values
    target_values = df['Target_MA60_Return_60d'].values
    index_values = df.index.to_list()

    for i in range(len(df) - seq_len):
        X.append(feature_values[i:i+seq_len])
        y.append(target_values[i+seq_len])
        y_dates.append(index_values[i+seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y, y_dates, feature_cols


# =========================
# 5. 모델
# =========================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layer_dim=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            dropout=dropout if layer_dim > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# =========================
# 6. 학습/평가 함수
# =========================
def train_one_fold(model, train_loader, val_loader, device, num_epochs=60, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_x)

        model.eval()
        val_loss_sum = 0.0
        preds = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                val_loss_sum += loss.item() * len(batch_x)
                preds.extend(pred.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())

        train_losses.append(train_loss_sum / len(train_loader.dataset))
        val_losses.append(val_loss_sum / len(val_loader.dataset))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            mae = mean_absolute_error(actuals, preds)
            print(f'  Epoch {epoch+1:03d} | TrainLoss {train_losses[-1]:.4f} | ValLoss {val_losses[-1]:.4f} | ValMAE {mae:.4f}%')

    return train_losses, val_losses


# =========================
# 7. 전체 실행
# =========================
def main():
    seed_everything(42)

    horizon = 60     # 60일 후
    seq_len = 120    # 최근 120일 패턴 입력
    batch_size = 32
    num_epochs = 60
    hidden_dim = 64
    layer_dim = 2
    learning_rate = 1e-3
    n_splits = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 장치: {device}')

    df_raw = fetch_gold_data(period='10y')
    df_feat = preprocess_data(df_raw, horizon=horizon)
    X, y, y_dates, feature_cols = prepare_sequences(df_feat, seq_len=seq_len)

    print(f'입력 X shape: {X.shape}')
    print(f'타깃 y shape: {y.shape}')
    print(f'사용 피처 수: {len(feature_cols)}')

    # 누수 방지용 gap
    effective_gap = seq_len + horizon - 1
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=effective_gap)

    fold_metrics = []
    last_fold_preds = None
    last_fold_actuals = None
    last_fold_dates = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        print('\n' + '=' * 70)
        print(f'Fold {fold}/{n_splits}')
        print(f'Train: {len(train_idx)} | Validation: {len(val_idx)}')

        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # 스케일러는 train만 fit
        scaler = StandardScaler()
        n_train, L, F = X_train.shape
        n_val = X_val.shape[0]

        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(n_train, L, F)
        X_val_scaled = scaler.transform(X_val.reshape(-1, F)).reshape(n_val, L, F)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )

        model = GRUModel(input_dim=F, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device)
        train_one_fold(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=learning_rate)

        # 검증 예측
        model.eval()
        preds = []
        actuals = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                preds.extend(pred.cpu().numpy().flatten())
                actuals.extend(batch_y.numpy().flatten())

        preds = np.array(preds)
        actuals = np.array(actuals)
        val_dates = [y_dates[i] for i in val_idx]

        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        direction_acc = np.mean((actuals >= 0) == (preds >= 0)) * 100

        fold_metrics.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'direction_acc': direction_acc,
        })

        print(f'Fold {fold} 결과 | MAE: {mae:.4f}% | RMSE: {rmse:.4f}% | 방향정확도: {direction_acc:.2f}%')

        if fold == n_splits:
            last_fold_preds = preds
            last_fold_actuals = actuals
            last_fold_dates = val_dates

    metric_df = pd.DataFrame(fold_metrics)
    print('\n' + '=' * 70)
    print('교차검증 평균 성능')
    print(metric_df[['mae', 'rmse', 'direction_acc']].mean())
    metric_df.to_csv('gold_gru_cv_metrics.csv', index=False)

    # =========================
    # 8. 전체 데이터 재학습 후 최신 시점 예측
    # =========================
    print('\n전체 데이터로 재학습 후 최신 시점(가장 최근 데이터 기준) 60일 후 MA60 수익률을 예측합니다...')

    scaler_all = StandardScaler()
    N, L, F = X.shape
    X_all_scaled = scaler_all.fit_transform(X.reshape(-1, F)).reshape(N, L, F)

    full_loader = DataLoader(
        TensorDataset(torch.tensor(X_all_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    final_model = GRUModel(input_dim=F, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    final_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in full_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = final_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Final Train Epoch {epoch+1:03d} | Loss {epoch_loss / len(full_loader.dataset):.4f}')

    # 최신 1건 예측
    latest_sequence = X[-1:]  # 마지막 시퀀스
    latest_sequence_scaled = scaler_all.transform(latest_sequence.reshape(-1, F)).reshape(1, L, F)
    latest_sequence_tensor = torch.tensor(latest_sequence_scaled, dtype=torch.float32).to(device)

    final_model.eval()
    with torch.no_grad():
        latest_pred = final_model(latest_sequence_tensor).cpu().numpy().flatten()[0]

    latest_date = y_dates[-1]
    current_close = df_feat.loc[latest_date, 'Close']
    current_ma60 = df_feat.loc[latest_date, 'MA_60']
    current_disparity_ma60 = df_feat.loc[latest_date, 'Disparity_MA_60']

    # 모델이 예측한 미래 MA60 값 역산
    predicted_future_ma60 = current_ma60 * (1 + latest_pred / 100.0)

    direction = '상승' if latest_pred >= 0 else '하락'

    summary_df = pd.DataFrame([{
        'latest_feature_date': latest_date,
        'current_close': current_close,
        'current_ma60': current_ma60,
        'current_disparity_vs_ma60_pct': current_disparity_ma60,
        'predicted_ma60_return_60d_pct': latest_pred,
        'predicted_future_ma60': predicted_future_ma60,
        'predicted_direction': direction,
    }])
    summary_df.to_csv('gold_gru_latest_forecast.csv', index=False)

    print('\n' + '=' * 70)
    print('[최신 예측 결과]')
    print(f'기준일: {latest_date.date()}')
    print(f'현재 종가: {current_close:.2f}')
    print(f'현재 MA60: {current_ma60:.2f}')
    print(f'현재 종가가 MA60 대비 얼마나 떨어져/올라 있는지: {current_disparity_ma60:+.2f}%')
    print(f'예측: 향후 60일 뒤 MA60은 현재 MA60 대비 {latest_pred:+.2f}% {direction} 예상')
    print(f'예상 미래 MA60: {predicted_future_ma60:.2f}')

    # =========================
    # 9. 저장 및 시각화
    # =========================
    if last_fold_preds is not None:
        result_df = pd.DataFrame({
            'Date': last_fold_dates,
            'Actual_Target_MA60_Return_60d': last_fold_actuals,
            'Predicted_Target_MA60_Return_60d': last_fold_preds,
        })
        result_df.to_csv('gold_gru_last_fold_predictions.csv', index=False)

        plt.figure(figsize=(14, 6))
        plt.plot(result_df['Date'], result_df['Actual_Target_MA60_Return_60d'], label='Actual')
        plt.plot(result_df['Date'], result_df['Predicted_Target_MA60_Return_60d'], label='Predicted')
        plt.axhline(0, linestyle='--')
        plt.title('Gold GRU - 60일 후 MA60 수익률 예측 (Last Fold)')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('gold_gru_last_fold_plot.png')
        print("마지막 Fold 예측 그래프 저장: gold_gru_last_fold_plot.png")

    # 피처 목록 저장
    pd.DataFrame({'feature_name': feature_cols}).to_csv('gold_gru_feature_list.csv', index=False)
    print('피처 목록 저장: gold_gru_feature_list.csv')
    print('교차검증 지표 저장: gold_gru_cv_metrics.csv')
    print('최신 예측 저장: gold_gru_latest_forecast.csv')


if __name__ == '__main__':
    main()
