

# ============================================================
# 2. import
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import copy
import math
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 3. 재현성
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# ============================================================
# 4. 데이터 수집
# ============================================================
tickers = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "tnx": "^TNX",
    "oil": "CL=F",
    "silver": "SI=F",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "gld": "GLD"
}

start_date = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

print("데이터 수집 기간:", start_date, "~", end_date)

raw = yf.download(
    list(tickers.values()),
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

close = raw["Close"].copy()
rename_map = {v: k for k, v in tickers.items()}
close = close.rename(columns=rename_map)

if isinstance(close, pd.Series):
    close = close.to_frame()

df = close.copy().sort_index().ffill().bfill()
df["gold_price"] = df["gold"]

# ============================================================
# 5. 60일 이동평균(smoothing)
# ============================================================
df["gold_smooth_60"] = df["gold_price"].rolling(60).mean()

# ============================================================
# 6. 파생변수 생성
# ============================================================
# 금 자체
df["gold_ret_1"] = df["gold_price"].pct_change(1)
df["gold_ret_5"] = df["gold_price"].pct_change(5)
df["gold_ret_10"] = df["gold_price"].pct_change(10)
df["gold_ret_20"] = df["gold_price"].pct_change(20)
df["gold_ret_60"] = df["gold_price"].pct_change(60)

df["gold_smooth_ret_5"] = df["gold_smooth_60"].pct_change(5)
df["gold_smooth_ret_20"] = df["gold_smooth_60"].pct_change(20)

df["gold_ma_20"] = df["gold_price"].rolling(20).mean()
df["gold_ma_60"] = df["gold_price"].rolling(60).mean()
df["gold_ma_120"] = df["gold_price"].rolling(120).mean()

df["gold_ma20_ma60_gap"] = (df["gold_ma_20"] - df["gold_ma_60"]) / df["gold_ma_60"]
df["gold_ma60_ma120_gap"] = (df["gold_ma_60"] - df["gold_ma_120"]) / df["gold_ma_120"]

df["gold_vol_20"] = df["gold_ret_1"].rolling(20).std()
df["gold_vol_60"] = df["gold_ret_1"].rolling(60).std()

df["gold_mom_20"] = df["gold_price"] / df["gold_price"].shift(20) - 1
df["gold_mom_60"] = df["gold_price"] / df["gold_price"].shift(60) - 1

# 연관 자산
related_assets = ["dxy", "tnx", "oil", "silver", "sp500", "vix", "gld"]

for col in related_assets:
    if col in df.columns:
        df[f"{col}_ret_1"] = df[col].pct_change(1)
        df[f"{col}_ret_5"] = df[col].pct_change(5)
        df[f"{col}_ret_20"] = df[col].pct_change(20)
        df[f"{col}_ma20"] = df[col].rolling(20).mean()
        df[f"{col}_ma60"] = df[col].rolling(60).mean()
        df[f"{col}_ma_gap"] = (df[f"{col}_ma20"] - df[f"{col}_ma60"]) / df[f"{col}_ma60"]

# 비율 feature
if {"gold", "silver"}.issubset(df.columns):
    df["gold_silver_ratio"] = df["gold"] / df["silver"]

if {"gold", "dxy"}.issubset(df.columns):
    df["gold_dxy_ratio"] = df["gold"] / df["dxy"]

if {"gold", "oil"}.issubset(df.columns):
    df["gold_oil_ratio"] = df["gold"] / df["oil"]

# 달력 feature
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# ============================================================
# 7. 타겟 정의 (회귀)
# ============================================================
# 목표:
# "현재 60일 이동평균 대비, 20거래일 뒤 60일 이동평균이 몇 % 변하는가?"
forecast_horizon = 60

df["future_ma60"] = df["gold_smooth_60"].shift(-forecast_horizon)
df["target_return_pct"] = (df["future_ma60"] / df["gold_smooth_60"]) - 1

# 참고용: 미래 MA 값 자체도 나중에 계산할 수 있음
# predicted_future_ma60 = current_ma60 * (1 + predicted_return_pct)

# ============================================================
# 8. 모델용 데이터 정리
# ============================================================
drop_cols = ["future_ma60", "target_return_pct"]
feature_cols = [c for c in df.columns if c not in drop_cols]

model_df = df[feature_cols + ["target_return_pct"]].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna().copy()

print("최종 데이터 크기:", model_df.shape)

# ============================================================
# 9. 시퀀스 생성
# ============================================================
seq_len = 60

def make_sequences(dataframe, feature_cols, target_col, seq_len=60):
    X_seq, y_seq = [], []
    seq_dates = []
    current_ma60_list = []
    current_price_list = []

    feature_data = dataframe[feature_cols].values
    target_data = dataframe[target_col].values
    dates = dataframe.index.values

    # 현재 기준값 저장용
    current_ma60 = dataframe["gold_smooth_60"].values
    current_price = dataframe["gold_price"].values

    for i in range(seq_len, len(dataframe)):
        X_seq.append(feature_data[i-seq_len:i])
        y_seq.append(target_data[i])
        seq_dates.append(dates[i])
        current_ma60_list.append(current_ma60[i])
        current_price_list.append(current_price[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(seq_dates),
        np.array(current_ma60_list, dtype=np.float32),
        np.array(current_price_list, dtype=np.float32)
    )

X_all, y_all, date_all, current_ma60_all, current_price_all = make_sequences(
    dataframe=model_df,
    feature_cols=feature_cols,
    target_col="target_return_pct",
    seq_len=seq_len
)

print("시퀀스 X shape:", X_all.shape)
print("시퀀스 y shape:", y_all.shape)

# ============================================================
# 10. Dataset 정의
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# 11. PyTorch GRU 회귀 모델
# ============================================================
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]   # 마지막 시점 출력만 사용
        out = self.fc(last_hidden)
        return out

# ============================================================
# 12. 학습 함수
# ============================================================
def train_one_fold(
    X_train, y_train,
    X_val, y_val,
    input_size,
    batch_size=32,
    lr=1e-3,
    epochs=40,
    patience=7
):
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRURegressor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # -------------------------
        # train
        # -------------------------
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # -------------------------
        # val
        # -------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_wts)
    return model

# ============================================================
# 13. 예측 함수
# ============================================================
def predict_model(model, X):
    dataset = TimeSeriesDataset(X, np.zeros(len(X), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.squeeze(1).cpu().numpy())

    return np.array(preds)

# ============================================================
# 14. TimeSeriesSplit 검증
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []
oof_preds = np.zeros(len(y_all), dtype=np.float32)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
    print(f"\n================ Fold {fold} ================")

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # --------------------------------------------------
    # scaler는 train에만 fit (누수 방지)
    # --------------------------------------------------
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape).astype(np.float32)

    model = train_one_fold(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_test_scaled,
        y_val=y_test,
        input_size=n_features,
        batch_size=32,
        lr=1e-3,
        epochs=40,
        patience=7
    )

    preds = predict_model(model, X_test_scaled)
    oof_preds[test_idx] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    fold_results.append({
        "fold": fold,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })

    print(f"train size : {len(train_idx)}")
    print(f"test size  : {len(test_idx)}")
    print(f"MAE        : {mae:.6f}")
    print(f"RMSE       : {rmse:.6f}")
    print(f"R2         : {r2:.6f}")

# ============================================================
# 15. 전체 검증 결과
# ============================================================
print("\n===== TimeSeriesSplit 결과 =====")
for item in fold_results:
    print(
        f"Fold {item['fold']} | "
        f"train={item['train_size']} | "
        f"test={item['test_size']} | "
        f"MAE={item['mae']:.6f} | "
        f"RMSE={item['rmse']:.6f} | "
        f"R2={item['r2']:.6f}"
    )

valid_mask = oof_preds != 0  # 초기 0인 앞부분 제외용
y_valid = y_all[valid_mask]
pred_valid = oof_preds[valid_mask]

print("\n===== 전체 OOF 회귀 성능 =====")
print("MAE :", round(mean_absolute_error(y_valid, pred_valid), 6))
print("RMSE:", round(np.sqrt(mean_squared_error(y_valid, pred_valid)), 6))
print("R2  :", round(r2_score(y_valid, pred_valid), 6))

# 방향성 정확도도 참고로 계산
direction_true = (y_valid > 0).astype(int)
direction_pred = (pred_valid > 0).astype(int)
direction_acc = (direction_true == direction_pred).mean()

print("방향성 정확도(상승/하락만 본 경우):", round(direction_acc, 4))

# ============================================================
# 16. 전체 데이터로 최종 재학습
# ============================================================
n_features = X_all.shape[2]

scaler_final = StandardScaler()
X_all_2d = X_all.reshape(-1, n_features)
X_all_scaled = scaler_final.fit_transform(X_all_2d).reshape(X_all.shape).astype(np.float32)

final_model = train_one_fold(
    X_train=X_all_scaled,
    y_train=y_all,
    X_val=X_all_scaled[-200:],   # 간단한 종료용
    y_val=y_all[-200:],
    input_size=n_features,
    batch_size=32,
    lr=1e-3,
    epochs=40,
    patience=7
)

# ============================================================
# 17. 최신 시점 예측
# ============================================================
latest_seq = X_all[-1:]
latest_seq_scaled = scaler_final.transform(
    latest_seq.reshape(-1, n_features)
).reshape(latest_seq.shape).astype(np.float32)

latest_pred_return = predict_model(final_model, latest_seq_scaled)[0]

latest_date = pd.to_datetime(date_all[-1])
current_ma60 = float(current_ma60_all[-1])
current_price = float(current_price_all[-1])

predicted_future_ma60 = current_ma60 * (1 + latest_pred_return)
predicted_vs_ma60_pct = latest_pred_return * 100
predicted_vs_price_pct = ((predicted_future_ma60 / current_price) - 1) * 100

print("\n===== 최신 시점 기준 예측 =====")
print("기준일:", latest_date.date())
print(f"현재 금 가격                 : {current_price:.2f}")
print(f"현재 60일 이동평균           : {current_ma60:.2f}")
print(f"예측 미래 60일 이동평균      : {predicted_future_ma60:.2f}")
print(f"현재 이동평균 대비 변화율     : {predicted_vs_ma60_pct:.2f}%")
print(f"현재 금 가격 대비 미래 MA60   : {predicted_vs_price_pct:.2f}%")

if latest_pred_return > 0:
    print("해석: 현재 60일 이동평균 대비 상승 예상")
else:
    print("해석: 현재 60일 이동평균 대비 하락 예상")

# ============================================================
# 18. 최근 구간 실제 vs 예측 보기
# ============================================================
result_df = pd.DataFrame({
    "date": pd.to_datetime(date_all),
    "current_price": current_price_all,
    "current_ma60": current_ma60_all,
    "actual_future_return_pct": y_all * 100,
    "pred_future_return_pct": oof_preds * 100
})

result_df = result_df[result_df["pred_future_return_pct"] != 0].copy()
result_df["actual_future_ma60"] = result_df["current_ma60"] * (1 + result_df["actual_future_return_pct"] / 100.0)
result_df["pred_future_ma60"] = result_df["current_ma60"] * (1 + result_df["pred_future_return_pct"] / 100.0)

print("\n===== 최근 10개 예측 결과 =====")
display(result_df.tail(10))

# ============================================================
# 19. 시각화 1 - 금 가격과 60일 이동평균
# ============================================================
plt.figure(figsize=(16, 6))
plt.plot(df.index, df["gold_price"], label="Gold Price")
plt.plot(df.index, df["gold_smooth_60"], label="Gold Smooth 60", linewidth=2)
plt.title("Gold Price and 60-day Smoothed Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 20. 시각화 2 - 실제 미래 변화율 vs 예측 변화율
# ============================================================
plt.figure(figsize=(16, 6))
plt.plot(result_df["date"], result_df["actual_future_return_pct"], label="Actual Future Return %")
plt.plot(result_df["date"], result_df["pred_future_return_pct"], label="Predicted Future Return %")
plt.axhline(0, linestyle="--")
plt.title(f"Future {forecast_horizon}-day MA60 Return Prediction (%)")
plt.xlabel("Date")
plt.ylabel("Return %")
plt.legend()
plt.grid(True)
plt.show()