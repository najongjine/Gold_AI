import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime

# --------------------------------------------------
# 1. 조회 기간 설정: 오늘 기준 10년 전 ~ 오늘
# --------------------------------------------------
end = datetime.datetime.today()

try:
    start = end.replace(year=end.year - 10)
except ValueError:
    start = end.replace(year=end.year - 10, day=28)

print(f"데이터 다운로드 기간: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")

# --------------------------------------------------
# 2. FRED 데이터 가져오기
# --------------------------------------------------
fred_df = pd.DataFrame()

try:
    print("FRED 데이터 다운로드 중...")
    fred_df = web.DataReader('IPG334S', 'fred', start, end)
    fred_df.rename(columns={'IPG334S': 'Computer_Electronic_Production_Index'}, inplace=True)
    print("FRED 데이터 로드 성공")
except Exception as e:
    print(f"FRED 데이터 로드 실패: {e}")

# --------------------------------------------------
# 3. yfinance 데이터 가져오기
# --------------------------------------------------
yf_df = pd.DataFrame()

yf_tickers = {
    'DX-Y.NYB': 'Dollar_Index',
    '^TNX': 'US_10Y_Yield',
    '^VIX': 'VIX',
    'BTC-USD': 'Bitcoin'
}

try:
    print("yfinance 데이터 다운로드 중...")
    temp = yf.download(
        list(yf_tickers.keys()),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if isinstance(temp.columns, pd.MultiIndex):
        yf_df = temp['Close'].copy()
    else:
        yf_df = temp.copy()

    yf_df.rename(columns=yf_tickers, inplace=True)
    print("yfinance 데이터 로드 성공")
except Exception as e:
    print(f"yfinance 데이터 로드 실패: {e}")

# --------------------------------------------------
# 4. 데이터 병합
# --------------------------------------------------
df_list = []

if not fred_df.empty:
    df_list.append(fred_df)

if not yf_df.empty:
    df_list.append(yf_df)

if not df_list:
    raise ValueError("가져온 데이터가 없습니다.")

combined_df = pd.concat(df_list, axis=1)
combined_df.sort_index(inplace=True)
combined_df.ffill(inplace=True)
combined_df.bfill(inplace=True)

print("\n병합 완료. 상위 5개 행:")
print(combined_df.head())

# --------------------------------------------------
# 5. 금 투자용 최소형 텍스트 생성 함수
# --------------------------------------------------
def get_past_value(series: pd.Series, days: int):
    """
    현재 시점에서 days일 전과 가장 가까운 과거 값을 찾음
    """
    if series.empty:
        return None

    target_date = series.index[-1] - pd.Timedelta(days=days)
    past_data = series[series.index <= target_date]

    if past_data.empty:
        return None

    return past_data.iloc[-1]

def calc_change_pct(current, past):
    if past is None or pd.isna(past) or past == 0:
        return None
    return ((current - past) / past) * 100

def make_gold_minimal_text(df: pd.DataFrame) -> str:
    """
    금 투자용 GPT 입력 최소형 텍스트 생성
    핵심 지표만 짧게 요약
    """
    # 금 투자에 상대적으로 중요한 지표만 우선 사용
    priority_cols = [
        'Dollar_Index',
        'US_10Y_Yield',
        'VIX',
    ]

    # 옵션 지표
    optional_cols = [
        'Bitcoin',
        'Computer_Electronic_Production_Index'
    ]

    lines = []
    lines.append("=== Gold Investment Macro Snapshot ===")
    lines.append(f"Base date: {df.index.max().strftime('%Y-%m-%d')}")
    lines.append("")

    for col in priority_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        current = series.iloc[-1]
        current_date = series.index[-1].strftime('%Y-%m-%d')

        v_1m = get_past_value(series, 30)
        v_3m = get_past_value(series, 90)
        v_1y = get_past_value(series, 365)

        ch_1m = calc_change_pct(current, v_1m)
        ch_3m = calc_change_pct(current, v_3m)
        ch_1y = calc_change_pct(current, v_1y)

        mean_10y = series.mean()
        min_10y = series.min()
        max_10y = series.max()

        if current > mean_10y:
            vs_mean = "above_10y_mean"
        elif current < mean_10y:
            vs_mean = "below_10y_mean"
        else:
            vs_mean = "near_10y_mean"

        lines.append(f"[{col}]")
        lines.append(f"- latest_date: {current_date}")
        lines.append(f"- current: {current:.4f}")
        lines.append(f"- change_1m_pct: {ch_1m:.2f}" if ch_1m is not None else "- change_1m_pct: N/A")
        lines.append(f"- change_3m_pct: {ch_3m:.2f}" if ch_3m is not None else "- change_3m_pct: N/A")
        lines.append(f"- change_1y_pct: {ch_1y:.2f}" if ch_1y is not None else "- change_1y_pct: N/A")
        lines.append(f"- mean_10y: {mean_10y:.4f}")
        lines.append(f"- min_10y: {min_10y:.4f}")
        lines.append(f"- max_10y: {max_10y:.4f}")
        lines.append(f"- position_vs_10y_mean: {vs_mean}")
        lines.append("")

    # 옵션 지표는 짧게만 넣기
    for col in optional_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        current = series.iloc[-1]
        mean_10y = series.mean()

        lines.append(f"[Optional:{col}]")
        lines.append(f"- current: {current:.4f}")
        lines.append(f"- mean_10y: {mean_10y:.4f}")
        lines.append("")

    lines.append("[Interpretation Guide]")
    lines.append("- Stronger dollar often pressures gold.")
    lines.append("- Higher US 10Y yield often pressures gold.")
    lines.append("- Higher VIX can increase safe-haven demand, but short-term reaction can vary.")
    lines.append("")

    return "\n".join(lines)

# --------------------------------------------------
# 6. 최종 텍스트 생성
# --------------------------------------------------
gold_text = make_gold_minimal_text(combined_df)

print("\n===== 금 투자용 최소형 텍스트 =====")
print(gold_text)

# --------------------------------------------------
# 7. 저장
# --------------------------------------------------
with open("gold_macro_minimal.txt", "w", encoding="utf-8") as f:
    f.write(gold_text)

combined_df.to_csv("macro_market_full_10y.csv", encoding="utf-8-sig")

print("\n저장 완료:")
print("- gold_macro_minimal.txt")
print("- macro_market_full_10y.csv")