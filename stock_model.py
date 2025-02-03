import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta  # 使用替代套件
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score

# ====================
# 核心功能模組
# ====================

def fetch_data(stock_code="2330.TW", years=3):
    """抓取歷史股價數據"""
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    data = yf.download(
        stock_code,
        start=start_date,
        end=end_date,
        progress=False
    )
    
    # 添加市場指數數據（台灣加權指數）
    twii = yf.download("^TWII", start=start_date, end=end_date)['Close']
    data['Market_Index'] = twii
    
    return data.dropna()

def calculate_features(df):
    """使用 ta 套件計算技術指標"""
    # ============== 技術指標 ==============
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()          # MACD線
    df['MACD_Signal'] = macd.macd_signal()  # 信號線
    
    # 布林通道
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # 能量潮 (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()
    
    # 平均趨向指數 (ADX)
    df['ADX'] = ta.trend.ADXIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).adx()
    
    # ============== 自訂特徵 ==============
    df['Price_Change'] = df['Close'].pct_change()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Market_Correlation'] = df['Close'].rolling(30).corr(df['Market_Index'])
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    return df.dropna()

def prepare_dataset(df, lookback=30):
    """準備機器學習數據"""
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 預測隔日漲跌
    
    features = df[['RSI', 'MACD', 'MACD_Signal', 'OBV', 'ADX', 'MA_20', 'MA_50', 'Market_Correlation']]
    targets = df['Target']
    
    # 滾動窗口數據
    X = np.lib.stride_tricks.sliding_window_view(features, (lookback, features.shape[1]))
    X = X.reshape(-1, lookback * features.shape[1])
    y = targets.iloc[lookback-1:]
    
    return X, y

# ====================
# 儀表板界面
# ====================

def main():
    st.set_page_config(page_title="台股智能分析系統", layout="wide")
    
    # 側邊欄控制
    st.sidebar.header("⚙️ 參數設定")
    stock_code = st.sidebar.text_input("股票代碼", "2330.TW")
    years = st.sidebar.slider("數據年份", 1, 10, 3)
    lookback = st.sidebar.slider("回溯期數", 10, 60, 30)
    
    # 數據處理
    raw_data = fetch_data(stock_code, years)
    if raw_data.empty:
        st.error("無法取得數據，請檢查股票代碼是否正確")
        return
    
    processed_data = calculate_features(raw_data)
    X, y = prepare_dataset(processed_data, lookback)
    
    # 主畫面佈局
    st.title(f"📊 {stock_code} 智能分析儀表板")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # 股價走勢圖
        st.subheader("股價走勢與技術指標")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=processed_data.index,
                        open=processed_data['Open'],
                        high=processed_data['High'],
                        low=processed_data['Low'],
                        close=processed_data['Close'],
                        name='K線'))
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA_20'], 
                               line=dict(color='orange'), name='20日均線'))
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA_50'], 
                               line=dict(color='purple'), name='50日均線'))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # 即時數據看板
        st.subheader("即時數據")
        latest = processed_data.iloc[-1]
        st.metric("當前股價", f"NT${latest['Close']:.2f}")
        st.metric("RSI指數", f"{latest['RSI']:.1f}", 
                 "超買" if latest['RSI'] > 70 else "超賣" if latest['RSI'] < 30 else "正常")
        st.metric("市場相關性", f"{latest['Market_Correlation']:.2f}")
    
    # 機器學習預測區
    st.subheader("📈 智能預測系統")
    if st.button("啟動模型訓練"):
        with st.spinner('模型訓練中...'):
            # 時間序列交叉驗證
            tss = TimeSeriesSplit(n_splits=3)
            accuracies = []
            
            for train_idx, test_idx in tss.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
                model.fit(X_train, y_train)
                accuracies.append(accuracy_score(y_test, model.predict(X_test)))
            
            avg_acc = np.mean(accuracies)
            latest_data = X[-1].reshape(1, -1)
            prediction = model.predict(latest_data)[0]
            
            st.success(f"模型平均準確率：{avg_acc:.2%}")
            st.markdown(f"### 明日走勢預測：{'📈 上漲' if prediction == 1 else '📉 下跌'}")

if __name__ == "__main__":
    main()