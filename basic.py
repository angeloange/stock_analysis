import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# 設定股票代號和時間範圍
ticker = '5439.TWO'  # 高技
start_date = '2018-01-01' 
end_date = '2025-02-28'

print(f"正在下載 {ticker} 的股票數據...")
stock_data = yf.download(ticker, start=start_date, end=end_date)
print(f"數據下載完成，共 {len(stock_data)} 筆交易日資料")

# 檢查數據結構
print("檢查數據結構:")
print(stock_data.columns)

# 詳細診斷
print("\n詳細列出所有列名:")
for col in stock_data.columns:
    print(f"- {col} (類型: {type(col)})")

# 如果數據有多層索引，簡化它
if isinstance(stock_data.columns, pd.MultiIndex):
    # 檢查每個級別
    for level in range(stock_data.columns.nlevels):
        print(f"級別 {level}: {stock_data.columns.get_level_values(level).tolist()}")
    
    # 使用第0級別的索引 (修改了這裡)
    stock_data.columns = stock_data.columns.get_level_values(0)
    print("簡化後的列名:", stock_data.columns)

# 確保必要的列存在
if 'Close' not in stock_data.columns:
    if 'Adj Close' in stock_data.columns:
        print("未找到'Close'列，使用'Adj Close'替代")
        stock_data['Close'] = stock_data['Adj Close']
    else:
        print("錯誤：數據缺少必要的列。重新嘗試下載...")
        # 重新下載，不使用自動調整
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        print("重新下載後的列名:", stock_data.columns)
        
        # 再次檢查必要的列
        if 'Close' not in stock_data.columns:
            print("錯誤：無法獲取必要的'Close'列，程序終止")
            exit(1)
# ============== 1. 計算全部技術指標 ==============
def calculate_indicators(df):
    """計算所有常用技術指標"""
    data = df.copy()
    
    # --- RSI (相對強弱指數) ---
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # 計算不同週期的 RSI
    for period in [9, 14, 25]:
        data[f'RSI_{period}'] = calculate_rsi(data['Close'], period)
    
    # --- MACD (移動平均匯聚背馳指標) ---
    # 標準 MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # --- 移動平均線 (MA) ---
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
    
    # --- 布林帶 (Bollinger Bands) ---
    for period in [20, 40]:
        for std_mult in [1.5, 2, 2.5]:
            bb_col = f'BB_{period}_{std_mult}'
            
            # 計算中間線、上下軌 (逐步計算避免多列問題)
            ma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            
            data[f'{bb_col}_MA'] = ma
            data[f'{bb_col}_Upper'] = ma + (std_mult * std)
            data[f'{bb_col}_Lower'] = ma - (std_mult * std)
            
            # 計算 %B 值
            upper = data[f'{bb_col}_Upper']
            lower = data[f'{bb_col}_Lower']
            close = data['Close']
            
            b_percent = (close - lower) / (upper - lower)
            data[f'{bb_col}_%B'] = b_percent
            
            # 處理可能的除零問題
            data[f'{bb_col}_%B'].fillna(0.5, inplace=True)
            data.loc[data[f'{bb_col}_%B'] == np.inf, f'{bb_col}_%B'] = 0.5
            data.loc[data[f'{bb_col}_%B'] == -np.inf, f'{bb_col}_%B'] = 0.5
    
    # --- KD 隨機指標 ---
    for window in [5, 9, 14]:
        for d_period in [3, 5]:
            k_col = f'%K_{window}'
            d_col = f'%D_{window}_{d_period}'
            
            # 計算 K、D 値 (逐步計算避免多列問題)
            low_min = data['Low'].rolling(window=window).min()
            high_max = data['High'].rolling(window=window).max()
            
            data[f'Low_{window}'] = low_min
            data[f'High_{window}'] = high_max
            
            # 計算 %K
            k_values = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            data[k_col] = k_values
            
            # 處理可能的除零或無效值
            data[k_col].fillna(50, inplace=True)
            data.loc[data[k_col] == np.inf, k_col] = 50
            data.loc[data[k_col] == -np.inf, k_col] = 50
            
            # 計算 %D (K的移動平均)
            data[d_col] = data[k_col].rolling(window=d_period).mean()
    
    # --- OBV (On-Balance Volume) ---
    if 'Volume' in data.columns:
        # 計算成交量移動平均
        for period in [5, 10, 20, 50]:
            data[f'Volume_MA_{period}'] = data['Volume'].rolling(window=period).mean()
        
        # 安全計算 OBV (確保使用純量值)
        obv = [0]  # 首筆資料 OBV 為 0
        for i in range(1, len(data)):
            current_close = float(data['Close'].iloc[i])
            prev_close = float(data['Close'].iloc[i-1])
            current_volume = float(data['Volume'].iloc[i])
            
            if current_close > prev_close:
                obv.append(obv[-1] + current_volume)
            elif current_close < prev_close:
                obv.append(obv[-1] - current_volume)
            else:
                obv.append(obv[-1])
        
        data['OBV'] = obv
    
    return data

# ============== 2. 計算全部交易信號 ==============
def calculate_signals(df):
    """計算各種技術指標的交易信號"""
    data = df.copy()
    
    # --- KD 隨機指標信號 ---
    for window in [5, 9, 14]:
        for d_period in [3, 5]:
            k_col = f'%K_{window}'
            d_col = f'%D_{window}_{d_period}'
            signal_col = f'KD_{window}_{d_period}_Signal'
            data[signal_col] = 0
            
            # 買入信號: K線在低檔區(20以下)向上穿越D線
            data.loc[(data[k_col] < 20) & 
                     (data[k_col] > data[d_col]) & 
                     (data[k_col].shift(1) <= data[d_col].shift(1)), 
                     signal_col] = 1
            
            # 賣出信號: K線在高檔區(80以上)向下穿越D線
            data.loc[(data[k_col] > 80) & 
                     (data[k_col] < data[d_col]) & 
                     (data[k_col].shift(1) >= data[d_col].shift(1)), 
                     signal_col] = -1
            
            # 備用信號: 黃金交叉/死亡交叉 (任何區域K線穿越D線)
            bullish_cross_col = f'KD_{window}_{d_period}_GoldenCross'
            bearish_cross_col = f'KD_{window}_{d_period}_DeathCross'
            
            data[bullish_cross_col] = 0
            data[bearish_cross_col] = 0
            
            # 黃金交叉 (K上穿D)
            data.loc[(data[k_col] > data[d_col]) & 
                     (data[k_col].shift(1) <= data[d_col].shift(1)), 
                     bullish_cross_col] = 1
            
            # 死亡交叉 (K下穿D)
            data.loc[(data[k_col] < data[d_col]) & 
                     (data[k_col].shift(1) >= data[d_col].shift(1)), 
                     bearish_cross_col] = -1
    
    # --- RSI 信號 --- (注意：這部分已從 KD 迴圈移出)
    for period in [9, 14, 25]:
        signal_col = f'RSI_{period}_Signal'
        data[signal_col] = 0
        
        # 買入: RSI低於30後回升
        data.loc[(data[f'RSI_{period}'] < 30) & (data[f'RSI_{period}'].shift(1) < 30) & 
                 (data[f'RSI_{period}'] > data[f'RSI_{period}'].shift(1)), signal_col] = 1
        
        # 賣出: RSI高於70後回落
        data.loc[(data[f'RSI_{period}'] > 70) & (data[f'RSI_{period}'].shift(1) > 70) & 
                 (data[f'RSI_{period}'] < data[f'RSI_{period}'].shift(1)), signal_col] = -1
    
    # --- MACD 信號 --- (注意：這部分已從 KD 迴圈移出)
    data['MACD_Signal_Col'] = 0
    
    # 買入: MACD上穿Signal Line
    data.loc[(data['MACD'] > data['MACD_Signal']) & 
             (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)), 'MACD_Signal_Col'] = 1
    
    # 賣出: MACD下穿Signal Line
    data.loc[(data['MACD'] < data['MACD_Signal']) & 
             (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)), 'MACD_Signal_Col'] = -1
    
    return data