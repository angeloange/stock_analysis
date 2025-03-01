from backtest import run_backtest
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 引入自定義模組
from basic import calculate_indicators, calculate_signals
from evaluation import evaluate_individual_signals, analyze_signal_combinations
from visualization import visualize_signal_performance

# 關閉警告
warnings.filterwarnings('ignore')

def main():
    print("=== 技術指標分析系統 ===")
    
    # 設定股票代號和時間範圍
    ticker = '5439.TWO'  # 預設股票
    start_date = '2018-01-01' 
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 下載股票數據
    print(f"\n正在下載 {ticker} 的股票數據...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(f"數據下載完成，共 {len(stock_data)} 筆交易日資料")
    
    # 處理多層索引
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # 確保必要的列存在
    if 'Close' not in stock_data.columns and 'Adj Close' in stock_data.columns:
        stock_data['Close'] = stock_data['Adj Close']
    
    # 計算技術指標
    print("\n計算技術指標中...")
    data_with_indicators = calculate_indicators(stock_data)
    
    # 計算交易信號
    print("計算交易信號中...")
    data_with_signals = calculate_signals(data_with_indicators)
    
    # 設定評估參數
    evaluation_periods = [5, 10, 20]  # 天數
    
    # 評估個別指標的績效
    for days in evaluation_periods:
        print(f"\n評估 {days} 天後的指標表現...")
        results = evaluate_individual_signals(data_with_signals, days_forward=days)
        
        # 顯示表現最好的指標
        top_results = results.sort_values('Buy_Accuracy', ascending=False)
        print(f"\n表現最佳的指標 (依{days}天後買入準確率排序):")
        print(top_results.head(10)[['Buy_Signals', 'Sell_Signals', 'Buy_Accuracy', 'Sell_Accuracy', 'Buy_Return', 'Sell_Return']])
        
        # 視覺化結果
        print("\n繪製績效圖表...")
        try:
            # 只調用一次
            visualize_signal_performance(top_results, days)
        except Exception as e:
            print(f"繪製圖表時發生錯誤: {e}")
            print("繼續執行下一步分析...")
        
        # 選擇前N個表現最好的信號進行組合分析
        N = 5
        top_signals = top_results.head(N).index.tolist()
        
        print(f"\n分析前{N}個最佳指標的組合...")
        combo_results = analyze_signal_combinations(data_with_signals, top_signals, days_forward=days)
        
        if not combo_results.empty:
            print("\n表現最佳的組合:")
            print(combo_results.head(5)[['Buy_Signals', 'Sell_Signals', 'Buy_Accuracy', 'Sell_Accuracy', 'Buy_Return', 'Sell_Return']])
        else:
            print("沒有找到有效的信號組合")
        
        # 對表現最好的信號進行回測
        print("\n對表現最好的信號進行回測...")
        backtest_top_n = 10  # 回測前10個信號
        signals_to_backtest = top_results.head(backtest_top_n).index.tolist()
        backtest_results, detailed_results = run_backtest(data_with_signals, signals_to_backtest, days)
        # 在 main.py 末尾添加這段代碼
        def check_current_signals(data, top_signals):
            """檢查當前最新的交易信號"""
            # 獲取最近的數據點
            latest_date = data.index[-1]
            print(f"\n===== 當前市場信號 ({latest_date}) =====")
            
            print("\n表現最佳的指標目前信號:")
            
            # 檢查每個頂級指標的最新信號
            for signal in top_signals:
                latest_value = data.loc[latest_date, signal]
                
                if latest_value > 0:
                    recommendation = "買入"
                    emoji = "📈"
                elif latest_value < 0:
                    recommendation = "賣出"
                    emoji = "📉"
                else:
                    recommendation = "持觀望"
                    emoji = "⏸️"
                    
                print(f"{emoji} {signal}: {recommendation}")
            
            # 額外檢查最佳組合信號
            print("\n最佳指標組合的綜合建議:")
            buy_signals = sum(1 for signal in top_signals[:3] if data.loc[latest_date, signal] > 0)
            sell_signals = sum(1 for signal in top_signals[:3] if data.loc[latest_date, signal] < 0)
            
            if buy_signals > sell_signals and buy_signals >= 2:
                print("✅ 多數頂級指標顯示買入信號 - 建議考慮買入")
            elif sell_signals > buy_signals and sell_signals >= 2:
                print("❌ 多數頂級指標顯示賣出信號 - 建議考慮賣出")
            else:
                print("⚠️ 信號分歧或不明確 - 建議持觀望態度")

        # 在主函數最後調用它
        for days in [5]:  # 我們主要關注短期信號
            # 獲取前幾名的信號
            top_signals = top_results.head(5).index.tolist()
            check_current_signals(data_with_signals, top_signals)
        print('結束程式碼')
if __name__ == "__main__":
    main()