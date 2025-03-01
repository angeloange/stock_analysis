import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def backtest_strategy(df, signal_cols, initial_capital=1000000, position_size=0.2):
    """
    對多個交易信號進行回測
    
    參數:
        df (DataFrame): 包含價格和信號的數據框
        signal_cols (list): 要回測的信號列名列表
        initial_capital (float): 初始資金
        position_size (float): 每次交易使用的資本比例 (0-1)
    
    回傳:
        DataFrame: 回測結果統計
    """
    results = {}
    
    for signal_col in signal_cols:
        # 複製數據避免修改原始數據
        data = df.copy()
        
        # 初始化回測變數
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(data)):
            date = data.index[i]
            signal = data[signal_col].iloc[i]
            price = data['Close'].iloc[i]
            
            # 買入信號
            if signal > 0 and position == 0:
                # 計算可買入的股數
                trade_capital = capital * position_size
                shares = int(trade_capital / price)
                
                if shares > 0:
                    cost = shares * price
                    position = shares
                    capital -= cost
                    
                    trades.append({
                        '日期': date,
                        '類型': '買入',
                        '價格': price,
                        '數量': shares,
                        '成本': cost,
                        '剩餘資金': capital
                    })
            
            # 賣出信號
            elif signal < 0 and position > 0:
                # 賣出持有的股票
                proceeds = position * price
                capital += proceeds
                
                trades.append({
                    '日期': date,
                    '類型': '賣出',
                    '價格': price,
                    '數量': position,
                    '所得': proceeds,
                    '剩餘資金': capital
                })
                
                position = 0
        
        # 如果結束時還有持倉，以最後價格平倉
        if position > 0:
            last_price = data['Close'].iloc[-1]
            proceeds = position * last_price
            capital += proceeds
            
            trades.append({
                '日期': data.index[-1],
                '類型': '結束平倉',
                '價格': last_price,
                '數量': position,
                '所得': proceeds,
                '剩餘資金': capital
            })
        
        # 計算總收益和其他統計數據
        total_return = (capital / initial_capital - 1) * 100
        trades_df = pd.DataFrame(trades)
        
        # 如果有交易，計算交易統計數據
        if len(trades) > 0:
            buy_trades = trades_df[trades_df['類型'] == '買入']
            sell_trades = trades_df[trades_df['類型'].isin(['賣出', '結束平倉'])]
            
            avg_holding_days = 0
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0
            
            # 計算每筆交易的盈虧
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                buy_count = len(buy_trades)
                min_count = min(len(buy_trades), len(sell_trades))
                
                for i in range(min_count):
                    buy_price = buy_trades.iloc[i]['價格']
                    buy_shares = buy_trades.iloc[i]['數量']
                    sell_price = sell_trades.iloc[i]['價格']
                    
                    # 計算交易盈虧
                    trade_profit = (sell_price - buy_price) * buy_shares
                    
                    if trade_profit > 0:
                        winning_trades += 1
                        total_profit += trade_profit
                    else:
                        losing_trades += 1
                        total_loss += trade_profit
                    
                    # 計算持有時間
                    if i < min_count:
                        buy_date = buy_trades.iloc[i]['日期']
                        sell_date = sell_trades.iloc[i]['日期']
                        holding_days = (sell_date - buy_date).days
                        avg_holding_days += holding_days
                
                if min_count > 0:
                    avg_holding_days /= min_count
            
            win_rate = winning_trades / len(trades) * 100 if len(trades) > 0 else 0
            
            results[signal_col] = {
                '初始資金': initial_capital,
                '最終資金': capital,
                '總收益率(%)': total_return,
                '交易次數': len(trades) // 2,  # 買入+賣出算一次交易
                '勝率(%)': win_rate,
                '平均持倉天數': avg_holding_days,
                '總盈利': total_profit,
                '總虧損': total_loss,
                '盈虧比': abs(total_profit / total_loss) if total_loss != 0 else float('inf'),
                '交易記錄': trades_df
            }
        else:
            results[signal_col] = {
                '初始資金': initial_capital,
                '最終資金': capital,
                '總收益率(%)': 0,
                '交易次數': 0,
                '勝率(%)': 0,
                '平均持倉天數': 0,
                '總盈利': 0,
                '總虧損': 0,
                '盈虧比': 0,
                '交易記錄': pd.DataFrame()
            }
    
    # 將結果轉換為DataFrame
    results_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != '交易記錄'} 
                              for k, v in results.items()}).T
    
    # 按總收益率排序
    results_df = results_df.sort_values('總收益率(%)', ascending=False)
    
    return results_df, results

def plot_equity_curve(df, signal_col, trades_df, title=None):
    """繪製權益曲線和交易點"""
    if trades_df.empty:
        print(f"沒有交易記錄可供繪製: {signal_col}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 繪製收盤價
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='收盤價', color='blue', alpha=0.6)
    
    # 標記買入點和賣出點
    buy_points = trades_df[trades_df['類型'] == '買入']
    sell_points = trades_df[trades_df['類型'].isin(['賣出', '結束平倉'])]
    
    for idx, row in buy_points.iterrows():
        date = row['日期']
        if date in df.index:
            plt.scatter(date, df.loc[date, 'Close'], color='green', s=100, marker='^')
    
    for idx, row in sell_points.iterrows():
        date = row['日期']
        if date in df.index:
            plt.scatter(date, df.loc[date, 'Close'], color='red', s=100, marker='v')
    
    plt.title(title or f'{signal_col} 交易信號回測')
    plt.ylabel('價格')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 繪製資金曲線
    plt.subplot(2, 1, 2)
    
    # 初始資金
    initial_capital = trades_df['剩餘資金'].iloc[0] + trades_df['成本'].iloc[0] \
                      if '成本' in trades_df.columns and len(trades_df) > 0 else 1000000
    
    # 計算每筆交易後的總資產
    equity = [initial_capital]
    dates = [df.index[0]]
    
    position = 0
    capital = initial_capital
    
    for idx, row in trades_df.iterrows():
        date = row['日期']
        
        if row['類型'] == '買入':
            capital -= row['成本']
            position = row['數量']
        else:  # 賣出或結束平倉
            capital += row['所得']
            position = 0
        
        # 計算總資產 = 現金 + 持倉價值
        total_equity = capital
        if position > 0:
            if date in df.index:
                total_equity += position * df.loc[date, 'Close']
        
        equity.append(total_equity)
        dates.append(date)
    
    # 添加最後一個日期點
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        last_equity = capital
        if position > 0:
            last_equity += position * df['Close'].iloc[-1]
        equity.append(last_equity)
    
    plt.plot(dates, equity, color='purple', label='資金曲線')
    
    # 添加基準比較 (Buy & Hold)
    initial_shares = initial_capital / df['Close'].iloc[0]
    benchmark = df['Close'] * initial_shares
    plt.plot(df.index, benchmark, color='gray', linestyle='--', alpha=0.7, label='買入持有策略')
    
    plt.xlabel('日期')
    plt.ylabel('資產價值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_backtest(data, top_signals, days=None):
    """執行回測並顯示結果"""
    print(f"\n開始回測 {len(top_signals)} 個頂級信號...")
    
    # 執行回測
    backtest_results, detailed_results = backtest_strategy(data, top_signals)
    
    # 顯示回測結果
    print("\n回測結果摘要:")
    print(backtest_results[['總收益率(%)', '交易次數', '勝率(%)', '平均持倉天數', '盈虧比']])
    
    # 對表現最好的前3個信號繪製權益曲線
    top_performers = backtest_results.head(3).index.tolist()
    
    for signal in top_performers:
        trades_df = detailed_results[signal]['交易記錄']
        title = f"{signal} 交易回測 ({days}天預測)" if days else f"{signal} 交易回測"
        plot_equity_curve(data, signal, trades_df, title)
    
    return backtest_results, detailed_results