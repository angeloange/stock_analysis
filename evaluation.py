import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def evaluate_individual_signals(df, days_forward=5):
    """評估各類技術指標的績效表現"""
    data = df.copy()
    
    # 計算未來n天後的回報率
    data[f'Return_{days_forward}d'] = data['Close'].pct_change(days_forward).shift(-days_forward)
    
    # 儲存各指標績效
    results = []
    
    # 找出所有信號欄位
    signal_cols = [col for col in data.columns if '_Signal' in col or col.endswith('_Col')]
    
    for signal_col in signal_cols:
        # 計算信號數量
        buy_signals = data.loc[data[signal_col] > 0]
        sell_signals = data.loc[data[signal_col] < 0]
        
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        total_signals = buy_count + sell_count
        
        if total_signals == 0:
            continue  # 跳過沒有交易信號的指標
        
        # 計算買入信號準確率 (n天後價格上漲的比例)
        if buy_count > 0:
            buy_accuracy = buy_signals[~buy_signals[f'Return_{days_forward}d'].isna()][f'Return_{days_forward}d'] > 0
            buy_accuracy = buy_accuracy.mean()
            buy_return = buy_signals[f'Return_{days_forward}d'].mean() * 100  # 轉換為百分比
        else:
            buy_accuracy = np.nan
            buy_return = np.nan
        
        # 計算賣出信號準確率 (n天後價格下跌的比例)
        if sell_count > 0:
            sell_accuracy = sell_signals[~sell_signals[f'Return_{days_forward}d'].isna()][f'Return_{days_forward}d'] < 0
            sell_accuracy = sell_accuracy.mean()
            sell_return = -sell_signals[f'Return_{days_forward}d'].mean() * 100  # 轉換為百分比
        else:
            sell_accuracy = np.nan
            sell_return = np.nan
        
        # 計算總準確率
        if buy_count > 0 and sell_count > 0:
            total_accuracy = (buy_signals[~buy_signals[f'Return_{days_forward}d'].isna()][f'Return_{days_forward}d'] > 0).sum() + \
                             (sell_signals[~sell_signals[f'Return_{days_forward}d'].isna()][f'Return_{days_forward}d'] < 0).sum()
            total_accuracy_denom = (~buy_signals[f'Return_{days_forward}d'].isna()).sum() + (~sell_signals[f'Return_{days_forward}d'].isna()).sum()
            total_accuracy = total_accuracy / total_accuracy_denom if total_accuracy_denom > 0 else np.nan
        elif buy_count > 0:
            total_accuracy = buy_accuracy
        elif sell_count > 0:
            total_accuracy = sell_accuracy
        else:
            total_accuracy = np.nan
        
        # 保存結果
        results.append({
            'Signal': signal_col,
            'Buy_Signals': buy_count,
            'Sell_Signals': sell_count,
            'Total_Signals': total_signals,
            'Buy_Accuracy': buy_accuracy,
            'Sell_Accuracy': sell_accuracy,
            'Total_Accuracy': total_accuracy,
            'Buy_Return': buy_return,
            'Sell_Return': sell_return
        })
    
    # 轉換為DataFrame
    results_df = pd.DataFrame(results).set_index('Signal')
    return results_df

def analyze_signal_combinations(data, indicators, days_forward=5):
    """分析多個指標信號共同出現時的表現"""
    from itertools import combinations
    
    df = data.copy()
    results = {}
    
    # 計算未來回報率
    if f'Return_{days_forward}d' not in df.columns:
        df[f'Return_{days_forward}d'] = df['Close'].pct_change(days_forward).shift(-days_forward)
    
    # 分析多種指標組合
    for i in range(2, min(len(indicators) + 1, 4)):  # 最多分析到3個指標的組合
        for combo in list(combinations(indicators, i)):
            combo_name = " + ".join(combo)
            
            # 找出所有指標同時發出買入信號的日期
            buy_mask = df[combo[0]] > 0  # 初始掩碼
            for ind in combo[1:]:
                buy_mask = buy_mask & (df[ind] > 0)
            
            # 找出所有指標同時發出賣出信號的日期
            sell_mask = df[combo[0]] < 0  # 初始掩碼
            for ind in combo[1:]:
                sell_mask = sell_mask & (df[ind] < 0)
            
            # 計算績效
            buy_signals = df[buy_mask]
            sell_signals = df[sell_mask]
            
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            
            if buy_count > 0:
                buy_accuracy = (buy_signals[f'Return_{days_forward}d'] > 0).mean()
                buy_return = buy_signals[f'Return_{days_forward}d'].mean() * 100
            else:
                buy_accuracy = float('nan')
                buy_return = float('nan')
            
            if sell_count > 0:
                sell_accuracy = (sell_signals[f'Return_{days_forward}d'] < 0).mean()
                sell_return = -sell_signals[f'Return_{days_forward}d'].mean() * 100
            else:
                sell_accuracy = float('nan')
                sell_return = float('nan')
            
            # 只保存至少有一個買入或賣出信號的組合
            if buy_count > 0 or sell_count > 0:
                results[combo_name] = {
                    'Indicators': combo,
                    'Buy_Signals': buy_count,
                    'Sell_Signals': sell_count,
                    'Buy_Accuracy': buy_accuracy,
                    'Sell_Accuracy': sell_accuracy,
                    'Buy_Return': buy_return,
                    'Sell_Return': sell_return,
                    'Total_Signals': buy_count + sell_count
                }


    # 轉換為DataFrame並排序
    if results:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        return results_df.sort_values(['Buy_Accuracy', 'Buy_Return'], ascending=False)
    else:
        return pd.DataFrame()