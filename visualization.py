import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import platform
# 在文件頂部
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
# 設定中文字體支援
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback']


def visualize_signal_performance(results, days_forward):
    """視覺化指標信號的績效表現"""
    if results.empty:
        print("沒有結果可以視覺化!")
        return
    
    # 排除NaN值的結果
    valid_results = results.dropna(subset=['Buy_Accuracy', 'Sell_Accuracy'])
    
    if valid_results.empty:
        print("沒有有效結果可以視覺化!")
        return
    
    # 限制顯示數量
    if len(valid_results) > 15:
        print(f"指標數量過多 ({len(valid_results)}), 僅顯示表現最好的15個")
        valid_results = valid_results.sort_values('Buy_Accuracy', ascending=False).head(15)
    
    # 自定義顏色映射 - 低於0.5為紅色，高於0.5為綠色
    cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                         [(0, 'indianred'), (0.5, 'white'), (1, 'forestgreen')])
    
    plt.figure(figsize=(14, 10))
    
    # 1. 準確率比較
    plt.subplot(2, 1, 1)
    ind = valid_results.index
    width = 0.35
    x = np.arange(len(ind))
    
    buy_bars = plt.bar(x - width/2, valid_results['Buy_Accuracy'], width, 
                     color=[cmap(acc) for acc in valid_results['Buy_Accuracy']], 
                     label='買入準確率')
    
    sell_bars = plt.bar(x + width/2, valid_results['Sell_Accuracy'], width, 
                      color=[cmap(acc) for acc in valid_results['Sell_Accuracy']], 
                      alpha=0.7, label='賣出準確率')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('指標')
    plt.ylabel('準確率')
    plt.title(f'各指標信號{days_forward}天後的準確率')
    plt.xticks(x, ind, rotation=45, ha='right')
    plt.legend()
    
    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(buy_bars):
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., 1.02*height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    for i, bar in enumerate(sell_bars):
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., 1.02*height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 回報率比較
    plt.subplot(2, 1, 2)
    # 替換NaN值為0以便繪圖
    buy_returns_data = valid_results['Buy_Return'].fillna(0)
    sell_returns_data = valid_results['Sell_Return'].fillna(0)
    
    buy_returns = plt.bar(x - width/2, buy_returns_data, width, 
                        color='green', alpha=0.7, label='買入回報率(%)')
    
    sell_returns = plt.bar(x + width/2, sell_returns_data, width, 
                         color='red', alpha=0.7, label='賣出回報率(%)')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('指標')
    plt.ylabel('回報率(%)')
    plt.title(f'各指標信號{days_forward}天後的回報率')
    plt.xticks(x, ind, rotation=45, ha='right')
    plt.legend()
    
    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(buy_returns):
        height = bar.get_height()
        if abs(height) > 0.1:  # 忽略接近零的值
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.3 if height > 0 else height - 0.8,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    for i, bar in enumerate(sell_returns):
        height = bar.get_height()
        if abs(height) > 0.1:  # 忽略接近零的值
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.3 if height > 0 else height - 0.8,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 修改圖表顯示方式，從互動式改為保存檔案
    plt.tight_layout()
    # plt.show()  # 註解這行
    plt.savefig(f'signal_performance_{days_forward}d.png')  # 保存為圖檔
    plt.close()  # 關閉圖表，釋放資源