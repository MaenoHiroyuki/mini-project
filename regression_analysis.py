import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据清理
def clean_data_for_regression(data, dependent_var, independent_var):
    """清理回归分析中使用的变量，删除 NaN 和 inf 值"""
    # 删除 NaN 值
    data_clean = data.dropna(subset=[dependent_var, independent_var])
    
    # 删除 inf 和 -inf 值
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[dependent_var, independent_var])
    
    return data_clean

# 回归分析可视化
def visualize_regression(data, independent_var, dependent_var):
    """可视化回归结果"""
    data_clean = data[data[dependent_var] < 500]  # 假设时长大于500分钟的数据为异常值

    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(data_clean[independent_var], data_clean[dependent_var], color='blue', label='Data Points', alpha=0.5)
    
    # 绘制回归线
    X = sm.add_constant(data_clean[[independent_var]])
    model = sm.OLS(data_clean[dependent_var], X).fit()
    plt.plot(data_clean[independent_var], model.predict(X), color='red', label='Regression Line')
    
    # 添加标题和标签
    plt.title(f'Regression: {dependent_var} vs {independent_var}', fontsize=14)
    plt.xlabel(independent_var, fontsize=12)
    plt.ylabel(dependent_var, fontsize=12)
    
    # 设置图例
    plt.legend(loc='best', fontsize=10)
    
    # 显示图表
    plt.show()

# 执行回归分析
def perform_regression(data, independent_var, dependent_var):
    print(f"\n执行回归分析 ({independent_var} vs {dependent_var})...")
    data_clean = clean_data_for_regression(data, dependent_var, independent_var)

    if not data_clean.empty:
        try:
            X = data_clean[[independent_var]]
            Y = data_clean[dependent_var]
            X = sm.add_constant(X)  # 添加常量项
            model = sm.OLS(Y, X).fit()
            print("\n回归分析结果：")
            print(model.summary())

            # 结果解释
            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\n结果解释：{independent_var} 对 {dependent_var} 有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（显著）。")
            else:
                print(f"\n结果解释：{independent_var} 对 {dependent_var} 没有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（不显著）。")
            
            # 调用可视化函数
            visualize_regression(data_clean, independent_var, dependent_var)

        except Exception as e:
            print(f"回归分析失败: {e}")
    else:
        print(f"数据集不包含有效的 {independent_var} 或 {dependent_var} 列。")

# 演员数量对电影时长的回归分析
def perform_actors_duration_regression(data):
    print("\n执行演员数量对电影时长的回归分析...")
    data_clean = clean_data_for_regression(data, 'DurationMinutes', 'Actors')
    
    if not data_clean.empty:
        try:
            # 计算每部电影的演员数量
            data_clean['NumActors'] = data_clean['Actors'].apply(lambda x: len(x.split('|')) if isinstance(x, str) else 0)
            X = data_clean[['NumActors']]
            Y = data_clean['DurationMinutes']
            X = sm.add_constant(X)  # 添加常量项
            model = sm.OLS(Y, X).fit()
            print("\n回归分析结果：")
            print(model.summary())

            # 结果解释
            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\n结果解释：演员数量对电影时长有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（显著）。")
            else:
                print(f"\n结果解释：演员数量对电影时长没有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（不显著）。")

            # 可视化回归结果
            visualize_regression(data_clean, 'NumActors', 'DurationMinutes')

        except Exception as e:
            print(f"回归分析失败: {e}")
    else:
        print("数据集不包含有效的 Actors 或 DurationMinutes 列。")

# 电影时长对情感的回归分析
def perform_duration_sentiment_regression(data):
    print("\n执行电影时长对情感的回归分析...")
    data_clean = clean_data_for_regression(data, 'DurationMinutes', 'Sentiment')
    
    if not data_clean.empty:
        try:
            # 将 Sentiment 转换为数值
            sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            data_clean['SentimentScore'] = data_clean['Sentiment'].map(sentiment_map)

            X = data_clean[['DurationMinutes']]
            Y = data_clean['SentimentScore']
            X = sm.add_constant(X)  # 添加常量项
            model = sm.OLS(Y, X).fit()
            print("\n回归分析结果：")
            print(model.summary())

            # 结果解释
            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\n结果解释：电影时长对情感有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（显著）。")
            else:
                print(f"\n结果解释：电影时长对情感没有显著影响，R平方为 {r_squared:.4f}，p值为 {p_value:.4f}（不显著）。")

            # 可视化回归结果
            visualize_regression(data_clean, 'DurationMinutes', 'SentimentScore')

        except Exception as e:
            print(f"回归分析失败: {e}")
    else:
        print("数据集不包含有效的 DurationMinutes 或 Sentiment 列。")

