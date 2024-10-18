import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as ols
import matplotlib.pyplot as plt
import seaborn as sns

# 将 ReleaseYear 转换为 YearCategory
def assign_yearcategory(data):
    # 按时间段将 ReleaseYear 转换为 YearCategory
    bins = [1900, 1924.9, 1949.9, 1974.9, 1999.9, 2024]
    labels = ['1900.0-1924.9', '1925.0-1949.9', '1950.0-1974.9', '1975.0-1999.9', '2000.0-2024.9']
    
    # 生成 YearCategory 列
    data['YearCategory'] = pd.cut(data['ReleaseYear'], bins=bins, labels=labels, right=False)
    return data


def clean_numactors(data):
    # 过滤掉 NumActors 为 0 的数据，因为电影通常至少有一名演员
    data_clean = data[data['NumActors'] > 0]
    print(f"清理后的 NumActors 列数据分布：\n{data_clean['NumActors'].value_counts()}")
    return data_clean

# 生成 NumActors 列，表示每部电影的演员数量
def preprocess_numactors(data):
    if 'Actors' in data.columns:
        # 将演员名单分割并统计每部电影的演员数量
        data['NumActors'] = data['Actors'].apply(lambda x: len(str(x).split('|')))
    else:
        print("数据集中没有 'Actors' 列。")
    return data


def balance_sample(data, column, target_sample_size=None):
    # 获取每个类别的样本数，如果不指定 target_sample_size 则获取最小样本数
    try:
        if target_sample_size is None:
            min_sample_size = data[column].value_counts().min()
        else:
            min_sample_size = target_sample_size
        
        # 如果有类别的样本数量为0，则跳过这个类别
        if min_sample_size == 0:
            raise ValueError(f"{column} 中的某些类别没有样本数据，无法进行均衡。")

        balanced_data = data.groupby(column).apply(lambda x: x.sample(min_sample_size)).reset_index(drop=True)
        return balanced_data
    except Exception as e:
        print(f"平衡样本时发生错误: {e}")
        return None



def encode_sentiment(data):
    # 创建新的 SentimentScore 列，将情感转换为数值
    sentiment_mapping = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
    }
    data['SentimentScore'] = data['Sentiment'].map(sentiment_mapping)
    
    # 检查是否存在未映射的情感，并填充为0（可以根据具体需求修改）
    data['SentimentScore'] = data['SentimentScore'].fillna(0)
    
    print("\n情感列已转换为数值型（SentimentScore）：")
    print(data[['Sentiment', 'SentimentScore']].head())
    
    return data


# ANOVA分析主函数
def perform_anova(data, independent_var, dependent_var, analysis_type):
    print(f"\n执行 {analysis_type} 的 ANOVA 分析...")

    # 打印数据检查是否有足够的数据
    print(data[[independent_var, dependent_var]].head(20))

    # 检查空值
    print(f"{independent_var} 列的空值数量：{data[independent_var].isna().sum()}")
    print(f"{dependent_var} 列的空值数量：{data[dependent_var].isna().sum()}")

    # 删除空值
    data_clean = data.dropna(subset=[independent_var, dependent_var])

    # 如果类别数过多，进行抽样或过滤
    if data_clean[independent_var].nunique() > 50:  # 根据情况调整类别数量阈值
        print(f"{independent_var} 的类别过多（{data_clean[independent_var].nunique()} 类别），进行抽样处理...")
        top_categories = data_clean[independent_var].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean[independent_var].isin(top_categories)]
        print(f"保留前 50 个最常见的类别用于分析。")

    # 如果清理后的数据行数过少，提示用户并退出
    if data_clean.shape[0] < 2:
        print(f"数据量不足，无法执行 {analysis_type}。")
        return

    # 生成公式
    formula = f'{dependent_var} ~ C({independent_var})'

    # 执行ANOVA分析
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print("\nANOVA 分析结果：")
        print(anova_table)

        # 提取 F 值和 p 值
        f_value = anova_table['F'].iloc[0]  # 修改为 iloc 避免 FutureWarning
        p_value = anova_table['PR(>F)'].iloc[0]  # 修改为 iloc 避免 FutureWarning

        # 解释结果
        interpret_anova_results(f_value, p_value, independent_var, dependent_var)

        # 可视化
        visualize_anova(data_clean, independent_var, dependent_var, analysis_type)

    except Exception as e:
        print(f"ANOVA 分析失败: {e}")



# 结果解释函数
def interpret_anova_results(f_value, p_value, independent_var, dependent_var, threshold=0.05):
    print(f"\nANOVA结果: F值={f_value}, p值={p_value}")
    if p_value < threshold:
        print(f"结论: 拒绝H0假设。{independent_var} 对 {dependent_var} 有显著影响 (p值 < {threshold})")
    else:
        print(f"结论: 无法拒绝H0假设。{independent_var} 对 {dependent_var} 没有显著影响 (p值 >= {threshold})")


# 更新后的 visualize_anova 函数
def visualize_anova(data, independent_var, dependent_var, analysis_type):
    # 去除 DurationMinutes 列中的异常值，假设超过 240 分钟为异常值
    if dependent_var == "DurationMinutes":
        data_clean = data[data[dependent_var] <= 240]
    else:
        data_clean = data.copy()

    # 处理 YearCategory 分类顺序
    if independent_var == 'YearCategory':
        data_clean = assign_yearcategory(data_clean)

    # 检查 YearCategory 列是否有空值
    print(f"{independent_var} 的值为: {data_clean[independent_var].unique()}")
    
    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=independent_var, y=dependent_var, data=data_clean)
    plt.title(f'Box Plot of {dependent_var} by {independent_var} ({analysis_type})')
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)
    plt.xticks(rotation=90)  # 避免文本过长时重叠
    plt.show()

def director_sentiment_vs_duration(data):
    print("\n执行 Director vs Sentiment vs Duration 的 ANOVA 分析...")

    # 打印数据，检查 Director 和 Sentiment 列
    print(data[['Director', 'Sentiment']].head(20))

    # 检查空值
    print(f"Director 列的空值数量：{data['Director'].isna().sum()}")
    print(f"Sentiment 列的空值数量：{data['Sentiment'].isna().sum()}")

    # 删除空值
    data_clean = data.dropna(subset=['Director', 'Sentiment', 'DurationMinutes'])

    # 将 Sentiment 转换为数值
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    data_clean['SentimentScore'] = data_clean['Sentiment'].map(sentiment_map)

    # 如果 Director 类别数过多，进行抽样或过滤
    if data_clean['Director'].nunique() > 50:
        print(f"Director 的类别过多（{data_clean['Director'].nunique()} 类别），进行抽样处理...")
        top_categories = data_clean['Director'].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean['Director'].isin(top_categories)]
        print(f"保留前 50 个最常见的类别用于分析。")

    # 如果清理后的数据行数过少，提示用户并退出
    if data_clean.shape[0] < 2:
        print("数据量不足，无法执行 Director vs Sentiment vs Duration 的 ANOVA 分析。")
        return

    # 生成公式
    formula = 'DurationMinutes ~ C(Director) + SentimentScore'

    # 执行ANOVA分析
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print("\nANOVA 分析结果：")
        print(anova_table)

        # 提取 F 值和 p 值
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]

        # 解释结果
        interpret_anova_results(f_value, p_value, 'Director 和 Sentiment', 'DurationMinutes')

        # 可视化
        visualize_anova(data_clean, 'Director', 'DurationMinutes', 'Director vs Sentiment vs Duration')

    except Exception as e:
        print(f"ANOVA 分析失败: {e}")


# 将情感转换为数值编码
def encode_sentiment(data):
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    data['SentimentScore'] = data['Sentiment'].map(sentiment_mapping)
    return data

# Year vs Sentiment ANOVA分析
def year_vs_sentiment_analysis(data):
    print("\n执行 Year vs Sentiment 的 ANOVA 分析...")
    
    # 对情感进行编码
    data = encode_sentiment(data)
    
    # 打印数据检查
    print(data[['ReleaseYear', 'SentimentScore']].head(20))
    
    # 检查空值
    print(f"ReleaseYear 列的空值数量：{data['ReleaseYear'].isna().sum()}")
    print(f"SentimentScore 列的空值数量：{data['SentimentScore'].isna().sum()}")

    # 删除空值
    data_clean = data.dropna(subset=['ReleaseYear', 'SentimentScore'])

    # 如果 ReleaseYear 类别数过多，进行抽样或过滤
    if data_clean['ReleaseYear'].nunique() > 50:
        print(f"ReleaseYear 的类别过多（{data_clean['ReleaseYear'].nunique()} 类别），进行抽样处理...")
        top_years = data_clean['ReleaseYear'].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean['ReleaseYear'].isin(top_years)]
        print(f"保留前 50 个最常见的年份用于分析。")

    # 执行 ANOVA 分析
    formula = 'SentimentScore ~ C(ReleaseYear)'
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print("\nANOVA 分析结果：")
        print(anova_table)

        # 提取 F 值和 p 值
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]

        # 结果解释
        print(f"\nANOVA结果: F值={f_value}, p值={p_value}")
        if p_value < 0.05:
            print(f"结论: 拒绝H0假设。ReleaseYear 对 SentimentScore 有显著影响 (p值 < 0.05)")
        else:
            print(f"结论: 无法拒绝H0假设。ReleaseYear 对 SentimentScore 没有显著影响 (p值 >= 0.05)")

        # 可视化：使用条形图展示年份对情感得分的影响
        visualize_year_vs_sentiment(data_clean)

    except Exception as e:
        print(f"ANOVA 分析失败: {e}")

# 使用条形图展示 Year vs Sentiment
def visualize_year_vs_sentiment(data):
    plt.figure(figsize=(12, 6))
    
    # 使用条形图展示每个年份的情感得分
    # 在 visualize_year_vs_sentiment_analysis 函数中修改绘图部分：
    sns.barplot(x="ReleaseYear", y="SentimentScore", data=data, estimator=sum, errorbar=None)

    
    plt.title('Bar Plot of SentimentScore by ReleaseYear (Year vs Sentiment)')
    plt.xlabel('ReleaseYear')
    plt.ylabel('SentimentScore (Sum)')
    plt.xticks(rotation=90)
    plt.show()


# 新增功能3：编剧 vs 时长的关系
def writer_vs_duration_analysis(data):
    print("\n执行 Writer vs Duration 的 ANOVA 分析...")
    perform_anova(data, 'Writer', 'DurationMinutes', 'Writer vs Duration')








# 现有可视化功能
def visualize_anova_yearcategory_vs_releaseyear(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="YearCategory", y="ReleaseYear", data=data)
    plt.title('Box Plot of ReleaseYear by YearCategory')
    plt.xlabel('YearCategory')
    plt.ylabel('ReleaseYear')
    plt.show()

def visualize_anova_director_vs_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Director", y="DurationMinutes", data=data)
    plt.title('Box Plot of DurationMinutes by Director')
    plt.xlabel('Director')
    plt.ylabel('DurationMinutes')
    plt.xticks(rotation=90)
    plt.show()

def visualize_anova_writer_vs_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Writer", y="DurationMinutes", data=data)
    plt.title('Box Plot of DurationMinutes by Writer')
    plt.xlabel('Writer')
    plt.ylabel('DurationMinutes')
    plt.xticks(rotation=90)
    plt.show()

def visualize_anova_numactors_vs_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="NumActors", y="DurationMinutes", data=data)
    plt.title('Box Plot of DurationMinutes by Number of Actors')
    plt.xlabel('NumActors')
    plt.ylabel('DurationMinutes')
    plt.show()

def visualize_anova_sentiment_vs_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Sentiment", y="DurationMinutes", data=data)
    plt.title('Box Plot of DurationMinutes by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('DurationMinutes')
    plt.show()

def visualize_anova_yearcategory_vs_sentiment(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="YearCategory", y="Sentiment", data=data)
    plt.title('Box Plot of Sentiment by YearCategory')
    plt.xlabel('YearCategory')
    plt.ylabel('Sentiment')
    plt.xticks(rotation=90)
    plt.show()
