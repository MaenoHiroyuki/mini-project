import anova_analysis
import regression_analysis
import sentiment_analysis
import pandas as pd

# 转换电影时长为分钟
def convert_duration_to_minutes(duration_str):
    if isinstance(duration_str, str):
        time_parts = duration_str.lower().replace('h', '').replace('min', '').strip().split()
        hours = int(time_parts[0]) if len(time_parts) > 0 else 0
        minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
        return hours * 60 + minutes
    return None

# 预处理数据集
# 基于 Actors 列计算 NumActors
def preprocess_data(data):
   if 'Actors' in data.columns:
        # 将 '|' 分隔的演员数量计为 `NumActors`
    data['NumActors'] = data['Actors'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)

    # 转换 Duration 列为分钟数
    data['DurationMinutes'] = data['Duration'].apply(convert_duration_to_minutes)

    # 将 YearCategory 转换为 category 类型
    data = anova_analysis.assign_yearcategory(data)

    print("\n转换后的时长列 (DurationMinutes)：")
    print(data[['Duration', 'DurationMinutes']].head())
    print("\nYearCategory 列预处理后前 5 行：")
    print(data[['ReleaseYear', 'YearCategory']].head())
    
    return data

# 加载数据集
def load_data():
    file_path = input("请输入数据集路径 (CSV格式): ")
    data = pd.read_csv(file_path)
    print("数据集加载成功！")
    print("\n数据集的基本信息：")
    print(data.info())
    print("\n数据集的描述性统计：")
    print(data.describe())
    
    # 调用预处理函数
    data = preprocess_data(data)
   
    
    print("\n转换后的数据集预览：")
    print(data[['Actors', 'NumActors']].head())  # 打印出新生成的NumActors列
    return data

# 主函数
def main():
    data = load_data()
    data = preprocess_data(data)  

    while True:
        print("\n请选择分析类型:")
        print("1. ANOVA分析 (YearCategory vs ReleaseYear)")
        print("2. ANOVA分析 (Director vs Duration)")
        print("3. ANOVA分析 (Writer vs Duration)")
        print("4. ANOVA分析 (演员数量 vs Duration)")
        print("5. 回归分析 (ReleaseYear vs Duration)")
        print("6. 回归分析 (演员数量 vs Duration)")
        print("7. 回归分析 (电影时长 vs 情感)")
        print("8. 情感分析 (Sentiment Analysis)")
        print("9. ANOVA分析 (Director vs Sentiment vs Duration)")
        print("10. ANOVA分析 (Year vs Sentiment)")
        print("11. ANOVA分析 (Writer_vs_Duration)")
        print("12. 退出程序")

        choice = input("请输入选择(1-12): ")

        if choice == '1':
            anova_analysis.perform_anova(data, 'YearCategory', 'ReleaseYear', 'YearCategory vs ReleaseYear')
        elif choice == '2':
            anova_analysis.perform_anova(data, 'Director', 'DurationMinutes', 'Director vs Duration')
        elif choice == '3':
            anova_analysis.perform_anova(data, 'Writer', 'DurationMinutes', 'Writer vs Duration')
        elif choice == '4':
            anova_analysis.perform_anova(data, 'Actors', 'DurationMinutes', 'Number of Actors vs Duration')
        elif choice == '5':
            regression_analysis.perform_regression(data, 'ReleaseYear', 'DurationMinutes')
        elif choice == '6':
            regression_analysis.perform_actors_duration_regression(data)
        elif choice == '7':
            regression_analysis.perform_duration_sentiment_regression(data)
        elif choice == '8':
            sentiment_analysis.perform_sentiment_analysis(data)
        elif choice == '9':
            anova_analysis.director_sentiment_vs_duration(data)
        elif choice == '10':
            anova_analysis.year_vs_sentiment_analysis(data)  # 正确的函数名是 year_vs_sentiment_analysis
        elif choice == '11':
            anova_analysis.writer_vs_duration_analysis(data)
        elif choice == '12':
            print("退出程序。")
            break
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    main()
