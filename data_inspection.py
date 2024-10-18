import pandas as pd
import re

def inspect_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("数据集加载成功！\n")
        print("数据集的基本信息：")
        print(data.info())
        print("\n数据集的描述性统计：")
        print(data.describe())

        # 数据清理
        data = clean_data(data)
        
        # 将 Duration 列转换为分钟数
        data['DurationMinutes'] = data['Duration'].apply(convert_duration_to_minutes)
        print("\n转换后的时长列 (DurationMinutes)：")
        print(data[['Duration', 'DurationMinutes']].head())
        
        return data
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None

def clean_data(data):
    """ 清理数据：处理无效字符、空值和重复值等问题 """
    
    # 移除每个字段中的多余空白字符
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # 移除非ASCII字符（如果存在）
    data['MovieName'] = data['MovieName'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x) if isinstance(x, str) else x)
    
    # 删除空值，仅对特定关键列处理空值，例如 MovieName 和 ReleaseYear
    data.dropna(subset=['MovieName', 'ReleaseYear'], inplace=True)
    
    # 删除重复的行
    data.drop_duplicates(inplace=True)
    
    # 确保 ReleaseYear 列为数值型，并移除无法转换的值
    data['ReleaseYear'] = pd.to_numeric(data['ReleaseYear'], errors='coerce')
    data.dropna(subset=['ReleaseYear'], inplace=True)
    
    return data

def convert_duration_to_minutes(duration_str):
    """将时长转换为分钟"""
    try:
        duration_str = duration_str.lower().replace(' ', '')  # 去掉空格，转为小写
        if 'h' in duration_str:
            time_parts = duration_str.split('h')
            hours = int(time_parts[0].strip())
            minutes = int(time_parts[1].replace('min', '').strip()) if len(time_parts) > 1 else 0
        else:
            # 如果只有分钟信息，如 '45min'
            hours = 0
            minutes = int(duration_str.replace('min', '').strip())
        return hours * 60 + minutes
    except ValueError as e:
        print(f"解析时长失败: {duration_str}, 错误: {e}")
        return None  # 解析失败时返回 None
