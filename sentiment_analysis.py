import matplotlib.pyplot as plt

def perform_sentiment_analysis(data):
    print("\n执行情感分析...")
    
    if 'Sentiment' in data.columns:
        sentiment_counts = data['Sentiment'].value_counts()
        print("\n情感分析结果：")
        print(sentiment_counts)

        # 可视化情感分布
        sentiment_counts.plot(kind='bar', title='Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()
    else:
        print("数据集不包含 'Sentiment' 列。")
