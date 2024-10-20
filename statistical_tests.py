import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'D:\SCSE2024\mini-project\data\anova_movie_data_corrected.csv'
data = pd.read_csv(file_path)

data['Duration'] = data['Duration'].astype(str).apply(lambda x: x if x not in ['nan', 'NaN'] else '0')

def convert_duration(duration_str):
    duration_str = duration_str.strip()
    if 'h' in duration_str and 'min' in duration_str:
        hours, minutes = duration_str.split('h')
        minutes = minutes.replace('min', '').strip()
        return int(hours) * 60 + int(minutes)
    elif 'min' in duration_str:
        return int(duration_str.replace('min', '').strip())
    return 0

data['Duration'] = data['Duration'].apply(convert_duration)

sentiment_groups = data.groupby('Sentiment')['Duration'].apply(list)
t_test_results = stats.ttest_ind(sentiment_groups['Positive'], sentiment_groups['Neutral'])

contingency_table = pd.crosstab(data['Sentiment'], data['YearCategory'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Sentiment', y='Duration', data=data, palette='Set2')
plt.title('Duration by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Duration (minutes)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title('Chi-Square Contingency Table')
plt.xlabel('Year Category')
plt.ylabel('Sentiment')
plt.show()

print("T-test results: T-statistic =", t_test_results.statistic, ", p-value =", t_test_results.pvalue)
if t_test_results.pvalue < 0.05:
    print("Reject the null hypothesis: There is a significant difference in duration between Positive and Neutral sentiments.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in duration between Positive and Neutral sentiments.")

print("Chi-square test results: chi2 =", chi2, ", p-value =", p)
if p < 0.05:
    print("Reject the null hypothesis: There is a significant association between sentiment and year category.")
else:
    print("Fail to reject the null hypothesis: There is no significant association between sentiment and year category.")
