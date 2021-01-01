import os
import jieba
import numpy as np
import pandas as pd

article_names = os.listdir('data')

articles = list()
for article_name in article_names:
    file_name = 'data/' + article_name
    article_file = open(file_name, 'r')
    article_content = article_file.read()
    articles.append(article_content)
    article_file.close()

def to_word_freq_table(article: str):
    
    seg_list = jieba.cut(article)
    
    freq_stat = dict()
    for word in seg_list:
        if not (word in freq_stat):
            freq_stat[word] = 1
        else:
            freq_stat[word] = freq_stat[word] + 1
            
    words = list()
    counts = list()
    for w, c in freq_stat.items():
        words.append(w)
        counts.append(c)
    
    return {
        'words': words,
        'counts': counts
    }

article_summaries = list()
for article in articles:
    article_summaries.append(
        to_word_freq_table(article)
    )

def to_summary_table(summary: dict):
    words = summary['words']
    counts = summary['counts']
    
    table = pd.DataFrame({
        'words': words,
        'counts': np.array(counts)
    })
    
    return table

article_summary_tables = list()
for article_summary in article_summaries:
    article_summary_table = to_summary_table(article_summary)
    article_summary_tables.append(article_summary_table)

for i in range(len(article_names)):
    article_name = article_names[i]
    article_summary_file_name = 'summaries/summary-of-'+article_name+'.csv'
    article_summary_table = article_summary_tables[i]
    article_summary_table.to_csv(path_or_buf=article_summary_file_name, index=False)