# 作用１：为 data 目录下的每一篇文章生成相应的 summary 放到 summaries 目录下
# 作用２：为 data 目录下的所有文章生成一个 term-document-matrix

import os
import jieba
import pandas as pd
import numpy as np

article_names = os.listdir('data')

articles = list()
for article_name in article_names:
    file_name = 'data/' + article_name
    article_file = open(file_name, 'r')
    article_content = article_file.read()
    articles.append(article_content)
    article_file.close()

all_words = set()

def to_word_freq_table(article: str):
    
    seg_list = jieba.cut(article)
    
    freq_stat = dict()
    for word in seg_list:
        
        all_words.add(word)
        
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
        'counts': counts,
        'stats': freq_stat
    }

article_summaries = list()

for article in articles:
    article_summary = to_word_freq_table(article)
    
    article_summaries.append(
        article_summary
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

doc_matrix = list()
all_words_vector = list(all_words)
for article_summary in article_summaries:
    article_word_vector = list()
    for word in all_words_vector:
        if word in article_summary['stats']:
            article_word_vector.append(article_summary['stats'][word])
        else:
            article_word_vector.append(0)
    doc_matrix.append(article_word_vector)

index_to_word = pd.DataFrame({
    'index': np.array(range(0, len(all_words))),
    'word': all_words_vector
})

index_to_article = pd.DataFrame({
    'index': np.array(range(0, len(article_names))),
    'article': article_names
})

index_to_word.to_csv('termdocmatrix/index_to_word.csv', index=False)

index_to_article.to_csv('termdocmatrix/index_to_article.csv', index=False)

f = open('termdocmatrix/matrix.txt', 'w')
doc_matrix_str = str()
for row in doc_matrix:
    doc_matrix_str = doc_matrix_str + ','.join(map(str,row))+'\n'
f.write(doc_matrix_str)
f.close()

