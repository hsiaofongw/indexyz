# 作用：验证 term-document-matrix 有没有算对

import pandas as pd
import numpy as np

doc_matrix = np.loadtxt(
    fname = 'termdocmatrix/matrix.txt', 
    dtype='int', 
    delimiter=','
)

index_to_word = pd.read_csv('termdocmatrix/index_to_word.csv')
index_to_article = pd.read_csv('termdocmatrix/index_to_article.csv')

for i in range(0, doc_matrix.shape[0]):
    article_name = index_to_article.iloc[i, 1]
    stats[article_name] = dict()
    for j in range(0, doc_matrix.shape[1]):
        word = index_to_word.iloc[j, 1]
        count = doc_matrix[i, j]
        if count > 0:
            stats[article_name][word] = count
            
    summary_file_name = 'summaries/summary-of-'+article_name+'.csv'
    summary = pd.read_csv(summary_file_name)
    somethingwrong = False
    for k in range(0, summary.shape[0]):
        word = summary.iloc[k, 0]
        
        pre_calculated_count = summary.iloc[k, 1]
        just_calculated_count = stats[article_name][word]
        
        if pre_calculated_count != just_calculated_count:
            print('At: ' + article_name)
            print('Pre: ' + str(pre_calculated_count))
            print('Just: '+ str(just_calculated_count))
            somethingwrong = True
    
    if not somethingwrong:
        print('Checked: ' + article_name)
