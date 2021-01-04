import pkuseg
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import AnyStr, Callable, Dict, Tuple

seg = pkuseg.pkuseg()

# tuple.1 is article_names, tuple.2 is article_contents
def get_articles_from_folder(foldername: str) -> ([str], [str]):

    abs_dir_path = os.path.abspath(foldername)
    article_names = os.listdir(abs_dir_path)
    article_contents = list()
    for article_name in article_names:
        full_path = os.path.join(abs_dir_path, article_name)
        file_name = full_path
        file = open(file_name, 'r')
        article_content = file.read()
        file.close()
        article_contents.append(article_content)
    
    return (
        article_names, 
        article_contents,
    )

# article_words 中的每一个元素是一个单词组成的列表，对应一篇文章
def cut_articles(
    article_contents: [str], 
    cutter: Callable[[str], AnyStr]
) -> [[str]]:
    article_words = list()
    for i in tqdm(range(len(article_contents))):
        article_content = article_contents[i]
        words = seg.cut(article_content)
        article_words.append(words)
    
    return article_words


# 统计一个列表中各个元素出现的次数
def counter(l: list) -> dict:
    stats = dict()
    for i in range(len(l)):
        item = l[i]
        if item not in stats:
            stats[item] = 1
        else:
            stats[item] = stats[item] + 1
    
    return stats

# 统计每篇文章的词频
def do_stats(article_words: [[str]]) -> [Dict[str, int]]:
    article_stats = list()
    for i in tqdm(range(len(article_words))):
        article_stats.append(counter(article_words[i]))
    
    return article_stats

# 统计每一个出现过的单词的出现频数
def make_all_term_stats(article_stats: [dict]) -> Dict[str, int]:
    all_term_stats = dict()
    for i in tqdm(range(len(article_stats))):
        article_stat = article_stats[i]
        for word, count in article_stat.items():
            if word not in all_term_stats:
                all_term_stats[word] = count
            else:
                all_term_stats[word] = all_term_stats[word] + count
    
    return all_term_stats

# 将所有出现过的单词列成一个表
def make_all_term_vector(all_term_stats: dict) -> [str]:
    all_term_vector = list(all_term_stats.keys())
    return all_term_vector

# 计算 term_doc_matrix
def make_doc_matrix(article_stats: [dict], all_term_vector: [str]) -> [[int]]:
    doc_matrix = list()
    for i in tqdm(range(len(article_stats))):
        stat_of_this_article = article_stats[i]
        doc_matrix_row = list()
        for term in all_term_vector:
            if term in stat_of_this_article:
                count = stat_of_this_article[term]
            else:
                count = 0
            doc_matrix_row.append(count)
        
        doc_matrix.append(doc_matrix_row)
    
    return doc_matrix

# 构建文章名索引与单词索引
def make_indexes(
    article_names: [str],
    all_term_vector: [str]
) -> Tuple[pd.DataFrame, pd.DataFrame,]:
    article_indexes = pd.DataFrame({
        'article_name': article_names,
        'row_num_in_doc_matrix': range(len(article_names))
    })

    term_indexes = pd.DataFrame({
        'term': all_term_vector,
        'col_num_in_doc_matrix': range(len(all_term_vector))
    })

    return (
        article_indexes,
        term_indexes,
    )

# 将资料存储到磁盘
def save_all_to_file(

    article_indexes: pd.DataFrame,
    article_indexes_filename: str,

    term_indexes: pd.DataFrame,
    term_indexes_filename: str,

    doc_matrix: [[int]],
    doc_matrix_filename: str

) -> None:

    article_indexes.to_csv(article_indexes_filename, index=False)
    term_indexes.to_csv(term_indexes_filename, index=False)

    np.savetxt(
        fname = doc_matrix_filename,
        X = np.array(doc_matrix).astype(int),
        delimiter = ',',
        newline = '\n',
        fmt = '%u'
    )