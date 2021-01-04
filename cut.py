import pkuseg
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    
    return (article_names, article_contents)

def cut_articles(articles: [str]) -> [[str]]:
    article_words = list()
    for i in tqdm(range(len(article_contents))):
        article_content = article_contents[i]
        words = seg.cut(article_content)
        article_words.append(words)
    
    return article_words

def counter(l: list) -> dict:
    stats = dict()
    for i in range(len(l)):
        item = l[i]
        if item not in stats:
            stats[item] = 1
        else:
            stats[item] = stats[item] + 1
    
    return stats

def do_stats(article_words: [[str]]) -> [dict[str, int]]:
    article_stats = list()
    for i in tqdm(range(len(article_words))):
        article_stats.append(counter(article_words[i]))
    
    return article_stats

def make_all_term_stats(article_stats: [dict]) -> dict[str, int]:
    all_term_stats = dict()
    for i in tqdm(range(len(article_stats))):
        article_stat = article_stats[i]
        for word, count in article_stat.items():
            if word not in all_term_stats:
                all_term_stats[word] = count
            else:
                all_term_stats[word] = all_term_stats[word] + count
    
    return all_term_stats

def make_all_term_vector(all_term_stats: dict) -> [str]:
    all_term_vector = list(all_term_stats.keys())
    return all_term_vector

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

def make_article_indexes(article_names: [str]) -> pd.DataFrame:
    article_indexes = pd.DataFrame({
        'article_name': article_names,
        'row_num_in_doc_matrix': range(len(article_names))
    })

    return article_indexes

def make_term_indexes(all_term_vector: [str]) -> pd.DataFrame:
    term_indexes = pd.DataFrame({
        'term': all_term_vector,
        'col_num_in_doc_matrix': range(len(all_term_vector))
    })

    return term_indexes

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