import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import pkuseg
from typing import Tuple

def load_data_from_file(
    term_indexes_filename: str,
    article_indexes_filename: str,
    term_doc_matrix_filename: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    
    # 加载 term_doc_matrix 的行标、列标与 term, article 的对应关系
    term_indexes = pd.read_csv(term_indexes_filename)
    articles = pd.read_csv(article_indexes_filename)

    # 加载 term_doc_matrix
    doc_matrix = np.loadtxt(term_doc_matrix_filename, delimiter=',')
    
    return (
        term_indexes,
        articles,
        doc_matrix,
    )

# 这个函数判断一个 word(term) 里面有没有标点符号
def no_punctuations_or_letters_or_digits_Q(word):
    
    # if word is not str, then False
    if not (type(word) is str):
        return False
    
    # if word has digits, letters or punctuations
    result = re.search(
        '[' +
        '\u2000-\u206F' +  # 一般的符号
        '\u0000-\u002F' +  # 符号
        '\u003A-\u0040' +  # 符号
        '\u005B-\u0060' +  # 符号
        '\u007B-\u007F' +  # 符号
        '\uFF00-\uFFEF' +  # 全角字符
        '\u3000-\u303F' +  # 中文标点
        ']',
        word
    )
    
    return (result is None)

# 这个函数判断一个 word(term) 的长度是不是大于或等于 2
def length_equal_or_greater_than_2_Q(word):
    return len(word) >= 2
    