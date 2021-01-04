import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import pkuseg
from typing import Tuple

# 读入必要的数据文件，一般只运行一次即可
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

# 返回符合条件的 term 的序号的列表
def get_satisfied_indexes(term_indexes: pd.DataFrame) -> [int]:

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

    # 按照上述规则进行筛选
    all_indexes = list(range(0, term_indexes.shape[0]))

    # 检验规则
    def pass_rule(x: str) -> bool:
        
        if not no_punctuations_or_letters_or_digits_Q(x):
            return False
        
        if not length_equal_or_greater_than_2_Q(x):
            return False
        
        return True
    
    selected_indexes = list(filter(
        lambda i: pass_rule(term_indexes.iloc[i, 0]),
        all_indexes
    ))

    return selected_indexes

# 选取 terms 的子集后，对 doc_matrix 和 term 与 index 的对应关系做相应的改动        
# 返回一个新的 term_indexes 与新的 doc_matrix
def correct_doc_matrix(
    term_indexes: pd.DataFrame, 
    selected_indexes: [int],
    doc_matrix: [[int]]
) -> Tuple[pd.DataFrame, [[int]]]:

    # 选取原先的 terms 的子集
    selected_terms = term_indexes.iloc[selected_indexes, :]
    doc_matrix = doc_matrix[:, selected_indexes]

    # 是否没有 doc 的坐标全为 0
    row_num_that_all_zeros = (np.sum(doc_matrix, axis = 1) == 0)

    # 如果有某一行全为 0, 那么就在那一行随机添加一个 1
    col_num_to_set_1 = random.randint(0, doc_matrix.shape[1]-1)

    doc_matrix[row_num_that_all_zeros, col_num_to_set_1] = 1

    # 计算子 term_doc_matrix 的角标与 term_index 的对应关系
    col_num_in_new_doc_matrix = pd.DataFrame({
        'col_num_in_new_doc_matrix': np.array(range(0, doc_matrix.shape[1]))
    })

    # 更新索引并合并
    selected_terms = selected_terms.reset_index(drop=True)
    selected_terms = pd.concat(
        objs = [selected_terms, col_num_in_new_doc_matrix],
        axis = 1
    )

    return (
        selected_terms,
        doc_matrix,
    )

# 计算tf-idf
def compute_tf_idf(doc_matrix: [[int]]) -> np.ndarray:

    idf = np.log(np.divide(
        doc_matrix.shape[0],
        np.count_nonzero(doc_matrix, axis = 0)
    ))

    tf_idf = doc_matrix * idf 

    normalization_factor = np.tile(np.sum(tf_idf, axis = 1), reps=[ doc_matrix.shape[1], 1 ]).T

    tf_idf = tf_idf/normalization_factor

    return tf_idf


# 通过 svd 进行 lsa 分析
def lsa_analysis(tf_idf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # 先进行 svd 操作
    u, s, vh = np.linalg.svd(tf_idf, full_matrices=True)

    # 再将 原始空间 的 doc 投影到 特征空间
    matS = np.zeros(shape=tf_idf.shape)
    matS[0:s.shape[0], 0:s.shape[0]] = np.diag(s)
    doc_coords = np.matmul(u, matS)

    return (
        u, s, vh, doc_coords,
    )


# 接受用户输入
query_terms = seg.cut('多边形')
query_terms

# 构建 query_row, 这个 query_row 相当于一个 doc
query_row = np.array(range(doc_matrix.shape[1]))
query_row[:] = 0
query_row = np.reshape(a = query_row, newshape = (1, query_row.shape[0]))
for term in query_terms:
    if term in term_to_col_index:
        col_index = term_to_col_index[term]
        query_row[0, col_index] = 1

# 是否可以进行搜索
np.sum(query_row) > 0

# 将 query_row 投影到 lsa 特征空间
query_coord = np.matmul(query_row, vh.T)

query_coord = query_coord[:, 0:doc_coords.shape[0]]
doc_coords = doc_coords[:, 0:doc_coords.shape[0]]

# 用这个函数计算两个向量的余弦值
def cos_of_two_vector(x1, x2):
    n_x1 = np.linalg.norm(x1)
    n_x2 = np.linalg.norm(x2)
    inner_prod = np.abs(np.sum(x1 * x2))
    
    if n_x1 * n_x2 == 0:
        n_x1 = n_x1 + (1E-10)
        n_x2 = n_x2 + (1E-10)
        
    return inner_prod / (n_x1 * n_x2)

# 计算 query_coord (它是查询关键字组成的 word_vector 的 lsa 特征空间的投影) 与
# doc 在特征空间中的投影的余弦值，以此来判断接近程度
cos_values = list()
for i in range(doc_coords.shape[0]):
    doc_coord = doc_coords[i, :]
    cos_value = cos_of_two_vector(query_coord, doc_coord)
    cos_values.append(cos_value)

# 更新匹配度那一列
articles['match_val'] = cos_values

# 展示搜索结果
articles.sort_values(by = 'match_val', ascending=False)