import numpy as np
from scipy import sparse
from typing import Tuple, List
from tqdm import tqdm

class Article:

    def __init__(self, name: str = '', terms: List[str] = []):
        self.name = name
        self.terms = terms

# 根据每一篇 article 所包含的 terms 计算出 term_document_matrix
class NameSpace:
    
    def __init__(self):
        
        # article -> index
        self.article_index = dict()

        # index -> article
        self.index_article = dict()
        
        # term -> index
        self.term_index = dict()
        
        # index -> term
        self.index_term = dict()
        
        # term_document_matrix, arguments for scipy.sparse.csr_matrix
        self.tdm_data = list()
        self.tdm_indptr = [0]
        self.tdm_indices = list()
    
    # 给 term_document_matrix 添加新的一行
    def append_new_row_in_term_document_matrix(self, term_indices: List[str]):

        self.tdm_indices = self.tdm_indices + term_indices
        n_terms = len(term_indices)
        self.tdm_data = self.tdm_data + [1.0/n_terms for i in range(n_terms)]
        self.tdm_indptr.append(self.tdm_indptr[-1]+n_terms)
    
    # 为每一个 term 分配一个编号！
    def get_term_indexes(self, terms: List[str]):
        # 要返回的每一个 term 的编号
        indices_of_terms = list()
        index_of_this_term: int = 0

        for term in terms:

            # 如果这个 term 之前已经遇到过了
            if term in self.term_index:

                # 那么取出它的编号
                index_of_this_term = self.term_index[term]

            # 否则，假如说这个 term 是新的
            else:

                # 那么给它分配一个编号
                index_of_this_term = len(self.term_index)

                # 然后更新 term -> index 的对应关系
                self.term_index[term] = index_of_this_term

                # 然后更新 index -> term 的对应关系
                self.index_term[index_of_this_term] = term

            # 并且放入那个篮子里
            indices_of_terms.append(index_of_this_term)
        
        return indices_of_terms

    # 用一篇文章更新 term-document-matrix
    def add_article(self, article: Article):

        # 先为这篇文章分配一个编号
        id_of_this_article = len(self.article_index)

        # 然后更新 article_name -> index 的映射
        self.article_index[article.name] = id_of_this_article

        # 然后更新 index -> article_name 的映射
        self.index_article[id_of_this_article] = article.name

        # 并且正常计算 term_document_matrix 的新的一行
        that_indices = self.get_term_indexes(article.terms)

        # 并且正常更新 term_index 与 index_term
        self.append_new_row_in_term_document_matrix(that_indices)
        
    
    def add_articles(self, articles: List[Article]):
        for i in range(len(articles)):
            article = articles[i]
            self.add_article(article)
    
    # 获取 term-document-matrix 的稀疏矩阵表示
    def get_term_document_matrix(self) -> sparse.csr_matrix:
        
        data = self.tdm_data
        indices = self.tdm_indices
        indptr = self.tdm_indptr
        
        return sparse.csr_matrix((data, indices, indptr), dtype=np.float32)
    

# 提供一套分析工具用于从 term_document_matrix 中发掘信息
def compute_tf_idf(doc_matrix: sparse.csr_matrix) -> sparse.csr_matrix:

    # 假设这里的 doc_matrix 已经是行和为 1，那么就只需计算 idf

    # 首先找出所有非零元的坐标
    i, nonzero_cols, v = sparse.find(doc_matrix)

    # 然后统计每一列的非零元的个数
    nonzero_cols, nonzero_col_appearences = np.unique(nonzero_cols, return_counts=True)

    # 有些列可能全为 0，所以
    indicator = np.zeros(shape=(1, doc_matrix.shape[1],), dtype=np.float32)
    indicator[0, nonzero_cols] = nonzero_col_appearences[:]
    n_articles = doc_matrix.shape[0]
    indicator = np.log(n_articles/indicator)

    # 此处为按元素乘
    tf_idf = doc_matrix.multiply(indicator)

    return tf_idf
    