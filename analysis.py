import numpy as np
from numpy.core.numeric import indices
from scipy.sparse.linalg import svds
from typing import Tuple, List
from scipy.sparse import csr_matrix
from random import randint
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
        self.tdm_data = self.tdm_data + [1 for i in range(len(term_indices))]
        self.tdm_indptr.append(self.tdm_indptr[-1]+len(term_indices))
    
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
        for i in tqdm(range(len(articles))):
            article = articles[i]
            self.add_article(article)
    
    # 获取 term-document-matrix 的稀疏矩阵表示
    def get_term_document_matrix(self) -> csr_matrix:
        
        data = self.tdm_data
        indices = self.tdm_indices
        indptr = self.tdm_indptr
        
        return csr_matrix((data, indices, indptr), dtype=int)
    

# 提供一套分析工具用于从 term_document_matrix 中发掘信息
class Analyzer:

    # term document matrix
    doc_matrix: csr_matrix = csr_matrix((0, 0))

    # term frequency
    tf: csr_matrix = csr_matrix((0, 0))

    # inverse document frequency
    idf: np.ndarray = np.array([])

    # tf * idf
    tf_idf: csr_matrix = csr_matrix((0, 0))

    # svd values
    svd_u: np.ndarray = np.array([])
    svd_s: np.ndarray = np.array([])
    svd_vh: np.ndarray = np.array([])
    doc_coords: np.ndarray = np.array([])

    def __init__(self, term_document_matrix: csr_matrix):
        self.doc_matrix = term_document_matrix
    
    # compute term frequency 
    def __compute_tf(self) -> None:
        
        doc_matrix = self.doc_matrix.toarray()
        self.tf = doc_matrix / np.atleast_2d(doc_matrix.sum(axis=1)).T
        
    # compute inverse document frequency
    def __compute_idf(self) -> None:

        # 对每个 term, 我要知道有多少篇文章出现了这个 term
        
        doc_matrix = self.doc_matrix.toarray()

        indicator = np.ones_like(doc_matrix, dtype=np.float32)
        indicator[doc_matrix == 0] = 0
        
        idf = np.sum(indicator, axis=0)
        n_articles = doc_matrix.shape[0]
        self.idf = np.log(n_articles/idf)

    def compute_tf_idf(self) -> None:

        self.__compute_tf()
        self.__compute_idf()
        
        self.tf_idf = self.tf * self.idf
        
    def compute_svd_from_tf_idf(self, k = 0) -> None:

        tf_idf = self.tf_idf

        if k == 0:
            k = min(tf_idf.shape)-1
        
        u, s, vh = svds(self.doc_matrix, k = k)
        
        doc_coords = np.matmul(self.svd_u, np.diag(self.svd_s))
        u, s, vh, doc_coords

        self.svd_u, self.svd_s, self.svd_vh = u, s, vh
        self.doc_coords = doc_coords
    
    def to_feature_coord(self, origin_coord: np.ndarray) -> np.ndarray:
        
        origin_coord = np.reshape(origin_coord, newshape=(1, self.svd_vh.T.shape[0],))
        feature_coord = np.matmul(origin_coord, self.svd_vh.T)
        
        return feature_coord[0, :]
    
    @classmethod
    def cos_of_two_vector(cls, v1: np.ndarray, v2: np.ndarray):
        return abs(np.inner(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
    
    # (indexes, cosine_values, )
    def find_nearest(self, input_feature: np.ndarray) -> Tuple[
        List[int], List[float]
    ]:
        
        cosine_values = list()
        
        for d in self.doc_coords:
            cos_value = Analyzer.cos_of_two_vector(d, input_feature)
            cosine_values.append(cos_value)
        
        indexes = list(range(len(self.doc_coords)))
        indexes.sort(
            reverse = True, 
            key = lambda i: cosine_values[i]
        )
        
        cosine_values = list(map(lambda i: cosine_values[i], indexes))
        
        return ( indexes, cosine_values, )
