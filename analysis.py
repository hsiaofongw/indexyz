import numpy as np
from scipy.sparse.linalg import svds
from typing import Tuple, List
from scipy.sparse import csr_matrix
from random import randint
from tqdm import tqdm

class Article:

    def __init__(self, name: str = '', terms: List[str] = []):
        self.name = name
        self.terms = terms

# 主要使命，根据每一篇 article 所包含的 terms 计算出 term_document_matrix
class ArticleStats:
    
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
        self.tdm_indptr.append(len(term_indices))
    
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

    def update_term_document_matrix_row(self, row_index: int, term_indices: List[int]) -> None:

        # 首先来看下是个什么样的情况
        length_of_updating_row = self.tdm_indptr[row_index+1] - self.tdm_indptr[row_index]
        length_of_new_row = len(term_indices)

        # 如果新的这行比原来那行长
        if length_of_new_row > length_of_updating_row:

            # 那么就要开辟一些空间
            spaces_to_extend = length_of_new_row - length_of_updating_row
            for i in range(spaces_to_extend):
                self.tdm_indices.insert(self.tdm_indptr[row_index]+1, 0)
                self.tdm_data.insert(self.tdm_indptr[row_index]+1, 1)

        # 如果新的这行比原来那行短
        elif length_of_new_row < length_of_updating_row:

            # 那么就要压缩一些空间
            spaces_to_squeeze = length_of_updating_row - length_of_new_row
            i = len(self.tdm_indptr) - 1
            while i > row_index:
                self.tdm_indptr[i] = self.tdm_indptr[i] - spaces_to_squeeze
                i = i - 1
            
            del self.tdm_indices[(len(self.tdm_indices)-spaces_to_squeeze):len(self.tdm_indices)]
            del self.tdm_data[(len(self.tdm_data)-spaces_to_squeeze):len(self.tdm_data)]

        # 如果新的这行和原来那行一样长
        else:

            # 那就没那么多花样
            pass
        
        # 现在正式开始更新 indices 
        self.tdm_indices[self.tdm_indptr[row_index]:self.tdm_indptr[row_index+1]] = term_indices

        # 如果新的这行是空的
        if self.tdm_indptr[row_index] == self.tdm_indptr[row_index+1]:
            # 那就干脆删除这行吧！
            del self.tdm_indptr[row_index]

    
    # 用一篇文章更新 term-document-matrix
    def add_article(self, article: Article):
        
        if article.name in self.article_index:

            # 这时这篇文章已经添加过了，所以我们需要更新 term_document_matrix
            # 怎么更新呢？

            # 首先确定要更新 term_document_matrix 的哪一行
            row_index = self.article_index[article.name]

            # 然后确定那一行要更新成什么样的值
            that_indices = self.get_term_indexes(article.terms)

            # 确定好后就去更新吧！
            self.update_term_document_matrix_row(row_index, that_indices)

        else:

            # 这说明这篇文章之前还没有被添加进来过

            # 那就先为这篇文章分配一个编号
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
    def get_term_document_matrix(self):
        
        data = self.tdm_data
        indices = self.tdm_indices
        indptr = self.tdm_indptr
        
        return csr_matrix((data, indices, indptr), dtype=int)
    
    # term_indexes: { 'a': 10, 'b': 2, 'c': 9, 'e': 7 }
    # terms:   [ 'c', 'a',  'a', 'b', 'b', 'e', 'b' ]
    # indexes: [  9,  10,   10,   2,   2,   7,   2  ]
    def terms_to_indexes(self, terms: List[str]) -> List[int]:
        
        nterms = len(self.term_index)
        indexes = list()
        for term in terms:
            indexes.append(self.term_index.get(term, randint(0, nterms-1)))
        
        return indexes
    
    def terms_to_new_row(self, terms: List[str]) -> np.ndarray:
        
        indices = np.array(self.terms_to_indexes(terms))
        data = np.ones(indices.shape)
        indptr = [0, indices.shape[0]]
        
        new_row = csr_matrix((data, indices, indptr), shape=(1, len(self.term_index), ))
        
        return new_row.toarray()

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
    def compute_tf(self) -> None:
        
        doc_matrix = self.doc_matrix
        self.tf = doc_matrix.multiply(1/doc_matrix.sum(axis=1))
        
    # compute inverse document frequency
    def compute_idf(self) -> None:
        
        doc_matrix = self.doc_matrix
        
        # (n_x[0], n_y[0]), (n_x[1], n_y[2]), ... 表示非零元素坐标
        n_x, n_y = doc_matrix.nonzero()
        
        # 去重
        s = set()
        for j in range(len(n_x)):
            s.add((n_x[j], n_y[j],))
        
        n_y = list()
        for x, y in s:
            n_y.append(y)
        
        col_indexes, non_zeros_count = np.unique(n_y, return_counts=True)

        self.idf = np.log(doc_matrix.shape[0]/non_zeros_count)
            
    def compute_tf_idf(self) -> None:
        
        tf = self.compute_tf()
        idf = self.compute_idf()
        
        self.tf_idf = tf.multiply(idf)
        
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
