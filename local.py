import pkuseg
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.sparse.linalg import svds
from typing import Tuple, List
from scipy.sparse import csr_matrix
from random import randint
from loader import read_articles_from_txt
from tqdm import tqdm

class Article:
        
    def __init__(self, name: str = '', terms: List[str] = []):
        self.name = name
        self.terms = terms

class ArticleStats:
    
    def __init__(self):
        
        self.article_names = list()
        
        # term -> index
        self.term_index = dict()
        
        # index -> term
        self.index_term = dict()
        
        # term_document_matrix, arguments for scipy.sparse.csr_matrix
        self.term_document_matrix_data = list()
        self.term_document_matrix_indptr = [0]
        self.term_document_matrix_indices = list()
    
    # 用一篇文章更新 term-document-matrix
    def add_article(self, article: Article):
        
        self.article_names.append(article.name)
        
        for term in article.terms:
            
            index_of_this_term = self.term_index.setdefault(
                term, 
                len(self.term_index)
            )
            
            self.index_term[index_of_this_term] = term
            
            self.term_document_matrix_indices.append(index_of_this_term)
            self.term_document_matrix_data.append(1)
        
        self.term_document_matrix_indptr.append(len(self.term_document_matrix_indices))
    
    def add_articles(self, articles: List[Article]):
        for i in tqdm(range(len(articles))):
            article = articles[i]
            self.add_article(article)
    
    # 获取 term-document-matrix 的稀疏矩阵表示
    def get_term_document_matrix(self):
        
        data = self.term_document_matrix_data
        indices = self.term_document_matrix_indices
        indptr = self.term_document_matrix_indptr
        
        return csr_matrix((data, indices, indptr), dtype=int)
    
    # 计算 tf
    def get_tf(self):
        
        doc_matrix = self.get_term_document_matrix()
        tf = doc_matrix.multiply(1/doc_matrix.sum(axis=1))
        
        return tf
    
    # 计算 idf
    def get_idf(self):
        
        doc_matrix = self.get_term_document_matrix()
        
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
        idf = np.log(doc_matrix.shape[0]/non_zeros_count)
        
        return idf
            
    # 计算 tf-idf
    def get_tf_idf(self):
        
        tf = self.get_tf()
        idf = self.get_idf()
        
        tf_idf = tf.multiply(idf)
        
        return tf_idf
    
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

class LatentSemanticAnalyzer:
    
    def __init__(self, term_document_matrix):
        self.doc_matrix = term_document_matrix.astype(np.float64)
        self.perform_svd()
        
    def perform_svd(self, k = 0):
        if k == 0:
            k = min(self.doc_matrix.shape)-1
        
        u, s, vh = svds(self.doc_matrix, k = k)
        
        self.svd_u = u
        self.svd_s = s
        self.svd_vh = vh
        
        self.doc_coords = np.matmul(self.svd_u, np.diag(self.svd_s))
    
    def to_feature_coord(self, origin_coord):
        
        origin_coord = np.reshape(origin_coord, newshape=(1, self.svd_vh.T.shape[0],))
        feature_coord = np.matmul(origin_coord, self.svd_vh.T)
        
        return feature_coord[0, :]
    
    @classmethod
    def cos_of_two_vector(cls, v1: np.ndarray, v2: np.ndarray):
        return abs(np.inner(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
    
    # (indexes, cosine_values, )
    def find_nearest(self, input_feature: np.ndarray):
        
        cosine_values = list()
        
        for d in self.doc_coords:
            cos_value = LatentSemanticAnalyzer.cos_of_two_vector(d, input_feature)
            cosine_values.append(cos_value)
        
        indexes = list(range(len(self.doc_coords)))
        indexes.sort(
            reverse = True, 
            key = lambda i: cosine_values[i]
        )
        
        cosine_values = list(map(lambda i: cosine_values[i], indexes))
        
        return ( indexes, cosine_values, )

def make_articles_from_contents(article_names: List[str], article_contents: List[str]):
        
    seg = pkuseg.pkuseg()
    cutter = lambda sentence: seg.cut(sentence)

    articles = list()
    for i in tqdm(range(len(article_names))):
        articles.append(Article(
            name = article_names[i],
            terms = cutter(article_contents[i])
        ))
    
    return articles

def start_on_local_mode(article_names, article_contents):

    print('正在进行分词……')
    articles = make_articles_from_contents(article_names, article_contents)

    print('正在构造 term-document-matrix ...')
    stats = ArticleStats()
    stats.add_articles(articles)

    print('开始进行 SVD 操作……')
    lsa = LatentSemanticAnalyzer(stats.get_tf_idf())

    print('加载其他必要的模块……')
    seg = pkuseg.pkuseg()
    cutter = lambda sentence: seg.cut(sentence)

    print('开始REPL：')
    while True:

        sentence = str()

        try:
            sentence = input('>>> ')
        except EOFError as e:
            print('Bye.')
            sys.exit()

        input_terms = cutter(sentence)

        new_row = stats.terms_to_new_row(input_terms)
        feature_input = lsa.to_feature_coord(new_row)
        indexes, cosines = lsa.find_nearest(feature_input)
        names = list(map(lambda i: stats.article_names[i], indexes))

        query_result = pd.DataFrame({
            'article_name': names,
            'cosine': cosines
        })

        print(tabulate(query_result, headers='keys', tablefmt='psql'))