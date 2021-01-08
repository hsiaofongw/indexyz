from typing import Dict
import os
import json
import analysis
import cut
import pkuseg
from tabulate import tabulate

def start_on_local_mode(article_names, article_contents):

    article_names, article_contents = None, None
    
    # 加载分词器
    print('正在加载分词器……')
    seg = pkuseg.pkuseg()
    cutter = lambda sentence: seg.cut(sentence)
    
    print('开始分词……')
    article_words = cut.cut_articles(article_contents, cutter)

    print('开始统计词频……')
    article_stats = cut.do_stats(article_words) 

    print('统计所有出现过的单词……')
    all_term_stats = cut.make_all_term_stats(article_stats)

    all_term_vector = cut.make_all_term_vector(all_term_stats)

    print('开始计算 term_document_matrix ...')
    doc_matrix = cut.make_doc_matrix(article_stats, all_term_vector)

    print('开始构建索引……')
    article_indexes, term_indexes = cut.make_indexes(article_names, all_term_vector)

    print('构建有效 term 索引的子集……')
    selected_indexes = analysis.get_satisfied_indexes(term_indexes)

    print('修正 term_indexes 与 doc_matrix ...')

    print('修正前：')
    print('doc_matrix.shape: ' + str(doc_matrix.shape))
    print('term_indexes.shape: ' + str(term_indexes.shape))

    term_indexes, doc_matrix = analysis.correct_doc_matrix(
        term_indexes, 
        selected_indexes, 
        doc_matrix
    )

    print('修正后：')
    print('doc_matrix.shape: ' + str(doc_matrix.shape))
    print('term_indexes.shape: ' + str(term_indexes.shape))

    print('计算tf_idf ...')
    tf_idf = analysis.compute_tf_idf(doc_matrix)

    print('通过 SVD 方法进行 LSA 操作……')
    u, s, vh, doc_coords = analysis.lsa_analysis(tf_idf)

    term_to_col_index = analysis.make_term_to_col_index_mapper(term_indexes)

    print('开始REPL：')
    while True:
        sentence = input('>>> ')
        words = cutter(sentence)

        query_result = analysis.do_query_by_words(
            words,
            article_indexes,
            doc_coords,
            vh, term_to_col_index
        )

        print(tabulate(query_result, headers='keys', tablefmt='psql'))

    # print(term_indexes)
    # print(selected_indexes)
    # print(doc_matrix)
