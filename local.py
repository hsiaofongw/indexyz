from typing import Dict
import os
import json
import analysis
import cut
import pkuseg
from tabulate import tabulate

def read_config(config_filename: str) -> Dict[str, any]:
    config_obj = None
    with open(config_filename, 'r') as f:
        config_obj = json.load(f)

    if config_obj is None:
        raise

    articles_dir = 'articles'
    articles_file = 'articles.json'
    load_articles_from = 'folder'

    if 'articles_dir' in config_obj:
        articles_dir = config_obj['articles_dir']
    
    if 'articles_file' in config_obj:
        articles_file = config_obj['articles_file']
    
    if 'load_articles_from_file_or_folder' in config_obj:
        load_articles_from = config_obj['load_articles_from_file_or_folder']

    return {
        'articles_dir': articles_dir,

        'articles_file': articles_file,

        'load_articles_from': load_articles_from
    }

    
    
def start_on_local_mode(data_folder: str):

    data_folder_full = os.path.abspath(data_folder)
    config_file_full_path = os.path.join(data_folder_full, 'config.json')

    this_config = read_config(config_file_full_path)

    load_articles_from = this_config['load_articles_from']

    articles_dir = this_config['articles_dir']
    article_names = None
    article_contents = None

    if load_articles_from == 'folder':

        articles_path_full = os.path.join(
            data_folder_full,
            articles_dir
        )

        print('文章文件夹为：' + articles_path_full)
        article_names, article_contents = cut.get_articles_from_folder(articles_path_full)
    
    elif load_articles_from == 'file':

        articles_file = this_config['articles_file']

        articles_file_path_full = os.path.join(
            data_folder_full,
            articles_file
        )

        print('文章读取自文件：' + articles_file_path_full)
        article_names, article_contents = cut.get_articles_from_file(articles_file_path_full)

    else:
        pass

    
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
