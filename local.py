import sys
import pandas as pd
from tabulate import tabulate
from cut import cutter, make_articles_from_contents
from analysis import ArticleStats, LatentSemanticAnalyzer

def start_on_local_mode(article_names, article_contents):

    print('正在进行分词……')
    articles = make_articles_from_contents(article_names, article_contents)

    print('正在构造 term-document-matrix ...')
    stats = ArticleStats()
    stats.add_articles(articles)

    print('开始进行 SVD 操作……')
    lsa = LatentSemanticAnalyzer(stats.get_tf_idf())

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
