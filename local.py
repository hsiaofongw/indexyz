import sys
import pandas as pd
from tabulate import tabulate
from cut import cutter, make_articles_from_contents
from analysis import ArticleStats, Analyzer

def start_on_local_mode(article_names, article_contents):

    print("正在进行分词……")
    articles = make_articles_from_contents(article_names, article_contents)

    print("正在构造 term-document-matrix ...")
    stats = ArticleStats()
    stats.add_articles(articles)

    print("正在初始化分析工具……")
    analyzer = Analyzer(stats.get_term_document_matrix())

    print("开始计算 tf ...")
    analyzer.compute_tf()

    print("开始计算 idf ...")
    analyzer.compute_idf()

    print("开始计算 tf_idf ...")
    analyzer.compute_tf_idf()

    print("开始进行 SVD 操作……")
    analyzer.compute_svd_from_tf_idf()

    print("开始REPL：")
    while True:

        sentence = str()

        try:
            sentence = input(">>> ")
        except EOFError as e:
            print("Bye.")
            sys.exit()

        input_terms = cutter(sentence)

        new_row = stats.terms_to_new_row(input_terms)
        feature_input = analyzer.to_feature_coord(new_row)
        indexes, cosines = analyzer.find_nearest(feature_input)
        names = list(map(lambda i: stats.index_article[i], indexes))

        query_result = pd.DataFrame({
            "article_name": names,
            "cosine": cosines
        })

        print(tabulate(query_result, headers="keys", tablefmt="psql"))
