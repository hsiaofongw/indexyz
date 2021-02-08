from loader import read_articles_from_txt
from cut import cutter, make_articles_from_contents
from analysis import Article, NameSpace, Analyzer

article1 = Article()
article1.name = 'article1'
article1.terms = ['aa', 'ab', 'aa', 'aa', 'ac']

article2 = Article()
article2.name = 'article2'
article2.terms = ['bc', 'bc', 'bd', 'aa', 'ac', 'ac']

articles = [article1, article2]

print('构建 Term-Document-Matrix')
stats = NameSpace()
stats.add_articles(articles)

term_document_matrix = stats.get_term_document_matrix()

print('计算 Tf-Idf')
lsa = Analyzer(term_document_matrix)
lsa.compute_tf_idf()