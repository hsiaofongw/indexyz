import numpy as np
import unittest
from analysis import Article, TermDocumentMatrix, compute_tf_idf

class TestAnalyzer(unittest.TestCase):

    def setUp(self):

        # test data
        # ========
        article1 = Article()
        article1.name = 'article1'
        article1.terms = ['aa', 'ab', 'aa', 'aa', 'ac']

        article2 = Article()
        article2.name = 'article2'
        article2.terms = ['bc', 'bc', 'bd', 'aa', 'ac', 'ac']

        articles = [article1, article2]

        true_tdm = np.array([
            [3, 1, 1, 0, 0],
            [1, 0, 2, 2, 1]
        ], dtype=np.float32)

        true_tdm[0, :] = true_tdm[0, :] / true_tdm[0, :].sum()
        true_tdm[1, :] = true_tdm[1, :] / true_tdm[1, :].sum()

        true_idf = np.array([
            [0.0, 0.6931471806, 0.0, 0.6931471806, 0.6931471806]
        ], dtype=np.float32)

        true_tf_idf = true_tdm * true_idf
        # ========

        self.true_doc_matrix = true_tdm
        self.true_tf_idf = true_tf_idf
        self.articles = articles

    def test_get_term_document_matrix(self):
        ns = TermDocumentMatrix()
        ns.add_articles(self.articles)
        doc_matrix = ns.get_term_document_matrix()
        doc_matrix = doc_matrix.toarray()
        max_error = np.max(np.abs(doc_matrix - self.true_doc_matrix))
        epsilon = 1e-4
        self.assertAlmostEqual(max_error, 0.0, delta=epsilon)

    def test_compute_tf_idf(self):
        ns = TermDocumentMatrix()
        ns.add_articles(self.articles)
        doc_matrix = ns.get_term_document_matrix()
        tf_idf = compute_tf_idf(doc_matrix)
        tf_idf = tf_idf.toarray()

        epsilon = 1e-4
        max_error = np.max(np.abs(tf_idf - self.true_tf_idf))
        self.assertAlmostEqual(max_error, 0.0, delta=epsilon)
    
if __name__ == '__main__':
    # unittest.main(argv=['first-arg-is-ignored', '-v'])
    unittest.main(verbosity=4)