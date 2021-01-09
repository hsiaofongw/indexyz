import pkuseg
from typing import List
from tqdm import tqdm
from analysis import Article

seg = pkuseg.pkuseg()

def cutter(sentence: str):
    return seg.cut(sentence)

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
