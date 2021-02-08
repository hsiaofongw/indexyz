import pymongo
from analysis import Article
from analysis import NameSpace as QuerySpace
from analysis import compute_tf_idf
from scipy.sparse.linalg import svds
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, status, Response
from model import *
from analysis import Searcher
import numpy as np
from typing import Dict
from math import isnan

from datetime import datetime

class GlobalState:

    def __init__(self) -> None:
        self.started_at = datetime.utcnow()
        self.request_counts = 0
        self.query_objects: List[Searcher] = []

    def get_db(self):
        dbname = "indexyz"
        client = pymongo.MongoClient(f"mongodb://127.0.0.1/{dbname}")
        db = client.indexyz
        return db

state = GlobalState()

app = FastAPI()

@app.get("/status")
async def get_status():

    return {
        "started_at": str(state.started_at)
    }

@app.post("/namespaces/", status_code=status.HTTP_201_CREATED)
async def create_namespace(ns: NameSpace):

    db = state.get_db()
    namespaces = db.namespaces
    ns = jsonable_encoder(ns)

    query_space = QuerySpace()
    for entry in ns['entries']:
        article = Article()
        article.name = entry['name_of_entry']
        article.terms = entry['terms']
        query_space.add_article(article)

    doc_matrix = query_space.get_term_document_matrix()
    tf_idf = compute_tf_idf(doc_matrix)
    u, s, vh = svds(tf_idf, k=min(tf_idf.shape)-1)

    term_index = query_space.term_index
    index_term = query_space.index_term
    article_index = query_space.article_index
    index_article = query_space.index_article

    query_object = Searcher(**{
        'svd_u': u,
        'svd_s': s,
        'svd_vh': vh,
        'term_index': term_index,
        'index_term': index_term,
        'article_index': article_index,
        'index_article': index_article
    })

    state.query_objects.append(query_object)

    now = datetime.utcnow()
    ns['created_at'] = now

    namespace_id = namespaces.insert_one(ns).inserted_id

    return {
        "namespace_object_id": str(namespace_id)
    }

@app.delete("/namespaces/{name_of_namespace}")
async def delete_namespace_by_name(name_of_namespace: str):

    db = state.get_db()
    namespaces = db.namespaces
    result = namespaces.delete_one({
        'name_of_namespace': name_of_namespace
    })

    return {
        "message": "namespace deleted.",
        "deleted_count": result.deleted_count,
        "name_of_namespace": name_of_namespace
    }

@app.post("/")
async def make_query_by_words(words_query: WordsQuery, response: Response):

    # pass
    if len(state.query_objects) == 0:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            'message': 'no query object available, please upload an namespace.'
        }
    
    query_object = state.query_objects[-1]
    q = query_object.make_query(words_query.words)
    indexes, cosines =  query_object.sort_index_by_cosine_similarity(q)
    name_of_entries = [query_object.index_article[i] for i in indexes.tolist()]
    cosines = cosines.tolist()
    cosines = [0 if isnan(x) else x for x in cosines]
    
    response.status_code = status.HTTP_200_OK

    result = {
        "words": words_query.words,
        "name_of_entries": name_of_entries,
        "cosines": cosines
    }

    return result