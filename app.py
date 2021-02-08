import pymongo
from analysis import Article
from analysis import NameSpace as QuerySpace
from analysis import compute_tf_idf
from scipy.sparse.linalg import svds
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, status
from model import *

from datetime import datetime

global_state = dict()
global_state['count'] = 0

def get_db():
    dbname = "indexyz"
    client = pymongo.MongoClient(f"mongodb://127.0.0.1/{dbname}")
    db = client.indexyz
    return db

app = FastAPI()

@app.get("/status")
async def get_status():

    global global_state
    global_state['count'] = global_state['count'] + 1

    return {
        "count": global_state['count']
    }

@app.post("/namespaces/", status_code=status.HTTP_201_CREATED)
async def create_namespace(ns: NameSpace):

    db = get_db()
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

    u = u.tolist()
    s = s.tolist()
    vh = vh.tolist()
    term_index = query_space.term_index
    index_term = query_space.index_term
    article_index = query_space.article_index
    index_article = query_space.index_article

    global global_state

    global_state['query_object']['u'] = u
    global_state['query_object']['s'] = s
    global_state['query_object']['vh'] = vh
    global_state['query_object']['term_index'] = term_index
    global_state['query_object']['index_term'] = index_term
    global_state['query_object']['article_index'] = article_index
    global_state['query_object']['index_article'] = index_article

    now = datetime.utcnow()
    ns['created_at'] = now

    namespace_id = namespaces.insert_one(ns).inserted_id

    return {
        "namespace_object_id": str(namespace_id)
    }

@app.delete("/namespaces/{name_of_namespace}")
async def delete_namespace():

    return {
        "message": "namespace deleted.",
        "name_of_namespace": "name"
    }

@app.get("/namespaces/{name_of_namespace}")
async def make_query_by_words(name_of_namespace: str, query: Query):

    result = {
        "words": query.words,
        "sentence": query.sentence,
        "name_of_entry": ["aa", "b1", "c", "d2", "e4"],
        "cosine": [0.9, 0.8, 0.7, 0.6, 0.4]
    }

    return result