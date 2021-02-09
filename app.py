from bson.objectid import ObjectId
import pymongo
from pymongo.son_manipulator import NamespaceInjector
from analysis import Article
from analysis import TermDocumentMatrix
from analysis import compute_tf_idf
from scipy.sparse.linalg import svds
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, status, Response
from model import *
from analysis import Searcher
from math import isnan
from cut import cutter

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

@app.post("/searchers/", status_code=status.HTTP_201_CREATED)
async def create_searcher(searcher_body: SearcherModel, response: Response):
    searcher_body.created_at = datetime.utcnow()

    db = state.get_db()
    namespaces_collection = db.namespaces
    ns_data = namespaces_collection.find_one({
        '_id': ObjectId(searcher_body.base_namespace_id)
    })

    if ns_data is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            'message': 'given namespace_id not found'
        }

    ns = NameSpaceModel(**ns_data)
    term_document_matrix_creator = TermDocumentMatrix()
    for entry in ns.entries:
        article = Article()
        article.name = entry.name_of_entry
        article.terms = entry.terms
        term_document_matrix_creator.add_article(article)
    
    doc_matrix = term_document_matrix_creator.get_term_document_matrix()
    tf_idf = compute_tf_idf(doc_matrix)
    u, s, vh = svds(tf_idf, k=min(tf_idf.shape)-1)
    searcher = Searcher(
        u, s, vh, 
        term_document_matrix_creator.term_index,
        term_document_matrix_creator.index_term,
        term_document_matrix_creator.article_index,
        term_document_matrix_creator.index_article
    )

    searcher_data = searcher.dumps()
    searcher_object = {
        'base_namespace_id': ObjectId(searcher_body.base_namespace_id),
        'created_at': datetime.utcnow(),
        'searcher_data': searcher_data
    }

    insert_result = db.searchers.insert_one(searcher_object)
    return {
        'inserted_id': str(insert_result.inserted_id)
    }

@app.get("/searchers")
async def list_searchers(status_code=status.HTTP_200_OK):
    db = state.get_db()
    searcher_objects_cursor = db.searchers.find({})
    results = []
    for searcher_object in searcher_objects_cursor:
        searcher_id = searcher_object['_id']
        base_namespace_id = searcher_object['base_namespace_id']
        base_namespace_name = db.namespaces.find_one({'_id': base_namespace_id})['name_of_namespace']

        results.append({
            '_id': str(searcher_id),
            'base_namespace_id': str(base_namespace_id),
            'base_namespace_name': base_namespace_name
        })
    
    return results

@app.post("/namespaces/", status_code=status.HTTP_201_CREATED)
async def create_namespace(ns: NameSpaceModel):

    db = state.get_db()
    namespaces = db.namespaces

    for entry in ns.entries:
        if len(entry.content) >= 1:
            entry.terms = entry.terms + cutter(entry.content)

    now = datetime.utcnow()
    ns = jsonable_encoder(ns)
    ns['created_at'] = now

    namespace_id = namespaces.insert_one(ns).inserted_id

    return {
        "namespace_object_id": str(namespace_id)
    }

@app.get("/namespaces")
async def list_namespaces():

    db = state.get_db()
    namespaces = db.namespaces.find({})

    results = []
    for x in namespaces:
        results.append({
            'object_id': str(x['_id']),
            'created_at': str(x['created_at']),
            'name_of_namespace': x['name_of_namespace']
        })
    
    return results

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
async def make_query_by_words(words_query: WordsQueryModel, response: Response):

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