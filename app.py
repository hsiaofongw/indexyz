from bson.objectid import ObjectId
from fastapi.param_functions import Query
import pymongo
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
        self.searchers: List[Searcher] = []

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
        ns = db.namespaces.find_one({'_id': base_namespace_id})
        base_namespace_name = ""
        if ns:
            base_namespace_name = ns['name_of_namespace']

        results.append({
            '_id': str(searcher_id),
            'base_namespace_id': str(base_namespace_id),
            'base_namespace_name': base_namespace_name
        })
    
    return results

@app.delete("/searchers/{searcher_id}")
async def delete_searcher_by_id(searcher_id: str, status_code=status.HTTP_202_ACCEPTED):
    db = state.get_db()
    result = db.searchers.delete_one(
        { '_id': ObjectId(searcher_id) }
    )

    return {
        'deleted_count': result.deleted_count
    }

@app.post("/searchers/{searcher_id}")
async def load_searcher(
    searcher_id: str = Query(default="", min_length=24, max_length=24), 
    response: Response = Response()
):
    db = state.get_db()
    searcher_object = db.searchers.find_one({
        '_id': ObjectId(searcher_id)
    })
    if searcher_object is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            'message': 'searcher not found.'
        }
    searcher = Searcher.from_data(searcher_object['searcher_data'])
    searcher.searcher_id = searcher_id
    state.searchers.append(searcher)

    return {
        'message': 'searcher is now loaded.',
        'loaded_searcher_ids': [x.searcher_id for x in state.searchers]
    }

@app.get("/loadedsearchers")
async def list_loaded_searchers():
    results = {
        'loaded_searcher_ids': [x.searcher_id for x in state.searchers]
    }
    return results

@app.post("/unloadsearcher/{searcher_id}")
async def unload_searcher(
    searcher_id: str,
    response: Response = Response()
):
    
    n_searchers = len(state.searchers)
    for i in range(n_searchers):
        searcher = state.searchers[i]
        if searcher.searcher_id == searcher_id:
            del state.searchers[i]

            return {
                'message': 'deleted one.'
            }

    response.status_code = status.HTTP_404_NOT_FOUND
    return {
        'message': 'no such searcher.'
    }


@app.post("/")
async def make_query_by_words(words_query: WordsQueryModel, response: Response):

    # pass
    if len(state.searchers) == 0:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            'message': 'no query object available, please upload an namespace.'
        }
    
    searcher = state.searchers[-1]
    q = searcher.make_query(words_query.words)
    indexes, cosines =  searcher.sort_index_by_cosine_similarity(q)

    name_of_entries = [searcher.index_article[str(i)] for i in indexes.tolist()]
    cosines = cosines.tolist()
    cosines = [0 if isnan(x) else x for x in cosines]
    
    response.status_code = status.HTTP_200_OK

    result = {
        "words": words_query.words,
        "name_of_entries": name_of_entries,
        "cosines": cosines
    }

    return result