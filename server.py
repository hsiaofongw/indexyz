from typing import Optional, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import datetime

class Article(BaseModel):
    article_name: str
    article_content: str

class Namespace(BaseModel):
    name_of_namespace: str
    api_key: str
    created_at: Optional[str] = None

class Auth(BaseModel):
    api_key: str

class UploadData(BaseModel):
    auth: Auth
    articles: List[Article]

app = FastAPI()

client = MongoClient('mongodb://127.0.0.1:27017/')

db = client.indexyz

start_log = {
    "message": "Server start now!",
    "date": datetime.datetime.utcnow()
}

logs = db.my_server_logs
log_id = logs.insert_one(start_log).inserted_id

print(log_id)

# hello, world
@app.get("/")
def read_root():
    return {"Hello": "World"}

# make an query
@app.get("/{namespace}/{sentence}")
def read_item(namespace: str, sentence: str = ''):
    results = []
    return {
        "namespace": namespace,
        "sentence": sentence,
        "result": results
    }

# create an namespace
@app.post('/', status_code=201)
def create_namespace(namespace: Namespace):

    namespace = namespace.copy()

    namespace.created_at = datetime.datetime.utcnow()

    namespaces = db.namespaces

    d = namespace.dict()
    inserted_id = namespaces.insert_one(d).inserted_id

    return {
        "ok": True, 
        "message": "namespace created.", 
        "name_namespace": namespace.name_of_namespace, 
        "api_key": namespace.api_key,
        "created_at": namespace.created_at,
        "inserted_id": str(inserted_id)
    }

@app.post('/{namespace}')
def append_entry_at_given_namespace(namespace: str, upload_data: UploadData):
    # api_key = upload_data.auth.api_key
    # print('api_key: ' + api_key)

    print(upload_data.dict())