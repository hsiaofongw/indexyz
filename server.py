from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import datetime

class Article(BaseModel):
    article_name: str
    article_content: str
    
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
@app.post('/{namespace}', status_code=201)
def create_namespace(namespace: str):
    print("create namespace: " + namespace)
    return {
        "ok": True, 
        "message": "namespace created.", 
        "namespace": namespace, 
        "key": "123456"
    }

@app.post('/{namespace}/{entry}')
def append_entry_at_given_namespace(namespace: str, entry: str, items: List[Article]):
    return items