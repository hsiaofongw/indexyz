from typing import Optional, List

from fastapi import FastAPI, Header, status
from pydantic import BaseModel

from datetime import datetime

class Entry(BaseModel):
    name_of_entry: str
    terms: List[str]

class NameSpace(BaseModel):
    name_of_namespace: str
    entries: List[Entry]

class Query(BaseModel):
    words: Optional[List[str]] = [""]
    sentence: Optional[str] = ""

app = FastAPI()

@app.get("/status")
async def get_status():

    return {
        "number_of_namespaces": 14,
        "namespaces": ['ns1', 'ns2'],
        "server_started_at": "when"
    }

@app.post("/namespaces/", status_code=status.HTTP_201_CREATED)
async def create_namespace(ns: NameSpace):

    now = datetime.utcnow()
    name = ns.name_of_namespace
    entries = ns.entries

    return {
        "message": "namespace created.",
        "name_of_namespace": name,
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