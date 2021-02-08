from typing import List, Optional
from pydantic import BaseModel

class Entry(BaseModel):
    name_of_entry: str
    terms: List[str]

class NameSpace(BaseModel):
    name_of_namespace: str
    entries: List[Entry]

class Query(BaseModel):
    words: Optional[List[str]] = [""]
    sentence: Optional[str] = ""