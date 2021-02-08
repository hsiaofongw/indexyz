from typing import List, Optional
from pydantic import BaseModel, Field

class Entry(BaseModel):
    name_of_entry: str
    terms: List[str]

class NameSpace(BaseModel):
    name_of_namespace: str
    entries: List[Entry]

class WordsQuery(BaseModel):
    words: List[str] = Field(min_items=1)
