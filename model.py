from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class EntryModel(BaseModel):
    name_of_entry: str
    terms: Optional[List[str]] = Field(default=[], min_items=1)
    content: Optional[str] = Field(default="", min_length=1)

class NameSpaceModel(BaseModel):
    name_of_namespace: str
    entries: List[EntryModel]

class WordsQueryModel(BaseModel):
    words: List[str] = Field(min_items=1)

class SearcherModel(BaseModel):
    base_namespace_id: str
    created_at: Optional[datetime]
    searcher_data: Optional[str]