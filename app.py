from analysis import Searcher
import json
from fastapi import FastAPI

svd_filename = 'searcher_svds.npz'
vocabulary_filename = 'vocabulary.gz'
web_table_filename = 'web.csv'

searcher = Searcher.load(
    svd_filename,
    vocabulary_filename,
    web_table_filename
)

del searcher.table['terms']

app = FastAPI()

@app.get("/query/{query}")
def query(query: str):

    result = searcher.query(query)
    result = result.iloc[0:100, :]

    return json.dumps(list(result.T.to_dict().values()), ensure_ascii=False)
