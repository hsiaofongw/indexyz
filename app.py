from fastapi import FastAPI, status, Response

from analysis import Searcher

app = FastAPI()

svd_filename = 'searcher_svds.npz'
vocabulary_filename = 'vocabulary.gz'
web_table_filename = 'web.csv'

searcher = Searcher.load(
    svd_filename,
    vocabulary_filename,
    web_table_filename
)

del searcher.table['terms']

@app.get("/query/{query}")
async def query(query: str, response: Response):

    return searcher.query(query).T.to_dict().values()
