from analysis import Searcher
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

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

@app.get("/")
def query():

    query = "数据挖掘 机器学习"
    result = searcher.query(query)
    result = result[0:9]

    del result['cosine']

    return jsonable_encoder(result)
