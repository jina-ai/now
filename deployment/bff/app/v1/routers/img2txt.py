from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter
from jina import Client

from deployment.bff.app.v1.models.image import NowImageSearchRequestModel
from deployment.bff.app.v1.models.text import (
    NowTextIndexRequestModel,
    NowTextResponseModel,
)
from deployment.bff.app.v1.routers.helper import process_query

router = APIRouter()


# Index
@router.post(
    "/index",
    summary='Add more text data to the indexer',
)
def index(data: NowTextIndexRequestModel):
    """
    Append the list of text to the indexer.
    """
    index_docs = DocumentArray()
    for text in data.texts:
        index_docs.append(Document(text=text))

    if 'wolf.jina.ai' in data.host:
        c = Client(host=data.host)
    else:
        c = Client(host=data.host, port=data.port)
    c.post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=List[NowTextResponseModel],
    summary='Search text data via image as query',
)
def search(data: NowImageSearchRequestModel):
    """
    Retrieve matching text for a given image query. Image query should be
    `base64` encoded using human-readable characters - `utf-8`.
    """
    query_doc = process_query(image=data.image)
    if 'wolf.jina.ai' in data.host:
        c = Client(host=data.host)
    else:
        c = Client(host=data.host, port=data.port)
    docs = c.post('/search', query_doc, parameters={"limit": data.limit})
    return docs[0].matches.to_dict()
