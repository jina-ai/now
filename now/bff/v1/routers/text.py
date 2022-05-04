from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter
from jina import Client

from now.bff.v1.models.text import NowTextResponseModel
from now.bff.v1.routers.helper import process_query

router = APIRouter()


# Index
@router.post(
    "/index",
    summary='Add more data to the indexer',
)
def index(data: List[str], host: str = 'localhost', port: int = 31080):
    """
    Append the list of texts to the indexer.
    """
    index_docs = DocumentArray()
    for text in data:
        index_docs.append(Document(text=text))

    c = Client(host=host, port=port)
    c.post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=List[NowTextResponseModel],
    summary='Search text data via text or image as query',
)
def search(
    text: str = None,
    image: str = None,
    host: str = 'localhost',
    port: int = 31080,
    limit: int = 10,
):
    """
    Retrieve matching texts for a given text as query. Query should be `base64` encoded
    using human-readable characters - `utf-8`.
    """
    query_doc = process_query(text, image)
    c = Client(host=host, port=port)
    docs = c.post('/search', query_doc, parameters={"limit": limit})
    return docs[0].matches.to_dict()
