import base64
from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter
from jina import Client

from now.bff.v1.models.image import NowImageResponseModel
from now.bff.v1.routers.helper import process_query

router = APIRouter()


# Index
@router.post(
    "/index",
    summary='Add more data to the indexer',
)
def index(data: List[str], host: str = 'localhost', port: int = 31080):
    """
    Append the list of image data to the indexer. Each image data should be
    `base64` encoded using human-readable characters - `utf-8`.
    """
    index_docs = DocumentArray()
    for image in data:
        base64_bytes = image.encode('utf-8')
        message = base64.decodebytes(base64_bytes)
        index_docs.append(Document(blob=message))

    c = Client(host=host, port=port)
    c.post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=List[NowImageResponseModel],
    summary='Search image data via text or image as query',
)
def search(
    text: str,
    image: str,
    host: str = 'localhost',
    port: int = 31080,
    limit: int = 10,
):
    """
    Retrieve matching images for a given query. Image query should be `base64` encoded
    using human-readable characters - `utf-8`.
    """
    query_doc = process_query(text, image)
    c = Client(host=host, port=port)
    docs = c.post('/search', query_doc, parameters={"limit": limit})
    return docs[0].matches.to_dict()
