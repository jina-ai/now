import base64
from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter

from deployment.bff.app.v1.models.image import (
    NowImageIndexRequestModel,
    NowImageResponseModel,
    NowImageSearchRequestModel,
)
from deployment.bff.app.v1.routers.helper import get_jina_client, process_query

router = APIRouter()


# Index
@router.post(
    "/index",
    summary='Add more image data to the indexer',
)
def index(data: NowImageIndexRequestModel):
    """
    Append the list of image data to the indexer. Each image data should be
    `base64` encoded using human-readable characters - `utf-8`.
    """
    index_docs = DocumentArray()
    for image, tags in zip(data.images, data.tags):
        base64_bytes = image.encode('utf-8')
        message = base64.decodebytes(base64_bytes)
        index_docs.append(Document(blob=message, tags=tags))

    get_jina_client(data.host, data.port).post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=List[NowImageResponseModel],
    summary='Search image data via image as query',
)
def search(data: NowImageSearchRequestModel):
    """
    Retrieve matching images for a given image query. Image query should be
    `base64` encoded using human-readable characters - `utf-8`.
    """
    query_doc = process_query(blob=data.image)
    docs = get_jina_client(data.host, data.port).post(
        '/search', query_doc, parameters={"limit": data.limit}
    )
    return docs[0].matches.to_dict()
