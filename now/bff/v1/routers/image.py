import base64
from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter, HTTPException
from jina import Client
from jina.serve.runtimes.gateway.http.models import JinaResponseModel

router = APIRouter()


def from_base64(data: str):
    return base64.b64decode(data)


# Index
@router.post(
    "/index",
    response_model=JinaResponseModel,
    summary='Add more data to the indexer',
)
def index(data: List[str], host: str = 'localhost', port: int = 31080):
    """
    Append the image data to the indexer
    """
    index_docs = DocumentArray()
    for text in data:
        index_docs.append(Document(text=text))

    c = Client(host=host, port=port)
    c.post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=JinaResponseModel,
    summary='Search image data via text as query',
)
def search(
    query: str,
    host: str = 'localhost',
    port: int = 31080,
    modality: str = 'text',
    limit: int = 10,
):
    """
    Retrieve matching images for a given query. Query should be base64 encoded
    using human-readable characters - `utf-8`.
    """
    try:
        base64_bytes = query.encode(
            'utf-8'
        )  # This might not be necessary as `utf-8` is set by default
        if modality == 'text':
            message_bytes = base64.b64decode(base64_bytes).decode('utf-8')
            query_doc = Document(text=message_bytes)
        elif modality == 'image':
            message_bytes = base64.decodebytes(base64_bytes)
            query_doc = Document(blob=message_bytes)
        else:
            raise HTTPException(status_code=404, detail=f'Wrong modality selected.')
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f'Not a correct encoded query. Please make sure it is base64 encoded. \n{e}',
        )
    c = Client(host=host, port=port)
    docs = c.post('/search', query_doc, parameters={"limit": limit})
    del docs[...][:, ('embedding', 'tensor')]
    return {"data": docs.to_dict()}
