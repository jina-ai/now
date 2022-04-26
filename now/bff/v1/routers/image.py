from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter, File, UploadFile
from jina import Client
from jina.serve.runtimes.gateway.http.models import JinaResponseModel

router = APIRouter()


# Index
@router.post(
    "/index",
    response_model=JinaResponseModel,
    summary='Add more data to the indexer',
)
def index(host: str, data: List[str]):
    """
    Append the image data to the indexer
    """
    index_docs = DocumentArray()
    for text in data:
        index_docs.append(Document(text=text))

    c = Client(host=host, port=31080)
    c.post('/index', index_docs)


# Search
@router.post(
    "/search/{query}",
    response_model=JinaResponseModel,
    summary='Search image data via text as query',
)
def search(query: str, host: str = 'localhost', limit: int = 10):
    """
    Retrieve matching images for a given text as query
    """
    query_doc = Document(text=query)
    c = Client(host=host, port=31080)
    docs = c.post('/search', query_doc, parameters={"limit": limit})
    del docs[...][:, ('embedding', 'tensor')]
    return {"data": docs.to_dict()}


@router.post(
    "/search",
    response_model=JinaResponseModel,
    summary='Search image data via image as query',
)
def search(
    host: str = 'localhost',
    image_file: UploadFile = File(...),
    limit: int = 10,
):
    """
    Retrieve matching images for a given image uri as query
    """
    # TODO: Uri or FileUploader?
    contents = image_file.file.read()
    query_doc = Document(blob=contents)
    query_doc.convert_blob_to_image_tensor(224, 224)
    c = Client(host=host, port=31080)
    docs = c.post('/search', query_doc, parameters={"limit": limit})
    del docs[...][:, ('embedding', 'tensor')]
    return {"data": docs.to_dict()}
