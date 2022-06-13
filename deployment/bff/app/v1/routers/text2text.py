from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter

from deployment.bff.app.v1.models.text import (
    NowTextIndexRequestModel,
    NowTextResponseModel,
    NowTextSearchRequestModel,
)
from deployment.bff.app.v1.routers.helper import get_jina_client, process_query

router = APIRouter()


# Index
@router.post(
    "/index",
    summary='Add more text data to the indexer',
)
def index(data: NowTextIndexRequestModel):
    """
    Append the list of text data to the indexer.
    """
    index_docs = DocumentArray()
    for text, tags in zip(data.texts, data.tags):
        index_docs.append(Document(text=text, tags=tags))
    get_jina_client(data.host, data.port).post('/index', index_docs)


# Search
@router.post(
    "/search",
    response_model=List[NowTextResponseModel],
    summary='Search text data via text as query',
)
def search(data: NowTextSearchRequestModel):
    """
    Retrieve matching text for a given text as query.
    """
    query_doc = process_query(text=data.text)
    docs = get_jina_client(data.host, data.port).post(
        '/search', query_doc, parameters={"limit": data.limit}
    )
    return docs[0].matches.to_dict()
