import base64
from typing import List

from docarray import Document, DocumentArray
from fastapi import APIRouter

from deployment.bff.app.v1.models.music import (
    NowMusicIndexRequestModel,
    NowMusicResponseModel,
    NowMusicSearchRequestModel,
)
from deployment.bff.app.v1.routers.helper import get_jina_client, process_query

router = APIRouter()


@router.post(
    "/index",
    summary='Add more data to the indexer',
)
def index(data: NowMusicIndexRequestModel):
    """
    Append the list of songs to the indexer. Each song data request should be
    `base64` encoded using human-readable characters - `utf-8`.
    """
    index_docs = DocumentArray()
    for audio, tags in zip(data.songs, data.tags):
        base64_bytes = audio.encode('utf-8')
        message = base64.decodebytes(base64_bytes)
        index_docs.append(Document(blob=message, tags=tags))

    get_jina_client(data.host, data.port).post('/index', index_docs)


@router.post(
    "/search",
    response_model=List[NowMusicResponseModel],
    summary='Search music data via text or music as query',
)
def search(data: NowMusicSearchRequestModel):
    """
    Retrieve matching songs for a given query. Song query should be `base64` encoded
    using human-readable characters - `utf-8`. In the case of music, the docs are already the matches.
    """
    query_doc = process_query(blob=data.song)
    docs = get_jina_client(data.host, data.port).post(
        '/search', query_doc, parameters={"limit": data.limit}
    )
    return docs.to_dict()
