import base64

from docarray import Document
from fastapi import HTTPException
from jina import Client


def process_query(text: str = None, blob: str = None) -> Document:
    if text is None and blob is None:
        raise ValueError('Please pass the query to make a search')
    try:
        if text is not None:
            query_doc = Document(text=text)
        else:
            base64_bytes = blob.encode('utf-8')
            message_bytes = base64.decodebytes(base64_bytes)
            query_doc = Document(blob=message_bytes)
    except BaseException as e:
        raise HTTPException(
            status_code=500,
            detail=f'Not a correct encoded query. Please see the error stack for more information. \n{e}',
        )
    return query_doc


def get_jina_client(host: str, port: int) -> Client:
    if 'wolf.jina.ai' in host:
        return Client(host=host)
    else:
        return Client(host=host, port=port)
