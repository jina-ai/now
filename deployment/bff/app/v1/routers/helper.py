import base64

from docarray import Document
from fastapi import HTTPException
from jina import Client


def process_query(text: str, blob: str) -> Document:
    if text is None and blob is None:
        raise ValueError('Please set one of the value - `blob` or `text`')
    if text is not None and blob is not None:
        raise ValueError('Please set either image or text not both!')
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
