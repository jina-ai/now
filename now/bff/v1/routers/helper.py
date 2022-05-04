import base64

from docarray import Document
from fastapi import HTTPException


def process_query(text: str, image: str):
    if text is None and image is None:
        raise ValueError('Please set one of the value - `image` or `text`')
    if text is not None and image is not None:
        raise ValueError('Please set either image or text not both!')
    try:
        if text is not None:
            query_doc = Document(text=text)
        else:
            base64_bytes = image.encode('utf-8')
            message_bytes = base64.decodebytes(base64_bytes)
            query_doc = Document(blob=message_bytes)
    except BaseException as e:
        raise HTTPException(
            status_code=500,
            detail=f'Not a correct encoded query. Please see the error stack for more information. \n{e}',
        )
    return query_doc
