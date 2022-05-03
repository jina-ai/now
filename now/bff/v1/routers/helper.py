import base64

from docarray import Document
from fastapi import HTTPException


def process_query(query: str, modality: str = 'text'):
    base64_bytes = query.encode('utf-8')
    if modality == 'text':
        message_bytes = base64.b64decode(base64_bytes).decode('utf-8')
        query_doc = Document(text=message_bytes)
    elif modality == 'image':
        message_bytes = base64.decodebytes(base64_bytes)
        query_doc = Document(blob=message_bytes)
    else:
        raise HTTPException(status_code=404, detail=f'Wrong modality selected.')
    return query_doc
