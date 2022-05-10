import numpy as np
from docarray import Document, DocumentArray
from jina import Executor, Flow, requests


class MostSimpleIndexer(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = DocumentArray()

    @requests
    def index(self, docs: DocumentArray, **kwargs):
        self.index.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters, **kwargs):
        docs.match(self.index, limit=parameters.get('limit', 20))
        return docs


if __name__ == '__main__':
    with Flow().add(uses=MostSimpleIndexer) as f:
        f.index(Document(text='abc', embedding=np.ones(5)))
        x = f.search(
            Document(text='abc', embedding=np.ones(5)),
            parameters={'limit': 9},
            return_results=True,
        )
