from typing import Callable, Dict, Generator, Optional, Union

import pytest
import requests
from docarray import Document, DocumentArray
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from deployment.bff.app.app import build_app

data_url = 'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets/jpeg/best-artworks.img10.bin'


class MockedJinaClient:
    """
    This class is used to override the JinaClient dependency in the bff.
    On each call, it returns a `DocumentArray` with the call args in the `Document` tags.
    """

    def __init__(self, response: DocumentArray):
        self.response = response

    def post(
        self,
        url: str,
        query_docs: Union[Document, DocumentArray],
        parameters: Optional[Dict] = None,
    ) -> DocumentArray:
        for doc in self.response.flatten():
            doc.tags = {'url': url, 'parameters': parameters}
        return self.response


@pytest.fixture
def client() -> Generator[requests.Session, None, None]:
    with TestClient(build_app()) as client:
        yield client


@pytest.fixture(scope='function')
def client_with_mocked_jina_client(
    mocker: MockerFixture,
) -> Callable[[DocumentArray], requests.Session]:
    def _fixture(response: DocumentArray) -> requests.Session:
        def _get_jina_client(host, port):
            return MockedJinaClient(response)

        mocker.patch(
            'deployment.bff.app.v1.routers.img2img.get_jina_client', _get_jina_client
        )
        mocker.patch(
            'deployment.bff.app.v1.routers.img2txt.get_jina_client', _get_jina_client
        )
        mocker.patch(
            'deployment.bff.app.v1.routers.music2music.get_jina_client',
            _get_jina_client,
        )
        mocker.patch(
            'deployment.bff.app.v1.routers.txt2img.get_jina_client', _get_jina_client
        )
        return TestClient(build_app())

    return _fixture
