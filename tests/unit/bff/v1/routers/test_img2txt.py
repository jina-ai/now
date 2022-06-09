import base64
import os
from typing import Callable

import pytest
import requests
from docarray import Document, DocumentArray
from starlette import status


@pytest.fixture
def base64_image_string(resources_folder_path: str) -> str:
    with open(
        os.path.join(resources_folder_path, 'image', '5109112832.jpg'), 'rb'
    ) as f:
        binary = f.read()
        img_string = base64.b64encode(binary).decode('utf-8')
    return img_string


@pytest.fixture
def sample_search_response_image() -> DocumentArray:
    result = DocumentArray([Document()])
    matches = DocumentArray([Document(uri='match')])
    result[0].matches = matches
    return result


def test_image_index_fails_with_no_flow_running(
    client: requests.Session, base64_image_string: str
):
    with pytest.raises(ConnectionError):
        client.post(
            '/api/v1/image-to-text/index',
            json={'images': [base64_image_string]},
        )


def test_image_search_fails_with_no_flow_running(
    client: requests.Session, base64_image_string: str
):
    with pytest.raises(ConnectionError):
        client.post(
            f'/api/v1/image-to-text/search',
            json={'image': base64_image_string},
        )


def test_image_search_fails_with_incorrect_query(client: requests.Session):
    response = client.post(
        f'/api/v1/image-to-text/search',
        json={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_image_search_fails_with_empty_query(client: requests.Session):
    with pytest.raises(ValueError):
        client.post(
            f'/api/v1/image-to-text/search',
            json={},
        )


def test_image_index(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_image_string: str,
):
    response = client_with_mocked_jina_client(DocumentArray()).post(
        '/api/v1/image-to-text/index', json={'images': [base64_image_string]}
    )
    assert response.status_code == status.HTTP_200_OK


def test_image_search_calls_flow(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_image_string: str,
    sample_search_response_image: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_image).post(
        '/api/v1/image-to-text/search', json={'image': base64_image_string}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    # the mock writes the call args into the response tags
    assert results[0].tags['url'] == '/search'
    assert results[0].tags['parameters']['limit'] == 10


def test_image_search_parse_response(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_image_string: str,
    sample_search_response_image: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_image).post(
        '/api/v1/image-to-text/search', json={'image': base64_image_string}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    assert len(results) == len(sample_search_response_image[0].matches)
    assert results[0].uri == sample_search_response_image[0].matches[0].uri
