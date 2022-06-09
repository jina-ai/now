from typing import Callable

import pytest
import requests
from docarray import DocumentArray
from starlette import status


def test_image_index_fails_with_no_flow_running(
    client: requests.Session, base64_image_string: str
):
    with pytest.raises(ConnectionError):
        client.post(
            '/api/v1/text-to-image/index',
            json={'images': [base64_image_string]},
        )


def test_image_search_fails_with_no_flow_running(client: requests.Session):
    with pytest.raises(ConnectionError):
        client.post(
            f'/api/v1/text-to-image/search',
            json={'text': 'Hello'},
        )


def test_image_search_fails_with_empty_query(client: requests.Session):
    with pytest.raises(ValueError):
        client.post(
            f'/api/v1/text-to-image/search',
            json={},
        )


def test_image_index(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_image_string: str,
):
    response = client_with_mocked_jina_client(DocumentArray()).post(
        '/api/v1/text-to-image/index', json={'images': [base64_image_string]}
    )
    assert response.status_code == status.HTTP_200_OK


def test_image_search_calls_flow(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    sample_search_response_image: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_image).post(
        '/api/v1/text-to-image/search', json={'text': 'Hello'}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    # the mock writes the call args into the response tags
    assert results[0].tags['url'] == '/search'
    assert results[0].tags['parameters']['limit'] == 10


def test_image_search_parse_response(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    sample_search_response_image: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_image).post(
        '/api/v1/text-to-image/search', json={'text': 'Hello'}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    assert len(results) == len(sample_search_response_image[0].matches)
    assert results[0].uri == sample_search_response_image[0].matches[0].uri
