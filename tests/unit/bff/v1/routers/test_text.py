from typing import Callable

import pytest
import requests
from docarray import Document, DocumentArray
from starlette import status


@pytest.fixture
def sample_search_response_text() -> DocumentArray:
    result = DocumentArray([Document()])
    matches = DocumentArray([Document(text='match')])
    result[0].matches = matches
    return result


def test_text_index_fails_with_no_flow_running(client: requests.Session):
    with pytest.raises(ConnectionError):
        client.post(f'/api/v1/text/index', json={'texts': ['Hello']})


def test_text_search_fails_with_no_flow_running(client: requests.Session):
    with pytest.raises(ConnectionError):
        client.post(
            f'/api/v1/text/search',
            json={'text': 'Hello'},
        )


def test_text_search_fails_with_incorrect_query(client):
    response = client.post(
        f'/api/v1/text/search',
        json={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_text_search_fails_with_emtpy_query(client: requests.Session):
    with pytest.raises(ValueError):
        client.post(
            f'/api/v1/text/search',
            json={},
        )


def test_text_index(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
):
    response = client_with_mocked_jina_client(DocumentArray()).post(
        '/api/v1/text/index', json={'texts': ['Hello'], 'tags': [{'tag': val}]}
    )
    assert response.status_code == status.HTTP_200_OK


def test_text_search_calls_flow(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    sample_search_response_text: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_text).post(
        '/api/v1/text/search', json={'text': 'Hello'}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    # the mock writes the call args into the response tags
    assert results[0].tags['url'] == '/search'
    assert results[0].tags['parameters']['limit'] == 10


def test_text_search_parse_response(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    sample_search_response_text: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_text).post(
        '/api/v1/text/search', json={'text': 'Hello'}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    assert len(results) == len(sample_search_response_text[0].matches)
    assert results[0].text == sample_search_response_text[0].matches[0].text
