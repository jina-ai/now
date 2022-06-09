import base64
import os
from typing import Callable

import pytest
import requests
from docarray import Document, DocumentArray
from starlette import status


@pytest.fixture
def base64_audio_string(resources_folder_path: str) -> str:
    with open(
        os.path.join(
            resources_folder_path,
            'music',
            '0ae22dba39adebd474025d6f97059d5e425e2cf2.mp3',
        ),
        'rb',
    ) as f:
        binary = f.read()
        audio_string = base64.b64encode(binary).decode('utf-8')
    return audio_string


@pytest.fixture
def sample_search_response_music() -> DocumentArray:
    result = DocumentArray([Document()])
    result[0].uri = 'match'
    return result


def test_music_index_fails_with_no_flow_running(
    client: requests.Session, base64_audio_string: str
):
    with pytest.raises(ConnectionError):
        client.post(
            f'/api/v1/music-to-music/index', json={'songs': [base64_audio_string]}
        )


def test_music_search_fails_with_no_flow_running(
    client: requests.Session, base64_audio_string: str
):
    with pytest.raises(ConnectionError):
        client.post(
            f'/api/v1/music-to-music/search',
            json={'song': base64_audio_string},
        )


def test_music_search_fails_with_incorrect_query(
    client: requests.Session, base64_audio_string: str
):
    response = client.post(
        f'/api/v1/music-to-music/search',
        json={'song': 'hh'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_music_search_fails_with_emtpy_query(client: requests.Session):
    with pytest.raises(ValueError):
        client.post(
            f'/api/v1/music-to-music/search',
            json={},
        )


def test_music_index(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_audio_string: str,
):
    response = client_with_mocked_jina_client(DocumentArray()).post(
        '/api/v1/music-to-music/index', json={'songs': [base64_audio_string]}
    )
    assert response.status_code == status.HTTP_200_OK


def test_music_search_calls_flow(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_audio_string: str,
    sample_search_response_music: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_music).post(
        '/api/v1/music-to-music/search', json={'song': base64_audio_string}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    # the mock writes the call args into the response tags
    assert results[0].tags['url'] == '/search'
    assert results[0].tags['parameters']['limit'] == 10


def test_music_search_parse_response(
    client_with_mocked_jina_client: Callable[[DocumentArray], requests.Session],
    base64_audio_string: str,
    sample_search_response_music: DocumentArray,
):
    response = client_with_mocked_jina_client(sample_search_response_music).post(
        '/api/v1/music-to-music/search', json={'song': base64_audio_string}
    )

    assert response.status_code == status.HTTP_200_OK
    results = DocumentArray.from_json(response.content)
    assert len(results) == len(sample_search_response_music)
    assert results[0].uri == sample_search_response_music[0].uri
