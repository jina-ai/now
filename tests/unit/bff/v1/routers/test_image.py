import base64

import pytest


def test_search_no_base64_query(test_client):
    response = test_client.post(
        f'/api/v1/image/search',
        params={'query': 'hello'},
    )
    assert response.status_code == 404
    assert 'Not a correct encoded query' in response.text


def test_search_base64_text(test_client):
    query = base64.b64encode('Hello'.encode('utf-8')).decode('utf-8')
    with pytest.raises(BaseException):
        test_client.post(
            f'/api/v1/image/search',
            params={'query': query, 'modality': 'text'},
        )


def test_search_base64_image(test_client):
    with open('./tests/image-data/kids2.jpg', 'rb') as f:
        binary = f.read()
        query = base64.b64encode(binary).decode('utf-8')
    with pytest.raises(BaseException):
        test_client.post(
            f'/api/v1/image/search',
            params={'query': query, 'modality': 'image'},
        )
