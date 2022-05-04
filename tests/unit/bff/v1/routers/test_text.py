import base64

import pytest
from grpc.aio import AioRpcError


def test_search(test_client):
    with pytest.raises(AioRpcError):
        test_client.post(
            f'/api/v1/text/search',
            params={'text': 'Hello'},
        )


def test_search_text_via_no_base64_image(test_client):
    response = test_client.post(
        f'/api/v1/text/search',
        params={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_search_text_via_base64_image(test_client):
    with open('./tests/image-data/kids2.jpg', 'rb') as f:
        binary = f.read()
        query = base64.b64encode(binary).decode('utf-8')
    with pytest.raises(AioRpcError):
        response = test_client.post(
            f'/api/v1/text/search',
            params={'image': query},
        )


def test_no_query(test_client):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/text/search',
        )


def test_both_query(test_client):
    with open('./tests/image-data/kids2.jpg', 'rb') as f:
        binary = f.read()
        img = base64.b64encode(binary).decode('utf-8')
    with pytest.raises(ValueError):
        test_client.post(f'/api/v1/text/search', params={'text': 'Hello', 'image': img})
