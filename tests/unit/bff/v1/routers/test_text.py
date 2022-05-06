import pytest
from grpc.aio import AioRpcError


def test_index(test_client, test_index_text):
    with pytest.raises(AioRpcError):
        test_client.post(f'/api/v1/text/index', json=test_index_text)


def test_search(test_client, test_search_text):
    with pytest.raises(AioRpcError):
        test_client.post(
            f'/api/v1/text/search',
            json=test_search_text,
        )


def test_search_text_via_no_base64_image(test_client):
    response = test_client.post(
        f'/api/v1/text/search',
        json={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_search_text_via_base64_image(test_client, test_search_image):
    with pytest.raises(AioRpcError):
        test_client.post(
            f'/api/v1/text/search',
            json=test_search_image,
        )


def test_no_query(test_client):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/text/search',
            json={},
        )


def test_both_query(test_client, test_search_both):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/text/search',
            json=test_search_both,
        )
