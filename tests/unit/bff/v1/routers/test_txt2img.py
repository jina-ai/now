import pytest


def test_index(test_client, test_index_image):
    with pytest.raises(ConnectionError):
        test_client.post(f'/api/v1/text-to-image/index', json=test_index_image)


def test_search(test_client, test_search_text):
    with pytest.raises(ConnectionError):
        test_client.post(
            f'/api/v1/text-to-image/search',
            json=test_search_text,
        )


def test_no_query(test_client):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/text-to-image/search',
            json={},
        )
