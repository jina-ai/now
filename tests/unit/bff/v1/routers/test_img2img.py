import pytest


def test_index(test_client, test_index_image):
    with pytest.raises(ConnectionError):
        test_client.post(
            f'/api/v1/image-to-image/index',
            json=test_index_image,
        )


def test_search_img_via_no_base64_image(test_client):
    response = test_client.post(
        f'/api/v1/image-to-image/search',
        json={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_search_img_via_base64_image(test_client, test_search_image):
    with pytest.raises(ConnectionError):
        test_client.post(
            f'/api/v1/image-to-image/search',
            json=test_search_image,
        )


def test_search_no_query(test_client):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/image-to-image/search',
            json={},
        )
