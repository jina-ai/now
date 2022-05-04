import base64

import pytest


def test_search(test_client):
    response = test_client.post(
        f'/api/v1/image/search',
        params={'text': 'Hello'},
    )
    assert response.status_code == 200


def test_search_img_via_no_base64_image(test_client):
    response = test_client.post(
        f'/api/v1/image/search',
        params={'image': 'hello'},
    )
    assert response.status_code == 500
    assert 'Not a correct encoded query' in response.text


def test_search_img_via_base64_image(test_client):
    with open('./tests/image-data/kids2.jpg', 'rb') as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    response = test_client.post(
        f'/api/v1/image/search',
        params={'image': img_query},
    )
    assert response.status_code == 200


def test_no_query(test_client):
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/image/search',
        )


def test_both_query(test_client):
    with open('./tests/image-data/kids2.jpg', 'rb') as f:
        binary = f.read()
        img = base64.b64encode(binary).decode('utf-8')
    with pytest.raises(ValueError):
        test_client.post(
            f'/api/v1/image/search', params={'text': 'Hello', 'image': img}
        )
