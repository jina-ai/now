import pytest


def test_search_image(test_client):
    query = 'hello'
    with pytest.raises(BaseException):
        test_client.post(
            f'/api/v1/image/search/{query}',
            params={},
        )
