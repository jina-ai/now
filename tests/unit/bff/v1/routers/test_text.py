import pytest


def test_search_text(test_client):
    query = 'hello'
    with pytest.raises(BaseException):
        test_client.post(
            f'/api/v1/text/search/{query}',
            params={},
        )
