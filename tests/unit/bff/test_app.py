import pytest

from now.constants import Apps


@pytest.mark.parametrize(
    'app', [Apps.TEXT_TO_IMAGE, Apps.IMAGE_TO_TEXT, Apps.IMAGE_TO_IMAGE]
)
class TestParametrized:
    def test_check_liveness(self, test_client, app):
        response = test_client.get(f'/api/v1/{app}/ping')
        assert response.status_code == 200
        assert response.json() == 'pong!'

    def test_read_root(self, test_client, app):
        response = test_client.get(f'/api/v1/{app}')
        assert response.status_code == 200

    def test_get_docs(self, test_client, app):
        response = test_client.get(f'/api/v1/{app}/docs')
        assert response.status_code == 200

    def test_get_redoc(self, test_client, app):
        response = test_client.get(f'/api/v1/{app}/redoc')
        assert response.status_code == 200
