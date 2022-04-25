from fastapi.testclient import TestClient

from now.bff.app import app

client = TestClient(app)


def test_read_app():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'Hello': 'World!'}
