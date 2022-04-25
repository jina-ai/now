import pytest
from api.app import build_app
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(test_session):
    app = build_app()
    return TestClient(app)
