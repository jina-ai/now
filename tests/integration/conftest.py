import pytest
from fastapi.testclient import TestClient

from now.bff.app import build_app


@pytest.fixture
def test_client():
    app = build_app()
    return TestClient(app)
