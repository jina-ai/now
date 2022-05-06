import pytest
from fastapi.testclient import TestClient

from now.bff.app import build_app

data_url = 'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets/jpeg/best-artworks.img10.bin'


@pytest.fixture
def test_client():
    app = build_app()
    return TestClient(app)
