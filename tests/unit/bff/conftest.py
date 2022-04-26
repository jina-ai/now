from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from now.bff.app import build_app
from now.cli import cli
from now.dialog import NEW_CLUSTER


@pytest.fixture
def test_client():
    app = build_app()
    return TestClient(app)


@pytest.fixture(scope='package')
def backend_server():
    kwargs = {
        'output_modality': 'image',
        'data': 'best-artworks',
        'quality': 'medium',
        'cluster': NEW_CLUSTER,
        'cluster_new': 'local',
        'proceed': True,
    }
    kwargs = Namespace(**kwargs)
    cli(args=kwargs)
