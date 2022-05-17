from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from now.bff.app import build_app
from now.cli import cli
from now.deployment.deployment import terminate_wolf
from now.run_all_k8s import get_remote_flow_details


@pytest.fixture
def test_client():
    app = build_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup(deployment_type, dataset):
    yield
    if deployment_type == 'remote':
        if dataset == 'best-artworks':
            flow_id = get_remote_flow_details()['flow_id']
            terminate_wolf(flow_id)
    else:
        kwargs = {
            'deployment_type': deployment_type,
            'now': 'stop',
            'cluster': 'kind-jina-now',
        }
        kwargs = Namespace(**kwargs)
        cli(args=kwargs)
