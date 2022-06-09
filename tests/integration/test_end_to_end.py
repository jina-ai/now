import json
import os
import time
from argparse import Namespace
from os.path import expanduser as user

import pytest
import requests

from now.cli import _get_kind_path, _get_kubectl_path, cli
from now.cloud_manager import create_local_cluster
from now.constants import JC_SECRET, Apps, DemoDatasets, Modalities
from now.deployment.deployment import cmd, terminate_wolf
from now.dialog import NEW_CLUSTER
from now.run_all_k8s import get_remote_flow_details


@pytest.fixture()
def cleanup(deployment_type, dataset):
    start = time.time()
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
    now = time.time() - start
    mins = int(now / 60)
    secs = int(now % 60)
    print(50 * '#')
    print(
        f'Time taken to execute `{deployment_type}` deployment with dataset `{dataset}`: {mins}m {secs}s'
    )
    print(50 * '#')


@pytest.mark.parametrize(
    'app, output_modality, dataset',
    [
        (Apps.TEXT_TO_IMAGE, Modalities.IMAGE, DemoDatasets.BIRD_SPECIES),
        (Apps.IMAGE_TO_IMAGE, Modalities.IMAGE, DemoDatasets.BEST_ARTWORKS),
        (Apps.IMAGE_TO_TEXT, Modalities.TEXT, DemoDatasets.ROCK_LYRICS),
        (Apps.TEXT_TO_TEXT, Modalities.TEXT, DemoDatasets.POP_LYRICS),
    ],
)  # art, rock-lyrics -> no finetuning, fashion -> finetuning
@pytest.mark.parametrize('quality', ['medium'])
@pytest.mark.parametrize('cluster', [NEW_CLUSTER['value']])
@pytest.mark.parametrize('deployment_type', ['local', 'remote'])
def test_backend(
    app: str,
    output_modality: str,
    dataset: str,
    quality: str,
    cluster: str,
    deployment_type: str,
    cleanup,
):
    if deployment_type == 'remote' and dataset != 'best-artworks':
        pytest.skip('Too time consuming, hence skipping!')

    os.environ['NOW_CI_RUN'] = 'True'
    kwargs = {
        'now': 'start',
        'app': app,
        'data': dataset,
        'quality': quality,
        'cluster': cluster,
        'deployment_type': deployment_type,
        'proceed': True,
    }
    # need to create local cluster and namespace to deploy playground and bff for WOLF deployment
    if deployment_type == 'remote':
        kind_path = _get_kind_path()
        create_local_cluster(kind_path, **kwargs)
        kubectl_path = _get_kubectl_path()
        cmd(f'{kubectl_path} create namespace nowapi')

    kwargs = Namespace(**kwargs)
    cli(args=kwargs)

    if dataset == DemoDatasets.BEST_ARTWORKS:
        search_text = 'impressionism'
    elif dataset == DemoDatasets.NFT_MONKEY:
        search_text = 'laser eyes'
    else:
        search_text = 'test'

    # Perform end-to-end check via bff
    request_body = {'text': search_text, 'limit': 9}
    if deployment_type == 'local':
        request_body['host'] = 'gateway'
        request_body['port'] = 8080
    elif deployment_type == 'remote':
        print(f"Getting gateway from flow_details")
        with open(user(JC_SECRET), 'r') as fp:
            flow_details = json.load(fp)
        request_body['host'] = flow_details['gateway']

    response = requests.post(
        f'http://localhost:30090/api/v1/{output_modality}/search', json=request_body
    )

    assert response.status_code == 200
    assert len(response.json()) == 9
