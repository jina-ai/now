import base64
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


@pytest.fixture
def test_search_image(resources_folder_path: str):
    with open(
        os.path.join(resources_folder_path, 'image', '5109112832.jpg'), 'rb'
    ) as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return img_query


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
            'delete-cluster': True,
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
    'app, input_modality, output_modality, dataset',
    [
        (
            Apps.TEXT_TO_IMAGE,
            Modalities.TEXT,
            Modalities.IMAGE,
            DemoDatasets.BIRD_SPECIES,
        ),
        (
            Apps.IMAGE_TO_IMAGE,
            Modalities.IMAGE,
            Modalities.IMAGE,
            DemoDatasets.BEST_ARTWORKS,
        ),
        (
            Apps.IMAGE_TO_TEXT,
            Modalities.IMAGE,
            Modalities.TEXT,
            DemoDatasets.ROCK_LYRICS,
        ),
        (
            Apps.TEXT_TO_TEXT,
            Modalities.TEXT,
            Modalities.TEXT,
            DemoDatasets.POP_LYRICS,
        ),
    ],
)  # art, rock-lyrics -> no finetuning, fashion -> finetuning
@pytest.mark.parametrize('quality', ['medium'])
@pytest.mark.parametrize('cluster', [NEW_CLUSTER['value']])
@pytest.mark.parametrize('deployment_type', ['local', 'remote'])
def test_backend(
    app: str,
    dataset: str,
    quality: str,
    cluster: str,
    deployment_type: str,
    test_search_image,
    cleanup,
    input_modality,
    output_modality,
):
    if deployment_type == 'remote' and dataset != 'best-artworks':
        pytest.skip('Too time consuming, hence skipping!')

    os.environ['NOW_CI_RUN'] = 'True'
    os.environ['JCLOUD_LOGLEVEL'] = 'DEBUG'
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
    response = cli(args=kwargs)

    assert (
        response['bff']
        == f'http://localhost:30090/api/v1/{app.replace("_", "-")}/redoc'
    )
    assert response['playground'].startswith('http://localhost:30080/?')
    assert response['input_modality'] == input_modality
    assert response['output_modality'] == output_modality
    if deployment_type == 'local':
        assert response['host'] == 'gateway.nowapi.svc.cluster.local'
    else:
        assert response['host'].startswith('grpcs://')
        assert response['host'].endswith('.wolf.jina.ai')
    assert response['port'] == 8080 or response['port'] is None

    if dataset == DemoDatasets.BEST_ARTWORKS:
        search_text = 'impressionism'
    elif dataset == DemoDatasets.NFT_MONKEY:
        search_text = 'laser eyes'
    else:
        search_text = 'test'

    # Perform end-to-end check via bff
    if app in [Apps.IMAGE_TO_IMAGE, Apps.IMAGE_TO_TEXT]:
        request_body = {'image': test_search_image, 'limit': 9}
    elif app in [Apps.TEXT_TO_IMAGE, Apps.TEXT_TO_TEXT]:
        request_body = {'text': search_text, 'limit': 9}
    else:  # Add different request body if app changes
        request_body = {}

    if deployment_type == 'local':
        request_body['host'] = 'gateway'
        request_body['port'] = 8080
    elif deployment_type == 'remote':
        print(f"Getting gateway from flow_details")
        with open(user(JC_SECRET), 'r') as fp:
            flow_details = json.load(fp)
        request_body['host'] = flow_details['gateway']

    response = requests.post(
        f'http://localhost:30090/api/v1/{input_modality}-to-{output_modality}/search',
        json=request_body,
    )

    assert response.status_code == 200
    assert len(response.json()) == 9
