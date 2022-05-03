from argparse import Namespace

import pytest
from docarray import Document
from jina import Client

from now.cli import cli
from now.dialog import NEW_CLUSTER
from now.log import log


@pytest.mark.parametrize(
    'output_modality, dataset',
    [('image', 'best-artworks'), ('image', 'deepfashion'), ('text', 'rock-lyrics')],
)  # art, rock-lyrics -> no finetuning, fashion -> finetuning
@pytest.mark.parametrize('quality', ['medium'])
@pytest.mark.parametrize('cluster', [NEW_CLUSTER['value']])
@pytest.mark.parametrize('new_cluster_type', ['local'])
def test_backend(
    output_modality: str,
    dataset: str,
    quality: str,
    cluster: str,
    new_cluster_type: str,
):
    log.TEST = True
    sandbox = dataset == 'best-artworks'
    kwargs = {
        'output_modality': output_modality,
        'data': dataset,
        'quality': quality,
        'sandbox': sandbox,
        'cluster': cluster,
        'new_cluster_type': new_cluster_type,
        'proceed': True,
    }
    kwargs = Namespace(**kwargs)
    cli(args=kwargs)

    if dataset == 'best-artworks':
        search_text = 'impressionism'
    elif dataset == 'nft-monkey':
        search_text = 'laser eyes'
    else:
        search_text = 'test'

    client = Client(
        host='localhost',
        protocol="grpc",
        port=31080,  # 30080 for frontend, 31080 for backend
    )
    response = client.search(
        Document(text=search_text),
        parameters={"limit": 9, "filter": {}},
    )

    assert response[0].matches
