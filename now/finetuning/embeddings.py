""" This module implements functionality to fine tune on the music dataset """
import math
import tempfile
from typing import Dict

from docarray import DocumentArray
from jina import Client, Flow
from tqdm import tqdm

from now.deployment.flow import batch, deploy_k8s

_KS_NAMESPACE = 'embed-now'


def embed_now(
    encoder_uses: str,
    encoder_uses_with: Dict,
    dataset: DocumentArray,
    kubectl_path: str,
):
    documents_without_embedding = DocumentArray(
        list(filter(lambda d: d.embedding is None, dataset))
    )

    flow = Flow(name=_KS_NAMESPACE, port_expose=8080, prefetch=10, cors=True).add(
        uses=encoder_uses, uses_with=encoder_uses_with
    )
    result = DocumentArray()
    with tempfile.TemporaryDirectory() as tmpdir:
        gateway_host, gateway_port, _, _ = deploy_k8s(
            flow, _KS_NAMESPACE, tmpdir, kubectl_path=kubectl_path
        )
        client = Client(host=gateway_host, port=gateway_port)
        print(f'▶ create embeddings for {len(documents_without_embedding)} documents')
        for x in tqdm(
            batch(documents_without_embedding, 16),
            total=math.ceil(len(documents_without_embedding) / 16),
        ):
            response = client.post('/index', request_size=16, inputs=x)
            result.extend(response)

    for doc in result:
        dataset[doc.id].embedding = doc.embedding
