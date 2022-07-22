""" This module implements functionality to fine tune on the music dataset """
import math
from typing import Dict

from docarray import DocumentArray
from tqdm import tqdm

from now.deployment.flow import batch, deploy_flow
from now.log import time_profiler

_KS_NAMESPACE = 'embed-now'


@time_profiler
def embed_now(
    deployment_type: str,
    flow_yaml: str,
    env_dict: Dict,
    dataset: DocumentArray,
    kubectl_path: str,
):
    documents_without_embedding = DocumentArray(
        list(filter(lambda d: d.embedding is None, dataset))
    )

    result = DocumentArray()
    client, _, _, _, _, = deploy_flow(
        deployment_type=deployment_type,
        flow_yaml=flow_yaml,
        ns=_KS_NAMESPACE,
        env_dict=env_dict,
        kubectl_path=kubectl_path,
    )
    print(f'▶ create embeddings for {len(documents_without_embedding)} documents')
    for x in tqdm(
        batch(documents_without_embedding, 16),
        total=math.ceil(len(documents_without_embedding) / 16),
    ):
        response = client.post('/index', request_size=16, inputs=x)
        result.extend(response)

    for doc in result:
        dataset[doc.id].embedding = doc.embedding
