import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from jina import DocumentArray, Executor, requests
from torch.nn import Linear, Module


def get_extended_embedding_if_needed(d, target_dim):
    emb = d.embedding
    if emb.shape[0] == target_dim:
        return emb

    zeros = np.zeros(emb.shape)
    if d.text:
        order = (zeros, emb)
    else:
        order = (emb, zeros)
    return np.concatenate(order)


def extend_embeddings(da, target_dim):
    for d in da:
        d.embedding = get_extended_embedding_if_needed(d, target_dim)


class LinearHead(Module):
    def __init__(self, final_layer_output_dim, embedding_size, mean_path=None):
        super(LinearHead, self).__init__()
        self.linear1 = Linear(final_layer_output_dim, embedding_size, bias=False)
        mean_path = (
            mean_path if mean_path else str(Path(__file__).parent)
        ) + '/mean.bin'
        self.mean = load_mean(mean_path)

    def forward(self, x):
        x = x.cpu()
        x -= self.mean
        x = x.float()
        x = self.linear1(x)
        normalized_embedding = F.normalize(x, p=2, dim=1)  # L2 normalize
        return normalized_embedding


def load_mean(mean_path):
    with open(mean_path, 'rb') as f:
        return pickle.load(f)


class FineTunedLinearHeadEncoder(Executor):
    def __init__(
        self,
        final_layer_output_dim,
        embedding_size,
        model_path=None,
        mean_path=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not model_path:
            model_path = Path(__file__).parent / 'best_model_ndcg'
        self.final_layer_output_dim = final_layer_output_dim
        self.model = LinearHead(final_layer_output_dim, embedding_size, mean_path)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    @requests
    def encode(self, docs: Optional[DocumentArray], **kwargs):
        blobs = []
        texts = []
        for d in docs:
            blobs.append(d.blob)
            texts.append(d.text)
            d.tensor = get_extended_embedding_if_needed(d, self.final_layer_output_dim)
        docs.embed(self.model)
        for d, blob, text in zip(docs, blobs, texts):
            if type(d.embedding) != np.ndarray:
                d.embedding = d.embedding.numpy()
            # TODO why is this working? blob can never be None
            # if blob != b'':
            if blob is not None:
                d.blob = blob
            elif text:
                d.text = text
            else:
                raise Exception('neither text nor image present')

        # for efficiency, it is in the same executor
        # for d in docs:
        #     if d.blob is not None:
        #         d.convert_image_tensor_to_uri()
        return docs
