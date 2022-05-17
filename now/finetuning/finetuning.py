import math
import os
import random
from copy import deepcopy

import finetuner
from docarray import DocumentArray
from docarray.math.evaluation import ndcg_at_k
from finetuner.tuner.callback import (
    BestModelCheckpoint,
    EarlyStopping,
    EvaluationCallback,
)
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
from jina import Client, Flow
from tqdm import tqdm

from now.deployment.deployment import cmd
from now.deployment.flow import batch, deploy_k8s
from now.hub.head_encoder.head_encoder import LinearHead
from now.log.log import yaspin_extended
from now.utils import get_device, sigmap

epochs = 50  # use early stopping


def finetune_layer(ds, batch_size, final_layer_output_dim, embedding_size, tmpdir):
    embedding_datasets = []
    for da in [ds['train'], ds['val'], ds['val_query'], ds['val_index']]:
        embedding_ds = DocumentArray()
        embedding_datasets.append(embedding_ds)
        for d in da:
            d = deepcopy(d)
            # TODO 3.0
            # d.tensor = d.embedding
            d.tensor = d.embedding
            # d.embedding = None
            embedding_ds.append(d)
    (
        train_embedding,
        validation_embedding,
        query_embedding,
        index_embedding,
    ) = embedding_datasets
    # print(
    #     f'data size: (train, {len(train_embedding)}), (val, {len(validation_embedding)}), (query, {len(query_embedding)}), (index, {len(index_embedding)})')
    save_dir = os.path.join(tmpdir, 'now/hub/head_encoder')
    callbacks = [
        EvaluationCallback(
            # TODO parameterize limit based on dataset
            query_embedding,
            index_embedding,
            limit=20,
            num_workers=8,
            metrics={'ndcg': (ndcg_at_k, {})},
        ),
        BestModelCheckpoint(monitor='ndcg', save_dir=save_dir),
        EarlyStopping(monitor='ndcg', verbose=False, patience=5),
    ]

    # def configure_optimizer(model: torch.nn.Module):
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=int(len(ds) / batch_size / 2),
    #         num_training_steps=epochs * math.ceil(len(train_embedding) / batch_size)
    #     )
    #     return optimizer, scheduler

    print('💪 fine-tuning:')
    mean_path = os.path.join(tmpdir, 'now/hub/head_encoder')
    head = LinearHead(final_layer_output_dim, embedding_size, mean_path=mean_path)
    # noinspection PyTypeChecker
    finetuner.fit(
        head,
        train_data=train_embedding,
        eval_data=validation_embedding,
        epochs=epochs,
        learning_rate=5e-4,
        batch_size=batch_size,
        loss=TripletLoss(
            # Todo maybe cosine worked as good as euclidean
            # margin seems to have no effect on the art dataset
            margin=0,
            miner=TripletEasyHardMiner(pos_strategy='hard', neg_strategy='hard'),
        ),
        device=get_device(),
        # configure_optimizer=configure_optimizer,
        num_items_per_class=4,
        callbacks=callbacks,
        tag_key=finetuner.__default_tag_key__,
    )
    print('  🧠 Perfect! Early stopping triggered since accuracy is great already')
    return f'{save_dir}/best_model_ndcg'


def add_clip_embeddings(dataset, vision_model, tmpdir, kubectl_path):
    need_to_add_embeddings = False
    with yaspin_extended(
        sigmap=sigmap, text="Check if embeddings already exist", color="green"
    ) as spinner:
        for k, da in dataset.items():
            if da is None:
                continue
            for d in da:
                if d.embedding is None:
                    need_to_add_embeddings = True
                    break
        if not need_to_add_embeddings:
            spinner.ok('👍')
            return
        spinner.fail('👎')

    ns = 'nowtmp'
    f = Flow(name=ns, port_expose=8080, cors=True,).add(
        name='clip',
        uses='jinahub+docker://CLIPEncoder/v0.2.1',
        uses_with={'pretrained_model_name_or_path': vision_model},
    )
    (
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = deploy_k8s(f, ns, 3, tmpdir, kubectl_path=kubectl_path)
    for k, da in dataset.items():
        if da is not None:
            # this is just to save computation in case we have the embeddings already
            results = DocumentArray()
            embedding_dataset = DocumentArray()
            no_embedding_dataset = DocumentArray()
            for d in da:
                if d.embedding is None:
                    no_embedding_dataset.append(d)
                else:
                    embedding_dataset.append(d)
            client = Client(host=gateway_host, port=gateway_port)
            print(f'▶ create embeddings for {len(no_embedding_dataset)} documents')
            for x in tqdm(
                batch(no_embedding_dataset, 16),
                total=math.ceil(len(no_embedding_dataset) / 16),
            ):
                response = client.post('/index', request_size=16, inputs=x)
                results.extend(response)

            dataset[k] = (embedding_dataset + results).shuffle(42)
    cmd(f'{kubectl_path} delete ns {ns}')


def fill_missing(ds, train_val_split_ratio, num_default_val_queries, is_debug):
    if ds['train'] is None:
        ds['train'] = ds['index']
    if ds['val'] is None:
        # TODO makes split based on classes rather than instances
        split_index = max(
            int(len(ds['train']) * train_val_split_ratio),
            len(ds['train']) - 5000,
        )
        train = ds['train']
        ds['train'], ds['val'] = train[:split_index], train[split_index:]

    if ds['val_index'] is None:
        ds['val_index'] = deepcopy(ds['val'])
    if ds['val_query'] is None:
        if is_debug:
            num_queries = 10
        else:
            num_queries = 100

        ds['val_query'] = DocumentArray(
            [deepcopy(doc) for doc in random.sample(ds['val_index'], num_queries)]
        )
        # for d in ds['val_query']:
        #     ds['val_index'].remove(d)

    if ds['val_index_image'] is None:
        ds['val_index_image'] = deepcopy(
            # DocumentArray(d for d in ds['val'] if d.blob is not None)
            DocumentArray(d for d in ds['val'] if d.blob != b'')
        )
    if ds['val_query_image'] is None:
        ds['val_query_image'] = DocumentArray(
            [
                deepcopy(doc)
                for doc in random.sample(ds['val_index_image'], num_default_val_queries)
            ]
        )
