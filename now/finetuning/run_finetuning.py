""" This module is the entry point to the finetuning package."""
import os
import random
import string
import sys
from time import sleep
from typing import Dict, Tuple

import finetuner
import numpy as np
from docarray import DocumentArray
from finetuner.callback import EarlyStopping, EvaluationCallback

from now.finetuning.dataset import FinetuneDataset, build_finetuning_dataset
from now.finetuning.embeddings import embed_now
from now.finetuning.settings import FinetuneSettings
from now.log import yaspin_extended
from now.now_dataclasses import UserInput
from now.utils import sigmap

_BASE_SAVE_DIR = 'now/hub/head_encoder'


def finetune_now(
    user_input: UserInput,
    dataset: DocumentArray,
    finetune_settings: FinetuneSettings,
    pre_trained_head_map: Dict[str, str],
    kubectl_path: str,
    encoder_uses: str,
    encoder_uses_with: Dict,
) -> Tuple[str, str]:
    """
    Performs the finetuning procedure:
     1. If embeddings are not present -> compute them using a k8s deployed flow
     2. If bi-modal, prepare the embeddings by concatenating zeros for the opposing modality
     3. Create model and run finetuning, get path to the tuned model and return

    Note, for music we use cached models because the datasets are too large and consume too much time

    :param user_input: The configured user input object
    :param dataset: The dataset with the finetuning labels on all documents. Embeddings are optional and can
        be computed on the fly
    :param finetune_settings: Mainly parameter configuration for the finetuner.fit
    :param kubectl_path: Path to the kubectl binary on the system

    :return: artifact of finetuned model and token required for FinetunerExecutor
    """
    if pre_trained_head_map is not None and user_input.data in pre_trained_head_map:
        print(f'⚡️ Using cached hub model for speed')
        return pre_trained_head_map[user_input.data]
    dataset = _maybe_add_embeddings(
        encoder_uses, encoder_uses_with, dataset, kubectl_path
    )

    dataset = dataset.shuffle(42)

    if finetune_settings.bi_modal:
        _prepare_dataset_bi_modal(dataset)

    finetune_ds = build_finetuning_dataset(dataset, finetune_settings)

    return _finetune_layer(finetune_ds, finetune_settings)


def _finetune_layer(
    finetune_ds: FinetuneDataset,
    finetune_settings: FinetuneSettings,
) -> Tuple[str, str]:
    assert all([d.embedding is not None for d in finetune_ds.index])

    print('💪 fine-tuning:')
    input_size = (
        finetune_settings.pre_trained_embedding_size
        if not finetune_settings.bi_modal
        else finetune_settings.pre_trained_embedding_size * 2
    )
    finetuner.login()

    callbacks = [
        EvaluationCallback(
            finetune_ds.val_query,
            finetune_ds.val_index,
            limit=finetune_settings.eval_match_limit,
            # metrics=['ndcg'],
        ),
        # BestModelCheckpoint(monitor='ndcg', save_dir=save_dir, verbose=True),
        EarlyStopping(
            monitor='ndcg',
            patience=finetune_settings.early_stopping_patience,
        ),
    ]
    if 'NOW_CI_RUN' in os.environ:
        experiment_name = 'now-ci-finetuning-' + _get_random_string(8)
    else:
        experiment_name = 'now-finetuning-' + _get_random_string(8)
    print(f'🧪 Creating finetune experiment ({experiment_name})')
    finetuner.create_experiment(experiment_name)

    run = finetuner.fit(
        model='mlp',
        model_options={
            'input_size': input_size,
            'hidden_sizes': finetune_settings.hidden_sizes,
            'l2': True,
            'bias': False if finetune_settings.bi_modal else True,
        },
        train_data=finetune_ds.train,
        experiment_name=experiment_name,
        eval_data=finetune_ds.val,
        callbacks=callbacks,
        learning_rate=finetune_settings.learning_rate,
    )

    run_failed = False
    with yaspin_extended(
        sigmap=sigmap, text='Waiting for finetune job to be assigned', color='green'
    ) as spinner:
        while run.status()['status'] == 'CREATED' and not run_failed:
            run_failed = run.status()['status'] == 'FAILED'
            sleep(1)
        if run_failed:
            spinner.fail('👎')
            print(run.logs())
            sys.exit(0)
        else:
            spinner.ok('👍')

    with yaspin_extended(
        sigmap=sigmap, text='Running finetune job', color='green'
    ) as spinner:
        while run.status()['status'] == 'STARTED' and not run_failed:
            run_failed = run.status()['status'] == 'FAILED'
            sleep(1)
        if run_failed:
            spinner.fail('👎')
            print(run.logs())
            sys.exit(0)
        else:
            spinner.ok('👍')

    if run.status()['status'] == 'FAILED':
        print('❌ finetune failed. See logs for details')
        print(run.status())
        print(run.logs())
        sys.exit(0)

    print('🧠 Perfect! Early stopping triggered since accuracy is great already')

    finetune_artifact = run.artifact_id
    token = finetuner.get_token()

    return finetune_artifact, token


def _maybe_add_embeddings(
    encoder_uses: str,
    encoder_uses_with: Dict,
    dataset: DocumentArray,
    kubectl_path: str,
):
    with yaspin_extended(
        sigmap=sigmap, text="Check if embeddings already exist", color="green"
    ) as spinner:
        if all([d.embedding is not None for d in dataset]):
            spinner.ok('👍')
            return dataset
        else:
            spinner.fail('👎')

    embed_now(encoder_uses, encoder_uses_with, dataset, kubectl_path=kubectl_path)

    assert all([d.embedding is not None for d in dataset]), (
        "Some docs slipped through and" " still have no embedding..."
    )


def get_bi_modal_embedding(doc) -> np.ndarray:
    attributes = [doc.text, doc.blob]
    if not any(attributes) or all(attributes):
        raise ValueError(
            f'Received doc (id={doc.id}) with either no text and blob or both.'
        )
    zeros = np.zeros(doc.embedding.shape)
    if doc.text:
        order = (zeros, doc.embedding)
    else:
        order = (doc.embedding, zeros)
    return np.concatenate(order, dtype=np.float32)


def _prepare_dataset_bi_modal(dataset: DocumentArray):
    for doc in dataset:
        doc.embedding = get_bi_modal_embedding(doc)


def _get_random_string(length) -> str:
    import time

    t = 1000 * time.time()  # current time in milliseconds
    random.seed(int(t) % 2**32)
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))
