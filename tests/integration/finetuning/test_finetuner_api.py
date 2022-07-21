""" This module implements integrations tests against the external finetune-api """

import numpy as np
import pytest
from docarray import Document, DocumentArray

from now.finetuning.dataset import FinetuneDataset, build_finetuning_dataset
from now.finetuning.settings import FinetuneSettings


@pytest.fixture()
def finetune_settings() -> FinetuneSettings:
    return FinetuneSettings(
        perform_finetuning=True, pre_trained_embedding_size=512, bi_modal=False
    )


@pytest.fixture()
def finetune_ds(finetune_settings: FinetuneSettings) -> FinetuneDataset:
    num_classes = 32
    num_images_per_class = 100
    embedding_dim = 512

    train_data = DocumentArray()
    for class_id in range(num_classes):
        for _ in range(num_images_per_class):
            doc = Document(
                embedding=np.random.rand(embedding_dim),
                tags={'finetuner_label': str(class_id)},
            )
            train_data.append(doc)
    return build_finetuning_dataset(train_data, finetune_settings)


# def test_end2end(
#     finetune_ds: FinetuneDataset,
#     finetune_settings: FinetuneSettings,
#     with_hubble_login_patch,
# ):
#
#     with tempfile.TemporaryDirectory() as tempdir:
#         _finetune_layer(
#             finetune_ds=finetune_ds,
#             finetune_settings=finetune_settings,
#             save_dir=tempdir,
#         )
#
#         assert os.path.isfile(
#             os.path.join(tempdir, 'now', 'hub', 'head_encoder', 'best_model_ndcg')
#         )
