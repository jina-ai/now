import numpy as np
import pytest
from docarray import Document, DocumentArray

from now.finetuning.dataset import build_finetuning_dataset
from now.finetuning.settings import FinetuneSettings


@pytest.fixture()
def docs() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                text='hello', embedding=np.array([0, 1]), tags={'finetuner_label': 0}
            )
            for _ in range(5)
        ]
        + [
            Document(
                text='world', embedding=np.array([1, 0]), tags={'finetuner_label': 1}
            )
            for _ in range(5)
        ]
    )


@pytest.fixture()
def finetune_setting() -> FinetuneSettings:
    return FinetuneSettings(
        pre_trained_embedding_size=512,
        perform_finetuning=True,
        bi_modal=True,
        num_val_queries=1,
    )


def test_create_finetuning_dataset_size(
    docs: DocumentArray, finetune_setting: FinetuneSettings
):
    dataset = build_finetuning_dataset(docs, finetune_setting)

    assert len(dataset.index) == len(docs)


def test_create_finetuning_dataset_split(
    docs: DocumentArray, finetune_setting: FinetuneSettings
):
    finetune_setting.train_val_split_ration = 0.6

    dataset = build_finetuning_dataset(docs, finetune_setting)

    assert len(dataset.train) == 6
    assert len(dataset.val) == 4


def test_create_finetuning_dataset_val_index(
    docs: DocumentArray, finetune_setting: FinetuneSettings
):
    finetune_setting.train_val_split_ration = 0.6

    dataset = build_finetuning_dataset(docs, finetune_setting)

    assert len(dataset.val) == len(dataset.val_index)
    assert len(dataset.val_query) == finetune_setting.num_val_queries
