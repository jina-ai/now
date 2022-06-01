""" This module contains pre-configurations for finetuning on the demo datasets. """
from dataclasses import dataclass
from typing import Optional, Tuple

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import Apps, Qualities
from now.dataclasses import UserInput

DEFAULT_EPOCHS = 50
DEFAULT_NUM_VAL_QUERIES = 50
DEFAULT_FINETUNED_EMBEDDING_SIZE = 128
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAIN_VAL_SPLIT_RATIO = 0.9
DEFAULT_EVAL_MATCH_LIMIT = 20
DEFAULT_NUM_ITEMS_PER_CLASS = 4
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_POS_MINING_START = 'hard'
DEFAULT_NEG_MINING_START = 'hard'

PRE_TRAINED_EMBEDDING_SIZE = {
    Apps.TEXT_TO_IMAGE: {
        Qualities.MEDIUM: 512,
        Qualities.GOOD: 512,
        Qualities.EXCELLENT: 768,
    },
    Apps.IMAGE_TO_TEXT: {
        Qualities.MEDIUM: 512,
        Qualities.GOOD: 512,
        Qualities.EXCELLENT: 768,
    },
    Apps.IMAGE_TO_IMAGE: {
        Qualities.MEDIUM: 512,
        Qualities.GOOD: 512,
        Qualities.EXCELLENT: 768,
    },
    Apps.MUSIC_TO_MUSIC: {
        Qualities.MEDIUM: 512,
    },
}


@dataclass
class FinetuneSettings:
    perform_finetuning: bool
    pre_trained_embedding_size: int
    bi_modal: bool  # atm, bi-modal means text and some blob value
    finetuned_model_name: Optional[str] = None

    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    finetune_layer_size: int = DEFAULT_FINETUNED_EMBEDDING_SIZE
    train_val_split_ration: int = DEFAULT_TRAIN_VAL_SPLIT_RATIO
    num_val_queries: int = DEFAULT_NUM_VAL_QUERIES
    eval_match_limit: int = DEFAULT_EVAL_MATCH_LIMIT
    num_items_per_class: int = DEFAULT_NUM_ITEMS_PER_CLASS
    learning_rate: int = DEFAULT_LEARNING_RATE
    pos_mining_strat: str = DEFAULT_POS_MINING_START
    neg_mining_strat: str = DEFAULT_NEG_MINING_START
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE


def _get_pre_trained_embedding_size(user_input: UserInput) -> int:
    """Returns the dimension of embeddings given the configured user input object."""
    return PRE_TRAINED_EMBEDDING_SIZE[user_input.app][user_input.quality]


def _is_finetuning(
    user_input: UserInput, dataset: DocumentArray, finetune_datasets: Tuple
) -> bool:
    if user_input.data in finetune_datasets:
        return True

    elif user_input.is_custom_dataset and all(
        ['finetuner_label' in d.tags for d in dataset]
    ):
        return True
    else:
        return False


def _is_bi_modal(user_input: UserInput, dataset: DocumentArray) -> bool:
    if user_input.is_custom_dataset:
        has_blob = any([d.blob != b'' for d in dataset])
        has_text = any([d.text != '' for d in dataset])
        return has_text and has_blob
    else:
        return True  # right now all demo cases are bi-modal


def parse_finetune_settings(
    app_instance: JinaNOWApp,
    user_input: UserInput,
    dataset: DocumentArray,
    finetune_datasets: Tuple,
) -> FinetuneSettings:
    """This function parses the user input configuration into the finetune settings"""
    return FinetuneSettings(
        pre_trained_embedding_size=app_instance.pre_trained_embedding_size[
            user_input.quality
        ],
        perform_finetuning=_is_finetuning(user_input, dataset, finetune_datasets),
        bi_modal=_is_bi_modal(user_input, dataset),
    )
