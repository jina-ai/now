""" This module contains pre-configurations for finetuning on the demo datasets. """
from dataclasses import dataclass
from typing import Optional

from docarray import DocumentArray

from now.constants import DemoDatasets, Modalities, Qualities
from now.dialog import UserInput

TUNEABLE_DEMO_DATASETS = {
    Modalities.IMAGE: [DemoDatasets.DEEP_FASHION, DemoDatasets.BIRD_SPECIES],
    Modalities.TEXT: [],
    Modalities.MUSIC: [
        DemoDatasets.MUSIC_GENRES_MID,
        DemoDatasets.MUSIC_GENRES_LARGE,
    ],
}


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
    Modalities.IMAGE: {
        Qualities.MEDIUM: 512,
        Qualities.GOOD: 512,
        Qualities.EXCELLENT: 768,
    },
    Modalities.TEXT: {
        Qualities.MEDIUM: 512,
        Qualities.GOOD: 512,
        Qualities.EXCELLENT: 768,
    },
    Modalities.MUSIC: 512,
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
    if user_input.output_modality == Modalities.MUSIC:
        assert user_input.quality is None, 'Music modality has no quality to select.'
        return PRE_TRAINED_EMBEDDING_SIZE[Modalities.MUSIC]
    else:
        assert user_input.quality is not None, (
            f'Missing quality ' f'for modality {user_input.output_modality}.'
        )
        return PRE_TRAINED_EMBEDDING_SIZE[user_input.output_modality][
            user_input.quality
        ]


def _is_finetuning(user_input: UserInput, dataset: DocumentArray) -> bool:
    if user_input.data in TUNEABLE_DEMO_DATASETS[user_input.output_modality]:
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
    user_input: UserInput, dataset: DocumentArray
) -> FinetuneSettings:
    """This function parses the user input configuration into the finetune settings"""
    return FinetuneSettings(
        pre_trained_embedding_size=_get_pre_trained_embedding_size(user_input),
        perform_finetuning=_is_finetuning(user_input, dataset),
        bi_modal=_is_bi_modal(user_input, dataset),
    )
