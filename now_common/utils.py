from typing import Dict, Optional, Tuple

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import PREFETCH_NR
from now.dataclasses import UserInput
from now.finetuning.run_finetuning import maybe_finetune
from now.finetuning.settings import FinetuneSettings, parse_finetune_settings


def get_clip_music_flow_env_dict(
    finetune_settings: FinetuneSettings,
    encoder_uses: str,
    encoder_uses_with: Dict,
    indexer_uses: str,
):
    """Returns dictionary for the environments variables for the clip & music flow.yml files."""
    indexer_name = f'jinahub+docker://' + indexer_uses
    encoder_name = f'jinahub+docker://' + encoder_uses
    linear_head_name = f'jinahub+docker://{finetune_settings.finetuned_model_name}'

    if finetune_settings.bi_modal:
        pre_trained_embedding_size = finetune_settings.pre_trained_embedding_size * 2
    else:
        pre_trained_embedding_size = finetune_settings.pre_trained_embedding_size
    config = {
        'ENCODER_NAME': encoder_name,
        'FINETUNE_LAYER_SIZE': finetune_settings.finetune_layer_size,
        'PRE_TRAINED_EMBEDDINGS_SIZE': pre_trained_embedding_size,
        'INDEXER_NAME': indexer_name,
        'PREFETCH': PREFETCH_NR,
    }
    if encoder_uses_with.get('pretrained_model_name_or_path'):
        config['PRE_TRAINED_MODEL_NAME'] = encoder_uses_with[
            "pretrained_model_name_or_path"
        ]
    if finetune_settings.perform_finetuning:
        config['LINEAR_HEAD_NAME'] = linear_head_name

    return config


def setup_clip_music_apps(
    app_instance: JinaNOWApp,
    user_input: UserInput,
    dataset: DocumentArray,
    encoder_uses: str,
    encoder_uses_with: Dict,
    indexer_uses: str,
    kubectl_path: str,
    finetune_datasets: Optional[Tuple] = (),
    pre_trained_head_map: Optional[Dict] = None,
) -> Dict:
    finetune_settings = parse_finetune_settings(
        app_instance=app_instance,
        user_input=user_input,
        dataset=dataset,
        finetune_datasets=finetune_datasets,
    )

    env_dict = get_clip_music_flow_env_dict(
        finetune_settings=finetune_settings,
        encoder_uses=encoder_uses,
        encoder_uses_with=encoder_uses_with,
        indexer_uses=indexer_uses,
    )

    if finetune_settings.perform_finetuning:
        maybe_finetune(
            finetune_settings=finetune_settings,
            app_instance=app_instance,
            dataset=dataset,
            user_input=user_input,
            env_dict=env_dict,
            kubectl_path=kubectl_path,
            pre_trained_head_map=pre_trained_head_map,
        )

    app_instance.set_flow_yaml(finetuning=finetune_settings.perform_finetuning)

    return env_dict
