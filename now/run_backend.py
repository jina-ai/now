import os
import pathlib
import tempfile
from typing import Dict, Optional, Tuple

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import Apps
from now.data_loading.data_loading import load_data
from now.dataclasses import UserInput
from now.deployment.flow import deploy_flow
from now.finetuning.embeddings import get_encoder_config
from now.finetuning.run_finetuning import finetune_now
from now.finetuning.settings import FinetuneSettings, parse_finetune_settings
from now.log import time_profiler

cur_dir = pathlib.Path(__file__).parent.resolve()


def finetune_flow_setup(
    app_instance: JinaNOWApp,
    dataset: DocumentArray,
    user_input,
    kubectl_path,
    encoder_uses: str,
    artifact: str,
    finetune_datasets: Tuple = (),
    pre_trained_head_map: Optional[Dict] = None,
):
    """
    Apply finetuning if possible, pushes the executor to hub and generated the related yaml file
    """
    finetune_settings = parse_finetune_settings(user_input, dataset, finetune_datasets)
    if finetune_settings.perform_finetuning:
        print(f'🔧 Perform finetuning!')
        finetune_settings.finetuned_model_name = finetune_now(
            user_input, dataset, finetune_settings, pre_trained_head_map, kubectl_path
        )

    finetuning = finetune_settings.perform_finetuning

    yaml_name = get_flow_yaml_name(user_input.app, finetuning)
    app_instance.flow_yaml = os.path.join(cur_dir, 'deployment', 'flow', yaml_name)

    env = get_custom_env_file(user_input, finetune_settings, encoder_uses, artifact)
    return env


@time_profiler
def run(app_instance: JinaNOWApp, user_input: UserInput, kubectl_path: str):
    """
    TODO: Write docs

    :param user_input:
    :param tmpdir:
    :param kubectl_path:
    :return:
    """
    dataset = load_data(app_instance.output_modality, user_input)
    env = app_instance.setup(dataset, user_input, kubectl_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = os.path.join(tmpdir, 'dot.env')
        write_env_file(env_file, env)
        (
            gateway_host,
            gateway_port,
            gateway_host_internal,
            gateway_port_internal,
        ) = deploy_flow(
            user_input=user_input,
            app_instance=app_instance,
            env_file=env_file,
            ns='nowapi',
            index=dataset,
            tmpdir=tmpdir,
            kubectl_path=kubectl_path,
        )
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


def write_env_file(env_file, config):
    config_string = '\n'.join([f'{key}={value}' for key, value in config.items()])
    with open(env_file, 'w+') as fp:
        fp.write(config_string)


def get_custom_env_file(
    user_input: UserInput,
    finetune_settings: FinetuneSettings,
    encoder_uses: str,
    artifact: str,
):
    indexer_name = f'jinahub+docker://DocarrayIndexer'
    encoder_config = get_encoder_config(encoder_uses, artifact)
    linear_head_name = f'jinahub+docker://{finetune_settings.finetuned_model_name}'

    if finetune_settings.bi_modal:
        pre_trained_embedding_size = finetune_settings.pre_trained_embedding_size * 2
    else:
        pre_trained_embedding_size = finetune_settings.pre_trained_embedding_size
    config = {
        'ENCODER_NAME': encoder_config.uses,
        'FINETUNE_LAYER_SIZE': finetune_settings.finetune_layer_size,
        'PRE_TRAINED_EMBEDDINGS_SIZE': pre_trained_embedding_size,
        'INDEXER_NAME': indexer_name,
    }
    if encoder_config.uses_with.get('pretrained_model_name_or_path'):
        config['CLIP_MODEL_NAME'] = encoder_config.uses_with[
            "pretrained_model_name_or_path"
        ]
    if finetune_settings.perform_finetuning:
        config['LINEAR_HEAD_NAME'] = linear_head_name

    return config


def get_flow_yaml_name(app: str, finetuning: bool) -> str:
    options = {
        Apps.TEXT_TO_IMAGE: {False: 'flow-clip.yml', True: 'ft-flow-clip.yml'},
        Apps.IMAGE_TO_IMAGE: {False: 'flow-clip.yml', True: 'ft-flow-clip.yml'},
        Apps.IMAGE_TO_TEXT: {False: 'flow-clip.yml'},
        Apps.MUSIC_TO_MUSIC: {True: 'ft-flow-music.yml'},
    }
    return options[app][finetuning]
