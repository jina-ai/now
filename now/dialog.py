"""
This module implements a command-line dialog with the user.
Its goal is to configure a UserInput object with users specifications.
Optionally, values can be passed from the command-line when jina-now is launched. In that case,
the dialog won't ask for the value.
"""
from __future__ import annotations, print_function, unicode_literals

import os
import pathlib
from dataclasses import dataclass
from os.path import expanduser as user
from typing import Dict, List, Optional, Union

import cowsay
from kubernetes import client, config
from pyfiglet import Figlet
from yaspin import yaspin

from now.constants import (
    AVAILABLE_DATASET,
    IMAGE_MODEL_QUALITY_MAP,
    DatasetTypes,
    Modalities,
    Qualities,
)
from now.deployment.deployment import cmd
from now.thridparty.PyInquirer import Separator
from now.thridparty.PyInquirer.prompt import prompt
from now.utils import ffmpeg_is_installed, gcloud_is_installed, sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()
NEW_CLUSTER = {'name': '🐣 create new', 'value': 'new'}
AVAILABLE_SOON = 'will be available in upcoming versions'


@dataclass
class UserInput:
    output_modality: Optional[Modalities] = None

    # data related
    data: Optional[str] = None
    is_custom_dataset: Optional[bool] = None

    custom_dataset_type: Optional[DatasetTypes] = None
    dataset_secret: Optional[str] = None
    dataset_url: Optional[str] = None
    dataset_path: Optional[str] = None

    # model related
    quality: Optional[Qualities] = None
    sandbox: bool = False
    model_variant: Optional[str] = None

    # cluster related
    cluster: Optional[str] = None
    create_new_cluster: Optional[bool] = None
    new_cluster_type: Optional[str] = None


def configure_user_input(**kwargs) -> UserInput:
    print_headline()

    user_input = UserInput()
    _configure_output_modality(user_input, **kwargs)
    _configure_dataset(user_input, **kwargs)
    _configure_quality(user_input, **kwargs)
    _configure_cluster(user_input, **kwargs)
    _configure_sandbox(user_input, **kwargs)

    return user_input


def maybe_prompt_user(questions, attribute, **kwargs):
    """
    Checks the `kwargs` for the `attribute` name. If present, the value is returned directly.
    If not, the user is prompted via the cmd-line using the `questions` argument.

    :param questions: A dictionary that is passed to `PyInquirer.prompt`
        See docs: https://github.com/CITGuru/PyInquirer#documentation
    :param attribute: Name of the value to get. Make sure this matches the name in `kwargs`

    :return: A single value of either from `kwargs` or the user cli input.
    """
    if kwargs and kwargs.get(attribute) is not None:
        return kwargs[attribute]
    else:
        answer = prompt(questions)
        if attribute in answer:
            return answer[attribute]
        else:
            print("\n" * 10)
            cowsay.cow('see you soon 👋')
            exit(0)


def print_headline():
    f = Figlet(font='slant')
    print('Welcome to:')
    print(f.renderText('Jina NOW'))
    print('Get your search case up and running - end to end.\n')
    print(
        'You can choose between image and text search. \nJina now trains a model, pushes it to the jina hub'
        ', deploys a flow and a frontend app in the cloud or locally. \nCheckout one of the demo cases or bring '
        'your own data.\n'
    )
    print(
        'If you want learn more about our framework please visit: https://docs.jina.ai/'
    )
    print(
        '💡 Make sure you give enough memory to your Docker daemon. '
        '5GB - 8GB should be okay.'
    )
    print()


def _configure_output_modality(user_input: UserInput, **kwargs):
    modality = _prompt_value(
        name='output_modality',
        choices=[
            {'name': '🏞 Image Search', 'value': Modalities.IMAGE},
            {'name': '📝 Text Search (experimental)', 'value': Modalities.TEXT},
            {
                'name': '🥁 Music Search',
                'value': Modalities.MUSIC,
                'disabled': AVAILABLE_SOON,
            },
        ],
        prompt_message='Which modalities you want to work with?',
        prompt_type='list',
        **kwargs,
    )
    user_input.output_modality = modality


def _configure_dataset(user_input: UserInput, **kwargs):
    if user_input.output_modality == Modalities.IMAGE:
        _configure_dataset_image(user_input, **kwargs)
    elif user_input.output_modality == Modalities.TEXT:
        _configure_dataset_text(user_input, **kwargs)
    else:
        if not ffmpeg_is_installed():
            _handle_ffmpeg_install_required()
        _configure_dataset_music(user_input, **kwargs)

    dataset = user_input.data
    if dataset in AVAILABLE_DATASET[user_input.output_modality]:
        user_input.is_custom_dataset = False
    else:
        user_input.is_custom_dataset = True
        if dataset == 'custom':
            _configure_custom_dataset(user_input, **kwargs)
        else:
            _parse_custom_data_from_cli(dataset, user_input)


def _configure_dataset_image(user_input: UserInput, **kwargs):
    dataset = _prompt_value(
        name='data',
        prompt_message='What dataset do you want to use?',
        choices=[
            {'name': '🖼  artworks (≈8K docs)', 'value': 'best-artworks'},
            {
                'name': '💰 nft - bored apes (10K docs)',
                'value': 'nft-monkey',
            },
            {'name': '👬 totally looks like (≈12K docs)', 'value': 'tll'},
            {'name': '🦆 birds (≈12K docs)', 'value': 'bird-species'},
            {'name': '🚗 cars (≈16K docs)', 'value': 'stanford-cars'},
            {
                'name': '🏞 geolocation (≈50K docs)',
                'value': 'geolocation-geoguessr',
            },
            {'name': '👕 fashion (≈53K docs)', 'value': 'deepfashion'},
            {
                'name': '☢️ chest x-ray (≈100K docs)',
                'value': 'nih-chest-xrays',
            },
            Separator(),
            {
                'name': '✨ custom',
                'value': 'custom',
            },
        ],
        **kwargs,
    )
    user_input.data = dataset


def _configure_dataset_text(user_input: UserInput, **kwargs):
    dataset = _prompt_value(
        name='data',
        prompt_message='What dataset do you want to use?',
        choices=[
            {'name': '🎤 rock lyrics (200K docs)', 'value': 'rock-lyrics'},
            {'name': '🎤 pop lyrics (200K docs)', 'value': 'pop-lyrics'},
            {'name': '🎤 rap lyrics (200K docs)', 'value': 'rap-lyrics'},
            {'name': '🎤 indie lyrics (200K docs)', 'value': 'indie-lyrics'},
            {'name': '🎤 metal lyrics (200K docs)', 'value': 'metal-lyrics'},
        ],
        **kwargs,
    )
    user_input.data = dataset


def _configure_dataset_music(user_input: UserInput, **kwargs):
    dataset = _prompt_value(
        name='data',
        prompt_message='What dataset do you want to use?',
        choices=[
            {'name': '🎸 music small (≈2K docs)', 'value': 'music-genres-small'},
            {'name': '🎸 music large (≈10K docs)', 'value': 'music-genres-large'},
            Separator(),
            {
                'name': '✨ custom',
                'value': 'custom',
            },
        ],
        **kwargs,
    )
    user_input.data = dataset


def _configure_sandbox(user_input: UserInput, **kwargs):
    user_input.sandbox = _prompt_value(
        name='sandbox',
        prompt_message='Use Sandbox to save memory? (process data on our servers)',
        choices=[
            {'name': '⛔ no', 'value': False},
            {'name': '✅ yes', 'value': True},
        ],
        **kwargs,
    )


def _configure_custom_dataset(user_input: UserInput, **kwargs):
    custom_dataset_type = _prompt_value(
        name='custom_dataset_type',
        prompt_message='How do you want to provide input? (format: https://docarray.jina.ai/)',
        choices=[
            {
                'name': 'docarray.pull id (recommended)',
                'value': 'docarray',
            },
            {
                'name': 'docarray URL',
                'value': 'url',
            },
            {
                'name': 'local path',
                'value': 'path',
            },
        ],
        **kwargs,
    )

    user_input.custom_dataset_type = custom_dataset_type
    if custom_dataset_type == 'docarray':
        user_input.dataset_secret = _prompt_value(
            name='dataset_secret',
            prompt_message='Please enter your docarray secret.',
            prompt_type='password',
        )

    elif custom_dataset_type == 'url':
        user_input.dataset_url = _prompt_value(
            name='dataset_url',
            prompt_message='Please paste in your url for the docarray.',
            prompt_type='input',
        )

    elif custom_dataset_type == 'path':
        user_input.dataset_path = _prompt_value(
            name='dataset_path',
            prompt_message='Please enter the path to the local folder.',
            prompt_type='input',
        )


def _configure_cluster(user_input: UserInput, **kwargs):
    def configure_new_cluster():
        new_cluster_type = _prompt_value(
            name='new_cluster_type',
            choices=[
                {
                    'name': '📍 local (Kubernetes in Docker)',
                    'value': 'local',
                },
                {'name': '⛅️ Google Kubernetes Engine', 'value': 'gke'},
                {
                    'name': '⛅️ Jina - Flow as a Service',
                    'disabled': AVAILABLE_SOON,
                },
                {
                    'name': '⛅️ Amazon Elastic Kubernetes Service',
                    'disabled': AVAILABLE_SOON,
                },
                {
                    'name': '⛅️ Azure Kubernetes Service',
                    'disabled': AVAILABLE_SOON,
                },
                {
                    'name': '⛅️ DigitalOcean Kubernetes',
                    'disabled': AVAILABLE_SOON,
                },
            ],
            prompt_message='Where do you want to create a new cluster?',
            prompt_type='list',
            **kwargs,
        )
        user_input.create_new_cluster = True
        user_input.new_cluster_type = new_cluster_type
        if user_input.new_cluster_type == 'gke':
            _maybe_install_gke(**kwargs)

    choices = _construct_cluster_choices(
        active_context=kwargs.get('active_context'), contexts=kwargs.get('contexts')
    )
    cluster = _prompt_value(
        name='cluster',
        choices=choices,
        prompt_message='Where do you want to deploy your search engine?',
        prompt_type='list',
        **kwargs,
    )
    if cluster == NEW_CLUSTER['value']:
        user_input.cluster = cluster
        configure_new_cluster()
    else:
        user_input.cluster = cluster
        if not _cluster_running(cluster):
            print(f'Cluster {cluster} is not running. Please select a different one.')
            _configure_cluster(user_input, **kwargs)


def _configure_quality(user_input: UserInput, **kwargs):
    if user_input.output_modality == Modalities.MUSIC:
        return
    quality = _prompt_value(
        name='quality',
        choices=[
            {'name': '🦊 medium (≈3GB mem, 15q/s)', 'value': Qualities.MEDIUM},
            {'name': '🐻 good (≈3GB mem, 2.5q/s)', 'value': Qualities.GOOD},
            {
                'name': '🦄 excellent (≈4GB mem, 0.5q/s)',
                'value': Qualities.EXCELLENT,
            },
        ],
        prompt_message='What quality do you expect?',
        prompt_type='list',
        **kwargs,
    )
    if quality == 'medium':
        print('  🚀 you trade-off a bit of quality for having the best speed')
    elif quality == 'good':
        print('  ⚖️ you have the best out of speed and quality')
    elif quality == 'excellent':
        print('  ✨ you trade-off speed to having the best quality')

    user_input.quality = quality
    _, user_input.model_variant = IMAGE_MODEL_QUALITY_MAP[quality]


def _construct_cluster_choices(active_context, contexts):
    context_names = _get_context_names(contexts, active_context)
    choices = [NEW_CLUSTER]
    if len(context_names) > 0 and len(context_names[0]) > 0:
        choices = context_names + choices
    return choices


def _prompt_value(
    name: str,
    prompt_message: str,
    prompt_type: str = 'input',
    choices: Optional[List[Union[Dict, str]]] = None,
    **kwargs: Dict,
):
    qs = {'name': name, 'type': prompt_type, 'message': prompt_message}

    if choices is not None:
        qs['choices'] = choices
        qs['type'] = 'list'
    return maybe_prompt_user(qs, name, **kwargs)


def _get_context_names(contexts, active_context=None):
    names = [c for c in contexts] if contexts is not None else []
    if active_context is not None:
        names.remove(active_context)
        names = [active_context] + names
    return names


def _cluster_running(cluster):
    config.load_kube_config(context=cluster)
    v1 = client.CoreV1Api()
    try:
        v1.list_namespace()
    except Exception as e:
        return False
    return True


def _maybe_install_gke(os_type: str, arch: str):
    if not gcloud_is_installed():
        if not os.path.exists(user('~/.cache/jina-now/google-cloud-sdk')):
            with yaspin(
                sigmap=sigmap, text='Setting up gcloud', color='green'
            ) as spinner:
                cmd(
                    f'/bin/bash {cur_dir}/scripts/install_gcloud.sh {os_type} {arch}',
                )
                spinner.ok('🛠️')


def _parse_custom_data_from_cli(data: str, user_input: UserInput):
    try:
        data = os.path.expanduser(data)
    except Exception:
        pass
    if os.path.exists(data):
        user_input.custom_dataset_type = DatasetTypes.PATH
        user_input.dataset_path = data
    elif 'http' in data:
        user_input.custom_dataset_type = DatasetTypes.URL
        user_input.dataset_url = data
    else:
        user_input.custom_dataset_type = DatasetTypes.DOCARRAY
        user_input.dataset_secret = data


def _handle_ffmpeg_install_required():
    bc_red = '\033[91m'
    bc_end = '\033[0m'
    print()
    print(
        f"{bc_red}Too use the audio modality you need the ffmpeg audio processing"
        f" library installed on your system.{bc_end}"
    )
    print(
        f"{bc_red}For MacOS please run 'brew install ffmpeg' and on"
        f" Linux 'apt-get install ffmpeg libavcodec-extra'.{bc_end}"
    )
    print(
        f"{bc_red}After the installation, restart Jina Now and have fun with music search 🎸!{bc_end}"
    )
    cowsay.cow('see you soon 👋')
    exit(1)
