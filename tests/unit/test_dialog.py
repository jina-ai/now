"""
Test the dialog.py module.

Patches the `prompt` method to mock user input via the command line.
"""
from typing import Dict

import pytest
from pytest_mock import MockerFixture

from now.dialog import (
    IMAGE_MODEL_QUALITY_MAP,
    Modalities,
    UserInput,
    configure_user_input,
)


class CmdPromptMock:
    def __init__(self, predefined_answers: Dict[str, str]):
        self._answers = predefined_answers

    def __call__(self, question: Dict):
        return {question['name']: self._answers[question['name']]}


MOCKED_DIALOGS_WITH_CONFIGS = [
    (
        {
            'output_modality': Modalities.MUSIC,
            'data': 'music-genres-small',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'output_modality': Modalities.MUSIC,
            'data': 'music-genres-large',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'output_modality': Modalities.IMAGE,
            'data': 'tll',
            'cluster': 'new',
            'sandbox': False,
            'quality': 'good',
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['good'][1],
        ),
    ),
    (
        {
            'output_modality': Modalities.IMAGE,
            'data': 'nih-chest-xrays',
            'cluster': 'new',
            'sandbox': False,
            'quality': 'medium',
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['medium'][1],
        ),
    ),
    (
        {
            'output_modality': Modalities.IMAGE,
            'data': 'custom',
            'custom_dataset_type': 'docarray',
            'dataset_secret': 'xxx',
            'cluster': 'new',
            'sandbox': False,
            'quality': 'medium',
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=True,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['medium'][1],
        ),
    ),
    (
        {
            'output_modality': Modalities.MUSIC,
            'data': 'custom',
            'custom_dataset_type': 'docarray',
            'dataset_secret': 'xxx',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=True,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'output_modality': Modalities.MUSIC,
            'data': 'custom',
            'custom_dataset_type': 'path',
            'dataset_path': 'xxx',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=True,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'output_modality': Modalities.MUSIC,
            'data': 'custom',
            'custom_dataset_type': 'url',
            'dataset_url': 'xxx',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=True,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'output_modality': Modalities.IMAGE,
            'data': 'custom',
            'custom_dataset_type': 'docarray',
            'dataset_secret': 'xxx',
            'quality': 'medium',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {},
        UserInput(
            is_custom_dataset=True,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['medium'][1],
        ),
    ),
    (
        {
            'output_modality': Modalities.IMAGE,
            'data': 'tll',
            'cluster': 'new',
            'sandbox': False,
            'quality': 'good',
            'new_cluster_type': 'gke',
        },
        {'os_type': 'darwin', 'arch': 'x86_64'},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['good'][1],
        ),
    ),
    (
        {
            'data': 'music-genres-small',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {'output_modality': Modalities.MUSIC},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
        ),
    ),
    (
        {
            'data': 'tll',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {'output_modality': Modalities.IMAGE, 'quality': 'good'},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['good'][1],
        ),
    ),
    (
        {
            'data': 'pop-lyrics',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
        },
        {'output_modality': Modalities.TEXT, 'quality': 'good'},
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['good'][1],
        ),
    ),
    (
        {
            'output_modality': Modalities.TEXT,
        },
        {
            'data': 'pop-lyrics',
            'cluster': 'new',
            'sandbox': False,
            'new_cluster_type': 'local',
            'quality': 'medium',
        },
        UserInput(
            is_custom_dataset=False,
            create_new_cluster=True,
            model_variant=IMAGE_MODEL_QUALITY_MAP['medium'][1],
        ),
    ),
]


@pytest.mark.parametrize(
    ('mocked_user_answers', 'configure_kwargs', 'expected_user_input'),
    MOCKED_DIALOGS_WITH_CONFIGS,
)
def test_configure_user_input(
    mocker: MockerFixture,
    mocked_user_answers: Dict[str, str],
    configure_kwargs: Dict,
    expected_user_input: UserInput,
):
    expected_user_input.__dict__.update(mocked_user_answers)
    expected_user_input.__dict__.update(configure_kwargs)
    mocker.patch('now.dialog.prompt', CmdPromptMock(mocked_user_answers))
    mocker.patch('now.dialog._maybe_install_gke', lambda os_type, arch: 0)

    user_input = configure_user_input(**configure_kwargs)

    assert user_input == expected_user_input
