""" Module holds reusable fixtures """

import os

import pytest


@pytest.fixture()
def resources_folder_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


@pytest.fixture(scope='session')
def service_account_file_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'service_account.json'
    )


@pytest.fixture()
def image_resource_path(resources_folder_path: str) -> str:
    return os.path.join(resources_folder_path, 'image')


@pytest.fixture()
def music_resource_path(resources_folder_path: str) -> str:
    return os.path.join(resources_folder_path, 'music')
