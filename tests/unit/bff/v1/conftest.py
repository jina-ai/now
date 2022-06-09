import base64
import os

import pytest


@pytest.fixture
def test_index_image(resources_folder_path: str):
    with open(
        os.path.join(resources_folder_path, 'image', '5109112832.jpg'), 'rb'
    ) as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'images': [img_query], 'tags': [{'tag1': 'val1'}]}


@pytest.fixture
def test_index_text():
    return {'texts': ['Hello'], 'tags': [{'tag1': 'val1'}]}


@pytest.fixture
def test_search_image(resources_folder_path: str):
    with open(
        os.path.join(resources_folder_path, 'image', '5109112832.jpg'), 'rb'
    ) as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'image': img_query}


@pytest.fixture
def test_search_text():
    return {'text': 'Hello'}


@pytest.fixture
def test_search_both(resources_folder_path: str):
    with open(
        os.path.join(resources_folder_path, 'image', '5109112832.jpg'), 'rb'
    ) as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'image': img_query, 'text': 'Hello'}
