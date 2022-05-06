import base64

import pytest


@pytest.fixture
def test_index_image():
    with open('./tests/resources/image/5109112832.jpg', 'rb') as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'images': [img_query]}


@pytest.fixture
def test_index_text():
    return {'texts': ['Hello']}


@pytest.fixture
def test_search_image():
    with open('./tests/resources/image/5109112832.jpg', 'rb') as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'image': img_query}


@pytest.fixture
def test_search_text():
    return {'text': 'Hello'}


@pytest.fixture
def test_search_both():
    with open('./tests/resources/image/5109112832.jpg', 'rb') as f:
        binary = f.read()
        img_query = base64.b64encode(binary).decode('utf-8')
    return {'image': img_query, 'text': 'Hello'}
