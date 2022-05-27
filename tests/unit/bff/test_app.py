import pytest


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_check_liveness(test_client, modality):
    response = test_client.get(f'api/v1/{modality}/ping')
    assert response.status_code == 200
    assert response.json() == 'pong!'


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_read_root(test_client, modality):
    response = test_client.get(f'/api/v1/{modality}')
    assert response.status_code == 200


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_get_docs(test_client, modality):
    response = test_client.get(f'api/v1/{modality}/docs')
    assert response.status_code == 200


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_get_redoc(test_client, modality):
    response = test_client.get(f'/api/v1/{modality}/redoc')
    assert response.status_code == 200
