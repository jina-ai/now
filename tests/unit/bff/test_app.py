import pytest


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_check_liveness(client, modality):
    response = client.get(f'/api/v1/{modality}/ping')
    assert response.status_code == 200
    assert response.json() == 'pong!'


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_read_root(client, modality):
    response = client.get(f'/api/v1/{modality}')
    assert response.status_code == 200


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_get_docs(client, modality):
    response = client.get(f'/api/v1/{modality}/docs')
    assert response.status_code == 200


@pytest.mark.parametrize('modality', ['image', 'text'])
def test_get_redoc(client, modality):
    response = client.get(f'/api/v1/{modality}/redoc')
    assert response.status_code == 200
