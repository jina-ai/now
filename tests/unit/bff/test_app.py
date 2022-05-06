def test_check_liveness(test_client):
    response = test_client.get('/ping')
    assert response.status_code == 200
    assert response.json() == 'pong!'


def test_read_root(test_client):
    response = test_client.get('/')
    assert response.status_code == 200


def test_get_docs(test_client):
    response = test_client.get('/docs')
    assert response.status_code == 200


def test_get_redoc(test_client):
    response = test_client.get('/redoc')
    assert response.status_code == 200
