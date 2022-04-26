def test_search_image(test_client, backend_server):
    query = 'hello'
    response = test_client.post(
        f'/api/v1/image/{query}',
        params={},
    )
    assert response.status_code == 200
