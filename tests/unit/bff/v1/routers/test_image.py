def test_search_image_with_text(test_client, backend_server):
    query = 'hello'
    response = test_client.post(
        f'/api/v1/image/{query}',
        params={},
    )
    assert response.status_code == 200
