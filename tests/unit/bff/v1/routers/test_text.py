def test_search_text(test_client, backend_server):
    query = 'hello'
    response = test_client.post(
        f'/api/v1/text/{query}',
        params={},
    )
    assert response.status_code == 200
