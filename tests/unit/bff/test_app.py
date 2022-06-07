import pytest


@pytest.mark.parametrize(
    'inp_mod, output_mod', [('text', 'image'), ('image', 'text'), ('image', 'image')]
)
class TestParametrized:
    def test_check_liveness(self, test_client, inp_mod, output_mod):
        response = test_client.get(f'/api/v1/{inp_mod}-to-{output_mod}/ping')
        assert response.status_code == 200
        assert response.json() == 'pong!'

    def test_read_root(self, test_client, inp_mod, output_mod):
        response = test_client.get(f'/api/v1/{inp_mod}-to-{output_mod}')
        assert response.status_code == 200

    def test_get_docs(self, test_client, inp_mod, output_mod):
        response = test_client.get(f'/api/v1/{inp_mod}-to-{output_mod}/docs')
        assert response.status_code == 200

    def test_get_redoc(self, test_client, inp_mod, output_mod):
        response = test_client.get(f'/api/v1/{inp_mod}-to-{output_mod}/redoc')
        assert response.status_code == 200
