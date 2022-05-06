import pathlib
from time import sleep

import requests

from now.deployment.deployment import apply_replace, cmd
from now.deployment.flow import wait_for_lb
from now.log.log import yaspin_extended
from now.utils import sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


def run(
    output_modality,
    dataset,
    gateway_host,
    gateway_port,
    gateway_host_internal,
    gateway_port_internal,
    docker_frontend_tag,
    tmpdir,
    kubectl_path,
):
    # deployment
    with yaspin_extended(
        sigmap=sigmap, text="Deploy frontend", color="green"
    ) as spinner:
        apply_replace(
            f'{cur_dir}/deployment/k8s_frontend-deployment.yml',
            {
                'output_modality': output_modality,
                'data': dataset,
                'gateway_host': gateway_host_internal,
                'gateway_port': gateway_port_internal,
                'docker_frontend_tag': docker_frontend_tag,
            },
            kubectl_path,
        )

        if gateway_host == 'localhost':
            cmd(
                f'{kubectl_path} apply -f {cur_dir}/deployment/k8s_frontend-svc-node.yml'
            )
            while True:
                try:
                    url = 'http://localhost:30080'
                    requests.get(url)
                    break
                except Exception:
                    sleep(1)
            frontend_host = 'http://localhost'
            frontend_port = '30080'
        else:
            cmd(f'{kubectl_path} apply -f {cur_dir}/deployment/k8s_frontend-svc-lb.yml')
            frontend_host = f'http://{wait_for_lb("frontend-lb", "nowapi")}'
            frontend_port = '80'

        spinner.ok('🚀')
        return frontend_host, frontend_port


if __name__ == '__main__':
    run(
        'best-artworks',
        'remote',
        None,
        'gateway.nowapi.svc.cluster.local',
        '8080',
        docker_frontend_tag='0.0.2',
    )
    # 31080
