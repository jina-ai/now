import os
import pathlib
from time import sleep

import requests

from now.deployment.deployment import apply_replace, cmd
from now.log import time_profiler, yaspin_extended
from now.utils import sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


@time_profiler
def run(
    gateway_host,
    docker_bff_playground_tag,
    kubectl_path,
):
    # deployment
    with yaspin_extended(
        sigmap=sigmap, text="Deploy playground and BFF", color="green"
    ) as spinner:
        apply_replace(
            f'{cur_dir}/deployment/k8s_playground-deployment.yml',
            {
                'docker_bff_playground_tag': docker_bff_playground_tag,
            },
            kubectl_path,
        )

        # remote signifies WOLF - Flow as a service
        if gateway_host == 'localhost' or (
            'NOW_CI_RUN' in os.environ and gateway_host == 'remote'
        ):
            cmd(
                f'{kubectl_path} apply -f {cur_dir}/deployment/k8s_playground-svc-node.yml'
            )
            bff_playground_host = 'http://localhost'
            playground_port = '30080'
            bff_port = '30090'
            while True:
                try:
                    requests.get(f"{bff_playground_host}:{playground_port}")
                    requests.get(f"{bff_playground_host}:{bff_port}")
                    break
                except Exception:
                    sleep(1)
        # else:
        #     cmd(f'{kubectl_path} apply -f {cur_dir}/deployment/k8s_playground-svc-lb.yml')
        #     playground_host = f'http://{wait_for_lb("playground-lb", "nowapi")}'
        #     playground_port = '80'

        spinner.ok('🚀')
        return bff_playground_host, bff_port, playground_port


if __name__ == '__main__':
    run(
        'best-artworks',
        'remote',
        None,
        'gateway.nowapi.svc.cluster.local',
        '8080',
        docker_bff_playground_tag='0.0.2',
    )
    # 31080
