import json
import pathlib
from typing import Optional

import cowsay
import docker
from kubernetes import client, config
from yaspin import yaspin

from now.deployment.deployment import cmd
from now.dialog import prompt_plus
from now.gke_deploy import create_gke_cluster
from now.utils import sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


def create_local_cluster(kind_path):
    out, err = cmd(f'{kind_path} get clusters')
    if err and 'No kind clusters' not in err.decode('utf-8'):
        print(err.decode('utf-8'))
        exit()
    cluster_name = 'jina-now'
    if cluster_name in out.decode('utf-8'):
        questions = [
            {
                'type': 'list',
                'name': 'proceed',
                'message': 'The local cluster is running already. '
                'Should it be recreated?',
                'choices': [
                    {'name': '⛔ no', 'value': False},
                    {'name': '✅ yes', 'value': True},
                ],
            },
        ]
        recreate = prompt_plus(questions, 'proceed')
        if recreate:
            with yaspin(
                sigmap=sigmap, text="Remove local cluster", color="green"
            ) as spinner:
                cmd(f'{kind_path} delete clusters {cluster_name}')
                spinner.ok('💀')
        else:
            cowsay.cow('see you soon 👋')
            exit(0)
    with yaspin(
        sigmap=sigmap, text="Setting up local cluster", color="green"
    ) as spinner:
        kindest_images = docker.from_env().images.list('kindest/node')
        if len(kindest_images) == 0:
            print(
                'Download kind image to set up local cluster - this might take a while :)'
            )
        _, err = cmd(
            f'{kind_path} create cluster --name {cluster_name} --config {cur_dir}/kind.yml',
        )
        if err and 'failed to create cluster' in err.decode('utf-8'):
            print('\n' + err.decode('utf-8').split('ERROR')[-1])
            exit(1)
        spinner.ok("📦")


def is_local_cluster(**kwargs):
    command = f'{kwargs["kubectl_path"]} get nodes -o json'
    out, error = cmd(f'{kwargs["kubectl_path"]} get nodes -o json')
    try:
        out = json.loads(out)
    except:
        print(f'Command {command} gives the following error: {error.decode("utf-8")}')
        exit(1)
    addresses = out['items'][0]['status']['addresses']
    is_local = len([a for a in addresses if a['type'] == 'ExternalIP']) == 0
    return is_local


def setup_cluster(
    cluster_name: Optional[str],
    provider: str,
    kubectl_path='kubectl',
    kind_path='kind',
    **kwargs,
):
    if cluster_name is not None:
        cmd(f'{kubectl_path} config use-context {cluster_name}')
        ask_existing(kubectl_path)
    else:
        if provider == 'local':
            create_local_cluster(kind_path)
        elif provider == 'gke':
            create_gke_cluster()


def ask_existing(kubectl_path):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    if 'nowapi' in [item.metadata.name for item in v1.list_namespace().items]:
        questions = [
            {
                'type': 'list',
                'name': 'proceed',
                'message': (
                    'jina-now is deployed already. Do you want to remove the '
                    'current data?'
                ),
                'choices': [
                    {'name': '⛔ no', 'value': False},
                    {'name': '✅ yes', 'value': True},
                ],
            },
        ]
        remove = prompt_plus(questions, 'proceed')
        if remove:
            with yaspin(
                sigmap=sigmap, text="Remove old deployment", color="green"
            ) as spinner:
                cmd(f'{kubectl_path} delete ns nowapi')
                spinner.ok('💀')
        else:
            cowsay.cow('see you soon 👋')
            exit(0)
