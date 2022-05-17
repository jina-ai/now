import json
import os
import tempfile
from os.path import expanduser as user

import cowsay

from now import run_backend, run_frontend
from now.cloud_manager import setup_cluster
from now.constants import JC_SECRET
from now.deployment.deployment import cmd, terminate_wolf
from now.dialog import _get_context_names, configure_user_input, maybe_prompt_user
from now.log.log import yaspin_extended
from now.system_information import get_system_state
from now.utils import sigmap

docker_frontend_tag = '0.0.16'


def get_remote_flow_details():
    with open(user(JC_SECRET), 'r') as fp:
        flow_details = json.load(fp)
    return flow_details


def stop_now(contexts, active_context, **kwargs):
    choices = _get_context_names(contexts, active_context)
    # Add remote Flow if it exists
    if os.path.exists(user(JC_SECRET)):
        flow_details = get_remote_flow_details()
        choices += [flow_details['gateway']]
        flow_id = flow_details['flow_id']
    if len(choices) == 0:
        cowsay.cow('nothing to stop')
        return
    else:
        questions = [
            {
                'type': 'list',
                'name': 'cluster',
                'message': 'Which cluster do you want to delete?',
                'choices': choices,
            }
        ]
        cluster = maybe_prompt_user(questions, 'cluster', **kwargs)
    if cluster == 'kind-jina-now':
        with yaspin_extended(
            sigmap=sigmap, text=f"Remove local cluster {cluster}", color="green"
        ) as spinner:
            cmd(f'{kwargs["kind_path"]} delete clusters jina-now')
            spinner.ok('💀')
        cowsay.cow('local jina NOW cluster removed')
    elif 'nowapi' in cluster:
        with yaspin_extended(
            sigmap=sigmap, text=f"Remove remote Flow {cluster}", color="green"
        ) as spinner:
            terminate_wolf(flow_id)
            os.remove(user(JC_SECRET))
            spinner.ok('💀')
        cowsay.cow(f'remote Flow `{cluster}` removed')
    else:
        with yaspin_extended(
            sigmap=sigmap, text=f"Remove jina NOW from {cluster}", color="green"
        ) as spinner:
            cmd(f'{kwargs["kubectl_path"]} delete ns nowapi')
            spinner.ok('💀')
        cowsay.cow(f'nowapi namespace removed from {cluster}')


def run_k8s(os_type: str = 'linux', arch: str = 'x86_64', **kwargs):
    contexts, active_context, is_debug = get_system_state(**kwargs)
    if ('cli' in kwargs and kwargs['cli'] == 'stop') or (
        'now' in kwargs and kwargs['now'] == 'stop'
    ):
        stop_now(contexts, active_context, **kwargs)
    else:
        user_input = configure_user_input(
            contexts=contexts,
            active_context=active_context,
            os_type=os_type,
            arch=arch,
            **kwargs,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_cluster(user_input, **kwargs)
            (
                gateway_host,
                gateway_port,
                gateway_host_internal,
                gateway_port_internal,
            ) = run_backend.run(
                user_input, is_debug, tmpdir, kubectl_path=kwargs['kubectl_path']
            )

            if 'NOW_CI_RUN' not in os.environ:
                # Do not deploy frontend when testing
                frontend_host, frontend_port = run_frontend.run(
                    output_modality=user_input.output_modality,
                    dataset=user_input.data,
                    gateway_host=gateway_host,
                    gateway_port=gateway_port,
                    gateway_host_internal=gateway_host_internal,
                    gateway_port_internal=gateway_port_internal,
                    docker_frontend_tag=docker_frontend_tag,
                    tmpdir=tmpdir,
                    kubectl_path=kwargs['kubectl_path'],
                )
                url = f'{frontend_host}' + (
                    '' if str(frontend_port) == '80' else f':{frontend_port}'
                )
                print()

                # print(f'✅ Your search case running.\nhost: {node_ip}:30080')
                # print(f'host: {node_ip}:30080')
                cowsay.cow(f'You made it:\n{url}')


if __name__ == '__main__':
    run_k8s(
        output_modality='music',
        data='music-genres-small',
        cluster='new',
        deployment_type='local',
        kubectl_path='/usr/local/bin/kubectl',
    )
