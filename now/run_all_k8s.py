import tempfile

import cowsay
from yaspin import yaspin

from now import run_backend, run_frontend
from now.cloud_manager import setup_cluster
from now.deployment.deployment import cmd
from now.dialog import get_context_names, get_user_input, prompt_plus
from now.system_information import get_system_state
from now.utils import sigmap


def stop_now(contexts, active_context, **kwargs):
    choices = get_context_names(contexts, active_context)
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
        cluster = prompt_plus(questions, 'cluster')
    if cluster == 'kind-jina-now':
        with yaspin(
            sigmap=sigmap, text=f"Remove local cluster {cluster}", color="green"
        ) as spinner:
            cmd(f'{kwargs["kind_path"]} delete clusters jina-now')
            spinner.ok('💀')
        cowsay.cow('local jina NOW cluster removed')
    else:
        with yaspin(
            sigmap=sigmap, text=f"Remove jina NOW from {cluster}", color="green"
        ) as spinner:
            cmd(f'{kwargs["kubectl_path"]} delete ns nowapi')
            spinner.ok('💀')
        cowsay.cow(f'nowapi namespace removed from {cluster}')


def run_k8s(os_type='linux', arch='x86_64', **kwargs):
    contexts, active_context, is_debug = get_system_state()
    if ('cli' in kwargs and kwargs['cli'] == 'stop') or (
        'now' in kwargs and kwargs['now'] == 'stop'
    ):
        stop_now(contexts, active_context, **kwargs)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            docker_frontend_tag = '0.0.1'

            user_input = get_user_input(
                contexts, active_context, os_type, arch, **kwargs
            )
            setup_cluster(user_input.cluster, user_input.new_cluster_type, **kwargs)
            (
                gateway_host,
                gateway_port,
                gateway_host_internal,
                gateway_port_internal,
            ) = run_backend.run(user_input, is_debug, tmpdir, **kwargs)
            frontend_host, frontend_port = run_frontend.run(
                user_input.dataset,
                gateway_host,
                gateway_port,
                gateway_host_internal,
                gateway_port_internal,
                docker_frontend_tag,
                tmpdir,
                **kwargs,
            )
            url = f'{frontend_host}' + (
                '' if str(frontend_port) == '80' else f':{frontend_port}'
            )
            print()

            # print(f'✅ Your search case running.\nhost: {node_ip}:30080')
            # print(f'host: {node_ip}:30080')
            cowsay.cow(f'You made it:\n{url}')


if __name__ == '__main__':
    run_k8s()
