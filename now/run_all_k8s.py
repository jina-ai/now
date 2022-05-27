import json
import os
import tempfile
from os.path import expanduser as user

import cowsay

from now import run_backend, run_bff_playground
from now.cloud_manager import setup_cluster
from now.constants import JC_SECRET, SURVEY_LINK
from now.deployment.deployment import cmd, status_wolf, terminate_wolf
from now.dialog import _get_context_names, configure_user_input, maybe_prompt_user
from now.log.log import yaspin_extended
from now.system_information import get_system_state
from now.utils import sigmap

docker_bff_playground_tag = '0.0.42'


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
    elif 'wolf.jina.ai' in cluster:
        _result = status_wolf(flow_id)
        if _result['status'] == 'ALIVE':
            terminate_wolf(flow_id)
        os.remove(user(JC_SECRET))
        cowsay.cow(f'remote Flow `{cluster}` removed')
    else:
        with yaspin_extended(
            sigmap=sigmap, text=f"Remove jina NOW from {cluster}", color="green"
        ) as spinner:
            cmd(f'{kwargs["kubectl_path"]} delete ns nowapi')
            spinner.ok('💀')
        cowsay.cow(f'nowapi namespace removed from {cluster}')


def get_task(kwargs):
    for x in ['cli', 'now']:
        if x in kwargs:
            return kwargs[x]
    raise Exception('kwargs do not contain a task')


def start_now(os_type, arch, contexts, active_context, is_debug, **kwargs):
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
        ) = run_backend.run(user_input, tmpdir, kubectl_path=kwargs['kubectl_path'])

        if gateway_host == 'localhost' or 'NOW_CI_RUN' in os.environ:
            # only deploy playground when running locally or when testing
            bff_playground_host, bff_port, playground_port = run_bff_playground.run(
                output_modality=user_input.output_modality,
                dataset=user_input.data,
                gateway_host=gateway_host,
                gateway_host_internal=gateway_host_internal,
                gateway_port_internal=gateway_port_internal,
                docker_bff_playground_tag=docker_bff_playground_tag,
                kubectl_path=kwargs['kubectl_path'],
            )
        else:
            bff_playground_host = 'https://nowrun.jina.ai'
            bff_port = '80'
            playground_port = '80'
        # TODO: add separate BFF endpoints in print output
        bff_url = (
            bff_playground_host
            + ('' if str(bff_port) == '80' else f':{bff_port}')
            + f'/api/docs'
        )
        playground_url = (
            bff_playground_host
            + ('' if str(playground_port) == '80' else f':{playground_port}')
            + (
                f'/?host='
                + (gateway_host_internal if gateway_host != 'localhost' else 'gateway')
                + f'&output_modality={user_input.output_modality}&data={user_input.data}'
            )
            + (f'&port={gateway_port_internal}' if gateway_port_internal else '')
        )
        print()
        print(f'BFF docs are accessible at:\n{bff_url}')
        print(f'Playground is accessible at:\n{playground_url}')


def run_k8s(os_type: str = 'linux', arch: str = 'x86_64', **kwargs):
    contexts, active_context, is_debug = get_system_state(**kwargs)
    task = get_task(kwargs)
    if task == 'start':
        start_now(os_type, arch, contexts, active_context, is_debug, **kwargs)
    elif task == 'stop':
        stop_now(contexts, active_context, **kwargs)
    elif task == 'survey':
        import webbrowser

        webbrowser.open(SURVEY_LINK, new=0, autoraise=True)
    else:
        raise Exception(f'unknown task, {task}')


if __name__ == '__main__':
    run_k8s(
        output_modality='music',
        data='music-genres-mid',
        cluster='new',
        quality='medium',
        deployment_type='local',
        kubectl_path='/usr/local/bin/kubectl',
        kind_path='/Users/sebastianlettner/.cache/jina-now/kind',
        proceed=True,
        now='start',
    )
