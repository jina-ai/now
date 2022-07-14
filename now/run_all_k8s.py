import json
import os
from os.path import expanduser as user

import cowsay

from now import run_backend, run_bff_playground
from now.cloud_manager import setup_cluster
from now.constants import DOCKER_BFF_PLAYGROUND_TAG, JC_SECRET, SURVEY_LINK
from now.deployment.deployment import cmd, list_all_wolf, status_wolf, terminate_wolf
from now.dialog import _get_context_names, configure_user_input, maybe_prompt_user
from now.log import yaspin_extended
from now.system_information import get_system_state
from now.utils import sigmap


def get_remote_flow_details():
    with open(user(JC_SECRET), 'r') as fp:
        flow_details = json.load(fp)
    return flow_details


def stop_now(contexts, active_context, **kwargs):
    choices = _get_context_names(contexts, active_context)
    # Add remote Flow if it exists
    if os.path.exists(user(JC_SECRET)):
        alive_flows = list_all_wolf(status='ALIVE')
        for flow_details in alive_flows:
            choices.append(flow_details['gateway'])
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
        delete_cluster = maybe_prompt_user(
            [
                {
                    'type': 'list',
                    'name': 'delete-cluster',
                    'message': 'Do you want to delete the entire cluster or just the namespace?',
                    'choices': [
                        {'name': '⛔ no, keep the cluster', 'value': False},
                        {'name': '✅ yes, delete everything', 'value': True},
                    ],
                }
            ],
            attribute='delete-cluster',
            **kwargs,
        )
        if delete_cluster:
            with yaspin_extended(
                sigmap=sigmap, text=f"Remove local cluster {cluster}", color="green"
            ) as spinner:
                cmd(f'{kwargs["kind_path"]} delete clusters jina-now')
                spinner.ok('💀')
            cowsay.cow('local jina NOW cluster removed')
        else:
            with yaspin_extended(
                sigmap=sigmap,
                text=f"Remove namespace nowapi NOW from {cluster}",
                color="green",
            ) as spinner:
                cmd(f'{kwargs["kubectl_path"]} delete ns nowapi')
                spinner.ok('💀')
            cowsay.cow(f'nowapi namespace removed from {cluster}')
    elif 'wolf.jina.ai' in cluster:
        flow_details = get_remote_flow_details()
        flow_id = flow_details['flow_id']
        _result = status_wolf(flow_id)
        if _result is None:
            print(f'❎ Flow not found in JCloud. Likely, it has been deleted already')
        if _result is not None and _result['status'] == 'ALIVE':
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


def start_now(os_type, arch, contexts, active_context, **kwargs):
    app_instance, user_input = configure_user_input(
        contexts=contexts,
        active_context=active_context,
        os_type=os_type,
        arch=arch,
        **kwargs,
    )

    setup_cluster(user_input, **kwargs)
    (
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = run_backend.run(app_instance, user_input, kubectl_path=kwargs['kubectl_path'])

    if gateway_host == 'localhost' or 'NOW_CI_RUN' in os.environ:
        # only deploy playground when running locally or when testing
        bff_playground_host, bff_port, playground_port = run_bff_playground.run(
            output_modality=user_input.output_modality,
            dataset=user_input.data,
            gateway_host=gateway_host,
            gateway_host_internal=gateway_host_internal,
            gateway_port_internal=gateway_port_internal,
            docker_bff_playground_tag=DOCKER_BFF_PLAYGROUND_TAG,
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
        + f'/api/v1/{app_instance.input_modality}-to-{app_instance.output_modality}/redoc'
    )
    playground_url = (
        bff_playground_host
        + ('' if str(playground_port) == '80' else f':{playground_port}')
        + (
            f'/?host='
            + (gateway_host_internal if gateway_host != 'localhost' else 'gateway')
            + f'&input_modality={app_instance.input_modality}'
            + f'&output_modality={app_instance.output_modality}'
            f'&data={user_input.data}'
        )
        + (f'&port={gateway_port_internal}' if gateway_port_internal else '')
    )
    print()
    print(f'BFF docs are accessible at:\n{bff_url}')
    print(f'Playground is accessible at:\n{playground_url}')
    return {
        'bff': bff_url,
        'playground': playground_url,
        'input_modality': app_instance.input_modality,
        'output_modality': app_instance.output_modality,
        'host': gateway_host_internal,
        'port': gateway_port_internal,
    }


def run_k8s(os_type: str = 'linux', arch: str = 'x86_64', **kwargs):
    contexts, active_context = get_system_state(**kwargs)
    task = get_task(kwargs)
    if task == 'start':
        return start_now(os_type, arch, contexts, active_context, **kwargs)
    elif task == 'stop':
        return stop_now(contexts, active_context, **kwargs)
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
