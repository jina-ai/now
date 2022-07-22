import os
import pathlib
import random
import sys
from time import sleep

from docarray import DocumentArray
from jina.clients import Client

from now.apps.base.app import JinaNOWApp
from now.data_loading.data_loading import load_data
from now.dataclasses import UserInput
from now.deployment.flow import deploy_flow
from now.log import time_profiler

cur_dir = pathlib.Path(__file__).parent.resolve()


@time_profiler
def run(app_instance: JinaNOWApp, user_input: UserInput, kubectl_path: str):
    """
    TODO: Write docs

    :param user_input:
    :param kubectl_path:
    :return:
    """
    dataset = load_data(app_instance.output_modality, user_input)
    env_dict = app_instance.setup(
        dataset=dataset, user_input=user_input, kubectl_path=kubectl_path
    )

    (
        client,
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = deploy_flow(
        deployment_type=user_input.deployment_type,
        flow_yaml=app_instance.flow_yaml,
        env_dict=env_dict,
        ns='nowapi',
        kubectl_path=kubectl_path,
    )

    if app_instance.output_modality == 'image':
        dataset = [x for x in dataset if x.text == '']
    elif app_instance.output_modality == 'text':
        dataset = [x for x in dataset if x.text != '']
    index_data(app_instance=app_instance, client=client, dataset=dataset)

    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


@time_profiler
def index_data(client: Client, dataset: DocumentArray):
    print(f'▶ indexing {len(dataset)} documents')
    request_size = estimate_request_size(dataset)

    # double check that flow is up and running - should be done by wolf/core in the future
    while True:
        try:
            client.post(
                '/index',
                inputs=DocumentArray(),
            )
            break
        except Exception as e:
            if 'NOW_CI_RUN' in os.environ:
                import traceback

                print(e)
                print(traceback.format_exc())
            sleep(1)

    client.post('/index', request_size=request_size, inputs=dataset, show_progress=True)

    print('⭐ Success - your data is indexed')


def estimate_request_size(index):
    if len(index) > 30:
        sample = random.sample(index, 30)
    else:
        sample = index
    size = sum([sys.getsizeof(x.content) for x in sample]) / 30
    max_size = 50000
    max_request_size = 128
    request_size = max(min(max_request_size, int(max_size / size)), 1)
    return request_size
