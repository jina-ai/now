import json
import math
import os.path
import pathlib
from os.path import expanduser as user
from time import sleep

from kubernetes import client as k8s_client
from kubernetes import config
from tqdm import tqdm
from yaspin.spinners import Spinners

from now.cloud_manager import is_local_cluster
from now.deployment.deployment import apply_replace, cmd
from now.log.log import TEST, yaspin_extended
from now.utils import deploy_wolf, sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def wait_for_lb(lb_name, ns):
    config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    while True:
        try:
            services = v1.list_namespaced_service(namespace=ns)
            ip = [
                s.status.load_balancer.ingress[0].ip
                for s in services.items
                if s.metadata.name == lb_name
            ][0]
            if ip:
                break
        except Exception:
            pass
        sleep(1)
    return ip


def wait_for_all_pods_in_ns(ns, num_pods, max_wait=1800):
    config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    for i in range(max_wait):
        pods = v1.list_namespaced_pod(ns).items
        not_ready = [
            'x'
            for pod in pods
            if not pod.status
            or not pod.status.container_statuses
            or not len(pod.status.container_statuses) == 1
            or not pod.status.container_statuses[0].ready
        ]
        if len(not_ready) == 0 and num_pods == len(pods):
            return
        sleep(1)


def deploy_k8s(f, ns, num_pods, tmpdir, kubectl_path):
    k8_path = os.path.join(tmpdir, f'k8s/{ns}')
    with yaspin_extended(
        sigmap=sigmap, text="Convert Flow to Kubernetes YAML", color="green"
    ) as spinner:
        f.to_k8s_yaml(k8_path)
        spinner.ok('🔄')

    # create namespace
    cmd(f'{kubectl_path} create namespace {ns}')

    # deploy flow
    with yaspin_extended(
        Spinners.earth,
        sigmap=sigmap,
        text="Deploy Jina Flow (might take a bit)",
    ) as spinner:
        gateway_host_internal = f'gateway.{ns}.svc.cluster.local'
        gateway_port_internal = 8080
        if is_local_cluster(kubectl_path):
            apply_replace(
                f'{cur_dir}/k8s_backend-svc-node.yml',
                {'ns': ns},
                kubectl_path,
            )
            gateway_host = 'localhost'
            gateway_port = 31080
        else:
            apply_replace(f'{cur_dir}/k8s_backend-svc-lb.yml', {'ns': ns}, kubectl_path)
            gateway_host = wait_for_lb('gateway-lb', ns)
            gateway_port = 8080
        cmd(f'{kubectl_path} apply -R -f {k8_path}')
        # wait for flow to come up
        wait_for_all_pods_in_ns(ns, num_pods)
        spinner.ok("🚀")
    # work around - first request hangs
    sleep(3)
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


def get_custom_env_file(
    indexer_name,
    encoder_name,
    linear_head_name,
    model,
    output_dim,
    embed_size,
    tmpdir,
):
    env_file = os.path.join(user('~/.jina'), 'dot.env')
    with open(env_file, 'w+') as fp:
        fp.write(
            f'ENCODER_NAME={encoder_name}\n'
            f'CLIP_MODEL_NAME={model}\n'
            f'LINEAR_HEAD_NAME={linear_head_name}\n'
            f'OUTPUT_DIM={output_dim}\n'
            f'EMBED_DIM={embed_size}\n'
            f'INDEXER_NAME={indexer_name}\n'
        )

    return env_file


def deploy_flow(
    executor_name,
    output_modality,
    index,
    vision_model,
    final_layer_output_dim,
    embedding_size,
    tmpdir,
    finetuning,
    sandbox,
    kubectl_path,
    deployment_type,
):
    from jina import Flow
    from jina.clients import Client

    if deployment_type == 'remote':
        indexer_name = (
            'jinahub+docker://MostSimpleIndexer:346e8475359e13d621717ceff7f48c2a'
        )
        encoder_name = 'jinahub+docker://CLIPEncoder/v0.2.1'
        executor_name = f'jinahub+docker://{executor_name}'
    else:
        indexer_name = (
            'jinahub+docker://MostSimpleIndexer:346e8475359e13d621717ceff7f48c2a'
        )
        encoder_name = 'jinahub+docker://CLIPEncoder/v0.2.1'
        executor_name = f'jinahub+docker://{executor_name}'

    env_file = get_custom_env_file(
        indexer_name,
        encoder_name,
        executor_name,
        vision_model,
        final_layer_output_dim,
        embedding_size,
        tmpdir,
    )

    ns = 'nowapi'
    if deployment_type == 'remote':
        # Deploy it on wolf
        if finetuning:
            flow = deploy_wolf(
                os.path.join(cur_dir, 'flow', 'ft-flow.yml'), name=ns, env=env_file
            )
        else:
            flow = deploy_wolf(
                os.path.join(cur_dir, 'flow', 'flow.yml'), name=ns, env=env_file
            )
        host = flow.gateway
        client = Client(host=host)
        with open(user('~/.cache/jina-now/wolf.json'), 'wb+') as fp:
            json.dump({'flow_id': host}, fp)

        # host & port
        gateway_host = 'remote'
        gateway_port = None
        gateway_host_internal = host
        gateway_port_internal = 443  # Since we know it will be `grpcs` - default
    else:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        if finetuning:
            f = Flow.load_config(os.path.join(cur_dir, 'flow', 'ft-flow.yml'))
        else:
            f = Flow.load_config(os.path.join(cur_dir, 'flow', 'flow.yml'))

        (
            gateway_host,
            gateway_port,
            gateway_host_internal,
            gateway_port_internal,
        ) = deploy_k8s(
            f,
            ns,
            2 + (2 if finetuning else 1) * (0 if sandbox else 1),
            tmpdir,
            kubectl_path=kubectl_path,
        )
        client = Client(host=gateway_host, port=gateway_port)

    if output_modality == 'image':
        index = [x for x in index if x.text == '']
    elif output_modality == 'text':
        index = [x for x in index if x.text != '']
    print(f'▶ indexing {len(index)} documents')
    request_size = 64

    progress_bar = (
        x
        for x in tqdm(
            batch(index, request_size),
            total=math.ceil(len(index) / request_size),
        )
    )

    def on_done(res):
        if not TEST:
            next(progress_bar)

    client.post('/index', request_size=request_size, inputs=index, on_done=on_done)

    print('⭐ Success - your data is indexed')
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal
