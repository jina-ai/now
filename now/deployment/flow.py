import math
import os.path
import pathlib
from time import sleep

from kubernetes import client as k8s_client
from kubernetes import config
from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

from now.cloud_manager import is_local_cluster
from now.deployment.deployment import apply_replace, cmd
from now.utils import sigmap

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


def deploy_k8s(f, ns, num_pods, tmpdir, **kwargs):
    k8_path = os.path.join(tmpdir, f'k8s/{ns}')
    with yaspin(
        sigmap=sigmap, text="Convert Flow to Kubernetes YAML", color="green"
    ) as spinner:
        f.to_k8s_yaml(k8_path)
        spinner.ok('🔄')

    # create namespace
    cmd(f'{kwargs["kubectl_path"]} create namespace {ns}')

    # deploy flow
    with yaspin(
        Spinners.earth,
        sigmap=sigmap,
        text="Deploy Jina Flow (might take a bit)",
    ) as spinner:
        gateway_host_internal = f'gateway.{ns}.svc.cluster.local'
        gateway_port_internal = 8080
        if is_local_cluster(**kwargs):
            apply_replace(
                f'{cur_dir}/k8s_backend-svc-node.yml',
                {'ns': ns},
                kwargs["kubectl_path"],
            )
            gateway_host = 'localhost'
            gateway_port = 31080
        else:
            apply_replace(
                f'{cur_dir}/k8s_backend-svc-lb.yml', {'ns': ns}, kwargs["kubectl_path"]
            )
            gateway_host = wait_for_lb('gateway-lb', ns)
            gateway_port = 8080
        cmd(f'{kwargs["kubectl_path"]} apply -R -f {k8_path}')
        # wait for flow to come up
        wait_for_all_pods_in_ns(ns, num_pods)
        spinner.ok("🚀")
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


def deploy_flow(
    executor_name,
    index,
    vision_model,
    final_layer_output_dim,
    embedding_size,
    tmpdir,
    finetuning,
    **kwargs,
):
    from jina import Flow
    from jina.clients import Client

    ns = 'nowapi'
    f = Flow(
        name=ns,
        port_expose=8080,
        cors=True,
    )
    f = f.add(
        name='encoder_clip',
        uses=f'jinahub+docker://CLIPEncoder/v0.2.1',
        uses_with={'pretrained_model_name_or_path': vision_model},
        env={'JINA_LOG_LEVEL': 'DEBUG'},
    )
    if finetuning:
        f = f.add(
            name='linear_head',
            uses=f'jinahub+docker://{executor_name}',
            uses_with={
                'final_layer_output_dim': final_layer_output_dim,
                'embedding_size': embedding_size,
            },
            env={'JINA_LOG_LEVEL': 'DEBUG'},
        )
    f = f.add(
        name='indexer',
        uses=f'jinahub+docker://MostSimpleIndexer:346e8475359e13d621717ceff7f48c2a',
        uses_with={'dim': embedding_size, 'metric': 'cosine'},
        uses_metas={'workspace': 'pq_workspace'},
        env={'JINA_LOG_LEVEL': 'DEBUG'},
    )
    # f.plot('./flow.png', vertical_layout=True)

    index = [x for x in index if x.text == '']

    (
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = deploy_k8s(f, ns, 7 if finetuning else 5, tmpdir, **kwargs)
    print(
        f'▶ indexing {len(index)} documents - if it stays at 0% for a while, it is all good - just wait :)'
    )

    client = Client(host=gateway_host, port=gateway_port)
    for x in tqdm(batch(index, 1024), total=math.ceil(len(index) / 1024)):
        # first request takes soooo long
        while True:
            try:
                client.post('/index', request_size=64, inputs=x)
                break
            except Exception as e:
                sleep(1)

    print('⭐ Success - your data is indexed')
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal
