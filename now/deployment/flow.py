import json
import os.path
import pathlib
import tempfile
from os.path import expanduser as user
from time import sleep

from jina import Flow
from jina.clients import Client
from kubernetes import client as k8s_client
from kubernetes import config
from yaspin.spinners import Spinners

from now.cloud_manager import is_local_cluster
from now.constants import JC_SECRET
from now.deployment.deployment import apply_replace, cmd, deploy_wolf
from now.log import time_profiler, yaspin_extended
from now.utils import sigmap, write_env_file

cur_dir = pathlib.Path(__file__).parent.resolve()


def batch(data_list, n=1):
    l = len(data_list)
    for ndx in range(0, l, n):
        yield data_list[ndx : min(ndx + n, l)]


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


def wait_for_all_pods_in_ns(f, ns, max_wait=1800):
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
        if len(not_ready) == 0 and f.num_deployments == len(pods):
            return
        sleep(1)


def deploy_k8s(f, ns, tmpdir, kubectl_path):
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
        wait_for_all_pods_in_ns(f, ns)
        spinner.ok("🚀")
    # work around - first request hangs
    sleep(3)
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


@time_profiler
def deploy_flow(
    deployment_type: str,
    flow_yaml: str,
    ns: str,
    env_dict: str,
    kubectl_path: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = os.path.join(tmpdir, 'dot.env')
        write_env_file(env_file, env_dict)

        if deployment_type == 'remote':
            flow = deploy_wolf(path=flow_yaml, env_file=env_file, name=ns)
            host = flow.gateway
            client = Client(host=host)

            # Dump the flow ID and gateway to keep track
            with open(user(JC_SECRET), 'w') as fp:
                json.dump({'flow_id': flow.flow_id, 'gateway': host}, fp)

            # host & port
            gateway_host = 'remote'
            gateway_port = None
            gateway_host_internal = host
            gateway_port_internal = None  # Since host contains protocol
        else:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            f = Flow.load_config(flow_yaml)
            (
                gateway_host,
                gateway_port,
                gateway_host_internal,
                gateway_port_internal,
            ) = deploy_k8s(
                f,
                ns,
                tmpdir,
                kubectl_path=kubectl_path,
            )
            client = Client(host=gateway_host, port=gateway_port)

        if os.path.exists(env_file):
            os.remove(env_file)

    return (
        client,
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    )
