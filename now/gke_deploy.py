import json
import pathlib
from os.path import expanduser as user

import cowsay

from now.deployment.deployment import cmd
from now.dialog import maybe_prompt_user
from now.log.log import yaspin_extended
from now.utils import custom_spinner, sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


def ask_projects(options):
    options = sorted(options, key=lambda x: x.lower())
    questions = [
        {
            'type': 'list',
            'name': 'project',
            'message': 'What project you want to use for kubernetes deployment?',
            'choices': options,
        }
    ]
    return maybe_prompt_user(questions, 'project')


def ask_regions(options):
    questions = [
        {
            'type': 'list',
            'name': 'region',
            'message': 'Which region to chose?',
            'choices': options,
            'filter': lambda val: val.lower(),
        }
    ]
    return maybe_prompt_user(questions, 'region')


def ask_zones(options):
    questions = [
        {
            'type': 'list',
            'name': 'zone',
            'message': 'Which zone you would like to select?',
            'choices': options,
        }
    ]
    return maybe_prompt_user(questions, 'zone')


# Google cloud authentication ->
def init_gcloud(gcloud_path):
    out, _ = cmd(f'{gcloud_path} auth list')
    if not out:
        print('Please perform gcloud authentication to deploy Flow on GKE')
        cmd(f'{gcloud_path} auth login', std_output=True)


# List the projects and present it as options to user
def get_project(gcloud_path):
    project_list = []
    output, _ = cmd(f'{gcloud_path} projects list --format=json')
    projects = output.decode('utf-8')
    projects = json.loads(projects)
    for proj in projects:
        project_list.append(proj['projectId'])

    return ask_projects(project_list)


def get_region(gcloud_path):
    regions_list = []
    output, _ = cmd(f'{gcloud_path} compute regions list --format=json')
    regions = output.decode('utf-8')
    regions = json.loads(regions)
    for region in regions:
        regions_list.append(region['name'])

    return ask_regions(regions_list)


def get_zone(region, gcloud_path):
    zones_list = []
    output, _ = cmd(f'{gcloud_path} compute zones list --format=json')
    zones = output.decode('utf-8')
    zones = json.loads(zones)
    for zone in zones:
        if region in zone['name']:
            zones_list.append(zone['name'])

    return ask_zones(zones_list)


def final_confirmation():
    questions = [
        {
            'type': 'list',
            'name': 'proceed',
            'message': 'Creating a cluster will create some costs. '
            'Are you sure you want to continue? '
            'Prices can be checked here: '
            'https://cloud.google.com/kubernetes-engine/pricing',
            'choices': [
                {'name': '💸🔥 yes', 'value': True},
                {'name': ':⛔ no', 'value': False},
            ],
        }
    ]
    proceed = maybe_prompt_user(questions, 'proceed')
    if not proceed:
        cowsay.cow('see you soon 👋')
        exit(0)


def create_gke_cluster():
    gcloud_path, _ = cmd('which gcloud')
    gcloud_path = gcloud_path.strip()
    if not gcloud_path:
        gcloud_path = user('~/.cache/jina-now/google-cloud-sdk/bin/gcloud')
    else:
        gcloud_path = gcloud_path.decode('utf-8')
    application_name = 'jina-now'
    init_gcloud(gcloud_path)
    proj = get_project(gcloud_path)
    cmd(f'{gcloud_path} config set project {proj}')
    region = get_region(gcloud_path)
    cmd(f'{gcloud_path} config set compute/region {region}')
    zone = get_zone(region, gcloud_path)
    cmd(f'{gcloud_path} config set compute/zone {zone}')
    final_confirmation()
    out, _ = cmd(f'{gcloud_path} container clusters list')
    out = out.decode('utf-8')
    if application_name in out and zone in out:
        with yaspin_extended(
            sigmap=sigmap, text='Cluster exists already', color='green'
        ) as spinner:
            spinner.ok('✅')
    else:
        with yaspin_extended(
            custom_spinner().weather, sigmap=sigmap, text="Create cluster"
        ) as spinner:
            cmd(
                f'/bin/bash {cur_dir}/scripts/gke_deploy.sh {application_name} {gcloud_path}'
            )
            spinner.ok('🌥')


if __name__ == '__main__':
    create_gke_cluster()
