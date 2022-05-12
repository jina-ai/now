import subprocess
import tempfile

from jcloud.flow import CloudFlow
from jina.helper import get_or_reuse_loop


def deploy_wolf(path: str, name: str):
    loop = get_or_reuse_loop()
    return loop.run_until_complete(CloudFlow(path=path, name=name).__aenter__())


def terminate_wolf(flow_id: str):
    loop = get_or_reuse_loop()
    loop.run_until_complete(CloudFlow(flow_id=flow_id).__aexit__())


def cmd(command, std_output=False, wait=True):
    if isinstance(command, str):
        command = command.split()
    if not std_output:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    else:
        process = subprocess.Popen(command)
    if wait:
        output, error = process.communicate()
        return output, error


def which(executable: str) -> bool:
    return bool(cmd('which ' + executable)[0])


def apply_replace(f_in, replace_dict, kubectl_path=None, ns=None):
    with open(f_in, "r") as fin:
        with tempfile.NamedTemporaryFile(mode='w') as fout:
            for line in fin.readlines():
                for key, val in replace_dict.items():
                    line = line.replace('{' + key + '}', str(val))
                fout.write(line)
            fout.flush()
            cmd(f'{kubectl_path} apply -f {fout.name}')


def apply_replace_for_flow(f_in, replace_dict, ns):
    with open(f_in, "r") as fin:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as fout:
            for line in fin.readlines():
                for key, val in replace_dict.items():
                    line = line.replace('${{ ENV.' + key + ' }}', str(val))
                fout.write(line)
            fout.flush()
            return deploy_wolf(fout.name, name=ns)
