import os.path
import pathlib
import subprocess
from datetime import datetime

from now.log import yaspin_extended
from now.utils import copytree, sigmap

cur_dir = pathlib.Path(__file__).parent.resolve()


def push_to_hub(tmpdir):
    """
    We need the trained model as hub executor and pushed into the docker registry of hubble.
    Otherwise, there is no possibility to run the executor on Kubernetes.
    In the past, we were saving the trained model in GCP. But the user would not have access to do so.

    Idea 1 create an executor and push the executor+model to hub
    or
    Idea 2 Misuse the Docarray.push() function. Store the model as tensor
    I would prefer Idea 1 since remote Docarrays vanish, and the password has to be part of the k8s config,...
    """
    name = f'linear_head_encoder_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    secret = '93ea59dbd1ee3fe0bdc44252c6e86a87'
    class_name = 'FineTunedLinearHeadEncoder'
    src_path = os.path.join(cur_dir, 'head_encoder')
    dst_path = os.path.join(tmpdir, 'now/hub/head_encoder')
    copytree(src_path, dst_path)
    bashCommand = f"jina hub push --private {dst_path} -t {name} --force-update {class_name} --secret {secret}"
    with yaspin_extended(
        sigmap=sigmap, text="Push fine-tuned model to Jina Hub", color="green"
    ) as spinner:
        with open(os.path.join(tmpdir, "NUL"), "w") as fh:
            if 'NOW_CI_RUN' in os.environ:
                print('bashCommand: ', bashCommand)
            process = subprocess.Popen(bashCommand.split(), stdout=fh)
        output, error = process.communicate()
        spinner.ok('⏫ ')
    return f'FineTunedLinearHeadEncoder:{secret}/{name}'
