import os
from typing import Dict

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import Modalities, Qualities
from now.dataclasses import UserInput
from now.run_backend import finetune_flow_setup


class TextToText(JinaNOWApp):
    def __init__(self):
        now_package_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        flow_dir = os.path.join(now_package_dir, 'deployment', 'flow')
        self._flow_yaml = os.path.join(flow_dir, 'flow-clip.yml')

    @property
    def description(self) -> str:
        return 'Text to text search'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.TEXT

    @property
    def output_modality(self) -> Modalities:
        return Modalities.TEXT

    @JinaNOWApp.flow_yaml.setter
    def flow_yaml(self, finetuning: bool):
        now_package_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        flow_dir = os.path.join(now_package_dir, 'deployment', 'flow')
        self._flow_yaml = os.path.join(flow_dir, 'flow-text.yml')
        # if finetuning:
        #     self._flow_yaml = os.path.join(flow_dir, 'ft-flow-clip.yml')
        # else:
        #     self._flow_yaml = os.path.join(flow_dir, 'flow-clip.yml')

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        return {
            Qualities.MEDIUM: 384,
        }

    def setup(
        self, da: DocumentArray, user_config: UserInput, kubectl_path: str
    ) -> Dict:
        return finetune_flow_setup(
            self,
            da,
            user_config,
            kubectl_path,
            encoder_uses='TransformerTorchEncoder/v0.4',
            artifact='sentence-transformers/all-MiniLM-L6-v2',
        )
