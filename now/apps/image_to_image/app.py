import os
from typing import Dict, List

from docarray import DocumentArray
from now_common import options

from now.apps.base.app import JinaNOWApp
from now.constants import (
    CLIP_USES,
    IMAGE_MODEL_QUALITY_MAP,
    DemoDatasets,
    Modalities,
    Qualities,
)
from now.dataclasses import UserInput
from now.run_backend import finetune_flow_setup


class ImageToImage(JinaNOWApp):
    def __init__(self):
        super().__init__()

    @property
    def description(self) -> str:
        return 'Image to text search'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.IMAGE

    @property
    def output_modality(self) -> Modalities:
        return Modalities.IMAGE

    def set_flow_yaml(self, finetuning: bool = False):
        now_package_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        flow_dir = os.path.join(now_package_dir, 'deployment', 'flow')
        if finetuning:
            self._flow_yaml = os.path.join(flow_dir, 'ft-flow-clip.yml')
        else:
            self._flow_yaml = os.path.join(flow_dir, 'flow-clip.yml')

    @property
    def options(self) -> List[Dict]:
        return [options.QUALITY_CLIP]

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        return {
            Qualities.MEDIUM: 512,
            Qualities.GOOD: 512,
            Qualities.EXCELLENT: 768,
        }

    def setup(self, da: DocumentArray, user_config: UserInput, kubectl_path) -> Dict:
        return finetune_flow_setup(
            self,
            da,
            user_config,
            kubectl_path,
            encoder_uses=CLIP_USES,
            artifact=IMAGE_MODEL_QUALITY_MAP[user_config.quality][1],
            finetune_datasets=(DemoDatasets.DEEP_FASHION, DemoDatasets.BIRD_SPECIES),
        )
