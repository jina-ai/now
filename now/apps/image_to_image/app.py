from typing import Dict, List

from docarray import DocumentArray
from now_common import options

from now.apps.base.app import JinaNOWApp
from now.constants import (
    CLIP_USES,
    IMAGE_MODEL_QUALITY_MAP,
    Apps,
    DemoDatasets,
    Modalities,
)
from now.dataclasses import UserInput
from now.run_backend import finetune_flow_setup


class ImageToImage(JinaNOWApp):
    @property
    def app(self) -> str:
        return Apps.IMAGE_TO_IMAGE

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return 'Image to image search app'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.IMAGE

    @property
    def output_modality(self) -> Modalities:
        return Modalities.IMAGE

    @property
    def options(self) -> List[Dict]:
        return [options.QUALITY_CLIP]

    def setup(self, da: DocumentArray, user_config: UserInput, kubectl_path) -> Dict:
        return finetune_flow_setup(
            self,
            da,
            user_config,
            kubectl_path,
            encoder_uses=CLIP_USES,
            encoder_uses_with={
                'pretrained_model_name_or_path': IMAGE_MODEL_QUALITY_MAP[
                    user_config.quality
                ][1]
            },
            finetune_datasets=(DemoDatasets.DEEP_FASHION, DemoDatasets.BIRD_SPECIES),
            indexer_uses='DocarrayIndexer',
        )
