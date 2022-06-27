import os
from typing import Dict

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import (
    IMAGE_MODEL_QUALITY_MAP,
    Apps,
    Modalities,
    Qualities,
)
from now.dataclasses import UserInput
from now.run_backend import finetune_flow_setup


class TextToMesh(JinaNOWApp):
    def __init__(self):
        super().__init__()

    @property
    def app(self) -> str:
        return Apps.TEXT_TO_MESH

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return 'Text to mesh search app'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.TEXT

    @property
    def output_modality(self) -> Modalities:
        return Modalities.MESH

    def set_flow_yaml(self, **kwargs):
        flow_dir = os.path.abspath(os.path.join(__file__, '..'))
        self.flow_yaml = os.path.join(flow_dir, 'flow-mesh.yml')
        # from jina import Flow
        # os.environ['TRAVERSAL_PATHS'] = '@c'
        # f = Flow().load_config(self.flow_yaml)
        # f.save_config('test.yml')
        #
        # os.environ['TRAVERSAL_PATHS'] = '@r'
        # f = Flow().load_config(self.flow_yaml)
        # f.save_config('test2.yml')

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
            encoder_uses='CLIPEncoder/v0.3.0',
            encoder_uses_with={
                'pretrained_model_name_or_path': IMAGE_MODEL_QUALITY_MAP[
                    user_config.quality
                ][1],
                # 'traversal_paths': '@c',
            },
            # finetune_datasets=(DemoDatasets.DEEP_FASHION, DemoDatasets.BIRD_SPECIES),
            indexer_uses='AnnLiteIndexer/0.3.0',
        )
