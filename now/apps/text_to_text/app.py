import os
from typing import Dict

from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import Apps, Modalities, Qualities
from now.dataclasses import UserInput
from now.run_backend import finetune_flow_setup


class TextToText(JinaNOWApp):
    def __init__(self):
        super().__init__()

    @property
    def app(self) -> str:
        return Apps.TEXT_TO_TEXT

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return 'Text to text search app'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.TEXT

    @property
    def output_modality(self) -> Modalities:
        return Modalities.TEXT

    def set_flow_yaml(self, **kwargs):
        flow_dir = os.path.abspath(os.path.join(__file__, '..'))
        self.flow_yaml = os.path.join(flow_dir, 'flow-sbert.yml')

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        return {Qualities.MEDIUM: 768}

    def setup(
        self, da: DocumentArray, user_config: UserInput, kubectl_path: str
    ) -> Dict:
        quality_pretrained_model_map = {
            Qualities.MEDIUM: 'sentence-transformers/msmarco-distilbert-base-v4',
        }
        return finetune_flow_setup(
            self,
            da,
            user_config,
            kubectl_path,
            encoder_uses='TransformerSentenceEncoder/v0.4',
            encoder_uses_with={
                'pretrained_model_name_or_path': quality_pretrained_model_map[
                    user_config.quality
                ]
            },
            indexer_uses='DocarrayIndexerV2',
        )
