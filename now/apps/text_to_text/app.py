import os
from typing import Dict

from docarray import DocumentArray
from now_common.utils import setup_clip_music_apps

from now.apps.base.app import JinaNOWApp
from now.constants import Apps, Modalities, Qualities
from now.dataclasses import UserInput


class TextToText(JinaNOWApp):
    def __init__(self):
        super().__init__()

    @property
    def app_name(self) -> str:
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

    @property
    def required_docker_memory_in_gb(self) -> int:
        return 8

    def set_flow_yaml(self, **kwargs):
        finetuning = kwargs.get('finetuning', False)
        encode = kwargs.get('encode', False)
        if finetuning and encode:
            raise ValueError(f"Can't set finetuning={finetuning} and encode={encode}")

        flow_dir = os.path.abspath(os.path.join(__file__, '..'))

        if finetuning:
            self.flow_yaml = os.path.join(flow_dir, 'ft-flow-sbert.yml')
        elif encode:
            self.flow_yaml = os.path.join(flow_dir, 'encode-flow-sbert.yml')
        else:
            self.flow_yaml = os.path.join(flow_dir, 'flow-sbert.yml')

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        return {Qualities.MEDIUM: 768}

    def setup(
        self, dataset: DocumentArray, user_input: UserInput, kubectl_path: str
    ) -> Dict:
        quality_pretrained_model_map = {
            Qualities.MEDIUM: 'sentence-transformers/msmarco-distilbert-base-v4',
        }
        return setup_clip_music_apps(
            app_instance=self,
            user_input=user_input,
            dataset=dataset,
            encoder_uses='TransformerSentenceEncoder/v0.4',
            encoder_uses_with={
                'pretrained_model_name_or_path': quality_pretrained_model_map[
                    user_input.quality
                ][1]
            },
            indexer_uses='DocarrayIndexerV2',
            kubectl_path=kubectl_path,
        )
