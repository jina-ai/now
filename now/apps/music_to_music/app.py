import os
from typing import Dict

import cowsay
from docarray import DocumentArray
from now_common.utils import setup_clip_music_apps

from now.apps.base.app import JinaNOWApp
from now.constants import Apps, DemoDatasets, Modalities, Qualities
from now.dataclasses import UserInput
from now.deployment.deployment import which


class MusicToMusic(JinaNOWApp):
    """
    Music2Music search.

    Pre-trained head weights are stored here
    https://console.cloud.google.com/storage/browser/jina-fashion-data/model/music?project=jina-simpsons-florian
    To re-built, go in now/hub/head_encoder with place the model weights in this folder
    and run "jina hub push --private . -t linear_head_encoder_music_2k"
    """

    def __init__(self):
        super().__init__()

    @property
    def app_name(self) -> str:
        return Apps.MUSIC_TO_MUSIC

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return 'Music to music search app'

    @property
    def input_modality(self) -> Modalities:
        return Modalities.MUSIC

    @property
    def output_modality(self) -> Modalities:
        return Modalities.MUSIC

    @property
    def required_docker_memory_in_gb(self) -> int:
        return 10

    def set_flow_yaml(self, **kwargs):
        finetuning = kwargs.get('finetuning', False)
        encode = kwargs.get('encode', False)
        if finetuning and encode:
            raise ValueError(f"Can't set finetuning={finetuning} and encode={encode}")

        flow_dir = os.path.abspath(os.path.join(__file__, '..'))

        if encode:
            self.flow_yaml = os.path.join(flow_dir, 'encode-flow-music.yml')
        else:
            self.flow_yaml = os.path.join(flow_dir, 'ft-flow-music.yml')

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        return {
            Qualities.MEDIUM: 512,
        }

    def check_requirements(self) -> bool:
        if not ffmpeg_is_installed():
            _handle_ffmpeg_install_required()
            return False
        return True

    def setup(
        self, dataset: DocumentArray, user_input: UserInput, kubectl_path
    ) -> Dict:
        return setup_clip_music_apps(
            app_instance=self,
            user_input=user_input,
            dataset=dataset,
            encoder_uses='BiModalMusicTextEncoderV2',
            encoder_uses_with={},
            pre_trained_head_map={
                DemoDatasets.MUSIC_GENRES_ROCK: 'FinetunedLinearHeadEncoderMusicRock',
                DemoDatasets.MUSIC_GENRES_MIX: 'FineTunedLinearHeadEncoderMusicMix',
            },
            indexer_uses='MusicRecommendationIndexerV2',
            finetune_datasets=(
                DemoDatasets.MUSIC_GENRES_MIX,
                DemoDatasets.MUSIC_GENRES_ROCK,
            ),
            kubectl_path=kubectl_path,
        )


def ffmpeg_is_installed():
    return which("ffmpeg")


def _handle_ffmpeg_install_required():
    bc_red = '\033[91m'
    bc_end = '\033[0m'
    print()
    print(
        f"{bc_red}To use the audio modality you need the ffmpeg audio processing"
        f" library installed on your system.{bc_end}"
    )
    print(
        f"{bc_red}For MacOS please run 'brew install ffmpeg' and on"
        f" Linux 'apt-get install ffmpeg libavcodec-extra'.{bc_end}"
    )
    print(
        f"{bc_red}After the installation, restart Jina Now and have fun with music search 🎸!{bc_end}"
    )
    cowsay.cow('see you soon 👋')
    exit(1)
