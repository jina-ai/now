import os
from typing import Dict

import cowsay
from docarray import DocumentArray

from now.apps.base.app import JinaNOWApp
from now.constants import Apps, DemoDatasets, Modalities, Qualities
from now.dataclasses import UserInput
from now.deployment.deployment import which
from now.run_backend import finetune_flow_setup


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
    def app(self) -> str:
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

    def set_flow_yaml(self, **kwargs):
        flow_dir = os.path.abspath(os.path.join(__file__, '..'))
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

    def setup(self, da: DocumentArray, user_config: UserInput, kubectl_path) -> Dict:
        return finetune_flow_setup(
            self,
            da,
            user_config,
            kubectl_path,
            encoder_uses='BiModalMusicTextEncoder:fcb025de625784073c4fcf5eb6ba2d50/v0.0.11',
            encoder_uses_with={},
            finetune_datasets=(
                DemoDatasets.MUSIC_GENRES_ROCK,
                DemoDatasets.MUSIC_GENRES_MIX,
            ),
            pre_trained_head_map={
                DemoDatasets.MUSIC_GENRES_ROCK: 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/'
                'linear_head_encoder_music_2k',
                DemoDatasets.MUSIC_GENRES_MIX: 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/'
                'linear_head_encoder_music_10k',  # music-genres-mix is sampled from the 10k dataset
            },
            indexer_uses='MusicRecommendationIndexer:e0b75cc6569bd73cee76e1161a433b9d/v0.0.5',
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
