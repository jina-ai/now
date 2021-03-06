from docarray import DocumentArray

from now.apps.music_to_music.app import MusicToMusic
from now.constants import Apps
from now.finetuning.run_finetuning import finetune_now
from now.finetuning.settings import parse_finetune_settings
from now.now_dataclasses import UserInput


def test_music_access_lookup_dict():
    user_input = UserInput(app=Apps.MUSIC_TO_MUSIC, data='a')
    dataset = DocumentArray()
    pre_trained_head_map = {'a': 'b'}
    cached = finetune_now(
        user_input,
        dataset,
        parse_finetune_settings(MusicToMusic(), user_input, dataset, ()),
        pre_trained_head_map,
        '',
        encoder_uses='',
        encoder_uses_with={},
    )
    assert cached == 'b'
