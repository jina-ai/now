import pytest
from docarray import DocumentArray
from pytest_mock import MockerFixture

from now.constants import Modalities
from now.dialog import UserInput
from now.finetuning.run_finetuning import finetune_now
from now.finetuning.settings import parse_finetune_settings


@pytest.fixture()
def fake_dict():
    class FakeDict:
        def __init__(self):
            self.hasBeenCalled = False

        def __getitem__(self, item):
            self.hasBeenCalled = True

    return FakeDict()


def test_music_access_lookup_dict(mocker: MockerFixture, fake_dict):
    mocker.patch(
        'now.finetuning.run_finetuning.PRE_TRAINED_LINEAR_HEADS_MUSIC', fake_dict
    )
    user_input = UserInput(output_modality=Modalities.MUSIC)
    dataset = DocumentArray()

    finetune_now(user_input, dataset, parse_finetune_settings(user_input, dataset), '')

    assert fake_dict.hasBeenCalled
