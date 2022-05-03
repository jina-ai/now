"""
This suite tests that the demo datasets are available
behind their URLs.
"""

import pytest
import requests

from now.constants import AVAILABLE_DATASET, Modalities, Qualities
from now.data_loading.data_loading import get_dataset_url


@pytest.mark.parametrize(
    'modality, ds_name',
    [(m, d) for m in Modalities.as_list() for d in AVAILABLE_DATASET[m]],
)
@pytest.mark.parametrize('quality', [q for q in Qualities.as_list()])
def test_dataset_is_available(
    ds_name: str,
    modality: Modalities,
    quality: Qualities,
):
    if modality == Modalities.MUSIC:  # music has no quality config
        quality = None
    url = get_dataset_url(ds_name, quality, modality)

    assert requests.head(url).status_code == 200
