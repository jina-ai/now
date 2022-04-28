import base64
import glob
import os
import uuid
from copy import deepcopy
from os.path import join as osp
from typing import Optional, Set

from docarray import Document, DocumentArray
from yaspin import yaspin

from now.constants import (
    BASE_STORAGE_URL,
    IMAGE_MODEL_QUALITY_MAP,
    DatasetType,
    Modality,
    Quality,
)
from now.data_loading.convert_datasets_to_jpeg import to_thumbnail_jpg
from now.data_loading.utils import load_mp3
from now.dialog import UserInput
from now.utils import download, sigmap

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
MUSIC_EXTENSIONS = {'.mp3'}


def load_data(user_input: UserInput) -> DocumentArray:
    """
    Based on the user input, this function will pull the configured DocArray.

    :param user_input: The configured user object. Result from the Jina Now cli dialog.
    :return: The loaded DocumentArray.
    """
    da = None

    if not user_input.is_custom_dataset:
        print('⬇  Download DocArray')
        url = get_dataset_url(
            user_input.data, user_input.quality, user_input.output_modality
        )
        da = _fetch_da_from_url(url)

    else:
        if user_input.custom_dataset_type == DatasetType.DOCARRAY:
            print('⬇  Pull DocArray')
            da = _pull_docarray(user_input.dataset_secret)
        elif user_input.custom_dataset_type == DatasetType.URL:
            print('⬇  Pull DocArray')
            da = _fetch_da_from_url(user_input.dataset_url)
        elif user_input.custom_dataset_type == DatasetType.PATH:
            print('💿  Loading DocArray from disk')
            da = _load_from_disk(user_input.dataset_path, user_input.output_modality)

    if da is None:
        raise ValueError(
            f'Could not load DocArray. Please check your configuration: {user_input}.'
        )
    da = da.shuffle(seed=42)  # TODO: why?
    da = _deep_copy_da(da)
    return da


def _fetch_da_from_url(
    url: str, downloaded_path: str = '~/.cache/jina-now'
) -> DocumentArray:
    data_dir = os.path.expanduser(downloaded_path)
    if not os.path.exists(osp(data_dir, 'data/tmp')):
        os.makedirs(osp(data_dir, 'data/tmp'))
    data_path = (
        data_dir
        + f"/data/tmp/{base64.b64encode(bytes(url, 'utf-8')).decode('utf-8')}.bin"
    )
    if not os.path.exists(data_path):
        download(url, data_path)

    with yaspin(sigmap=sigmap, text="Extracting dataset", color="green") as spinner:
        da = DocumentArray.load_binary(data_path)
        spinner.ok("📂")
    return da


def _pull_docarray(dataset_secret: str):
    try:
        return DocumentArray.pull(token=dataset_secret, show_progress=True)
    except Exception:
        print(
            '💔 oh no, the secret of your docarray is wrong, or it was deleted after 14 days'
        )
        exit(1)


def _folder_contains_only(path: str, extensions: Set) -> bool:
    def valid_file(filename: str) -> bool:
        return filename.endswith(tuple(extensions))

    files = glob.glob(os.path.join(path, '/**'))
    map(valid_file, files)
    return all(files)


def _load_from_disk(dataset_path: str, modality: Modality) -> DocumentArray:
    if os.path.isfile(dataset_path):
        try:
            return DocumentArray.load_binary(dataset_path)
        except Exception as e:
            print(f'Failed to load the binary file provided under path {dataset_path}')
            exit(1)
    elif os.path.isdir(dataset_path):
        da = DocumentArray.from_files(dataset_path + '/**')
        convert_fn = None
        if modality == Modality.IMAGE:
            assert _folder_contains_only(dataset_path, IMAGE_EXTENSIONS)

            def convert_fn(d: Document):
                try:
                    d.load_uri_to_image_tensor()
                    return to_thumbnail_jpg(d)
                except Exception as e:
                    return d

        elif modality == Modality.MUSIC:
            assert _folder_contains_only(dataset_path, MUSIC_EXTENSIONS)

            def convert_fn(d: Document):
                arr, sr = load_mp3(d.uri)
                d.tensor = arr
                d.tags['sr'] = sr
                return d

        if convert_fn is not None:
            with yaspin(
                sigmap=sigmap, text="Pre-processing data", color="green"
            ) as spinner:
                da.apply(convert_fn)
                if modality == Modality.IMAGE:
                    da = DocumentArray(d for d in da if d.blob != b'')
                elif modality == Modality.MUSIC:
                    da = DocumentArray(d for d in da if d.tensor is not None)
        spinner.ok('🏭')

        return da
    else:
        raise ValueError(
            f'The provided dataset path {dataset_path} does not'
            f' appear to be a valid file or folder on your system.'
        )


def get_dataset_url(
    dataset: str, model_quality: Optional[Quality], output_modality: Modality
) -> str:
    data_folder = None
    if output_modality == Modality.IMAGE:
        data_folder = 'jpeg'
    elif output_modality == Modality.TEXT:
        data_folder = 'text'
    elif output_modality == Modality.MUSIC:
        data_folder = 'music'

    if model_quality is not None:
        return f'{BASE_STORAGE_URL}/{data_folder}/{dataset}.{IMAGE_MODEL_QUALITY_MAP[model_quality][0]}.bin'
    else:
        return f'{BASE_STORAGE_URL}/{data_folder}/{dataset}.bin'


def _deep_copy_da(da: DocumentArray) -> DocumentArray:
    new_da = DocumentArray()
    for i, d in enumerate(da):
        new_doc = deepcopy(d)
        new_doc.id = str(uuid.uuid4())
        new_da.append(new_doc)
    return new_da
