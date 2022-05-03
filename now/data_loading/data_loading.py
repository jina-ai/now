import base64
import os
import uuid
from copy import deepcopy
from os.path import join as osp
from pathlib import Path
from typing import List, Optional, Union

import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from docarray import Document, DocumentArray
from yaspin import yaspin

from now.constants import (
    BASE_STORAGE_URL,
    IMAGE_MODEL_QUALITY_MAP,
    DatasetTypes,
    Modalities,
    Qualities,
)
from now.data_loading.convert_datasets_to_jpeg import to_thumbnail_jpg
from now.dialog import UserInput
from now.utils import download, sigmap

_tokenizer = _Tokenizer()


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
        if user_input.custom_dataset_type == DatasetTypes.DOCARRAY:
            print('⬇  Pull DocArray')
            da = _pull_docarray(user_input.dataset_secret)
        elif user_input.custom_dataset_type == DatasetTypes.URL:
            print('⬇  Pull DocArray')
            da = _fetch_da_from_url(user_input.dataset_url)
        elif user_input.custom_dataset_type == DatasetTypes.PATH:
            print('💿  Loading DocArray from disk')
            da = _load_from_disk(user_input.dataset_path, user_input.output_modality)

    if da is None:
        raise ValueError(
            f'Could not load DocArray. Please check your configuration: {user_input}.'
        )
    da = da.shuffle(seed=42)
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


def _load_from_disk(dataset_path: str, modality: Modalities) -> DocumentArray:
    if os.path.isfile(dataset_path):
        try:
            return DocumentArray.load_binary(dataset_path)
        except Exception as e:
            print(f'Failed to load the binary file provided under path {dataset_path}')
            exit(1)
    elif os.path.isdir(dataset_path):
        with yaspin(
            sigmap=sigmap, text="Loading and pre-processing data", color="green"
        ) as spinner:
            if modality == Modalities.IMAGE:
                return _load_images_from_folder(dataset_path)
            elif modality == Modalities.TEXT:
                return _load_texts_from_folder(dataset_path)
            elif modality == Modalities.MUSIC:
                return _load_music_from_folder(dataset_path)
            spinner.ok('🏭')
    else:
        raise ValueError(
            f'The provided dataset path {dataset_path} does not'
            f' appear to be a valid file or folder on your system.'
        )


def _load_images_from_folder(path: str) -> DocumentArray:
    def convert_fn(d):
        try:
            d.load_uri_to_image_tensor()
            return to_thumbnail_jpg(d)
        except:
            return d

    da = DocumentArray.from_files(path + '/**')
    da.apply(convert_fn)
    return DocumentArray(d for d in da if d.blob != b'')


def _load_music_from_folder(path: str):
    from pydub import AudioSegment

    def convert_fn(d: Document):
        try:
            AudioSegment.from_file(d.uri)  # checks if file is valid
            with open(d.uri, 'rb') as fh:
                d.blob = fh.read()
            return d
        except Exception as e:
            return d

    da = DocumentArray.from_files(path + '/**')
    da.apply(convert_fn)
    return DocumentArray(d for d in da if d.blob != b'')


def _load_texts_from_folder(path: str) -> DocumentArray:
    def convert_fn(d):
        try:
            d.load_uri_to_text()
            d.tags['additional_info'] = str(Path(d.uri).relative_to(path))
            return d
        except:
            return d

    def split_by_tokens(d):
        tokens = tokenize_sliding_window(d.text)
        # remove start of text and end of text tokens
        tokens = tokens[:, 1:]
        tokens[tokens == _tokenizer.encoder["<|endoftext|>"]] = 0
        return DocumentArray(
            (
                Document(
                    mime_type='text',
                    text=_tokenizer.decode(token.tolist()),
                    tags=d.tags,
                )
                for token in tokens
            )
        )

    da = DocumentArray.from_files(path + '/*.txt')
    da.apply(convert_fn)

    ret = DocumentArray()
    for d in da:
        ret += split_by_tokens(d)
    return ret


def tokenize_sliding_window(
    texts: Union[str, List[str]], context_length: int = 77, stride: int = 50
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    stride: int
        Determines the overlap between two parts of the context when splitting is needed.
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings plus number of overlaps, context_length].
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    context_length_tokens = context_length - 2
    assert (
        stride <= context_length_tokens
    ), f"stride ({stride}) is longer than number of tokens ({context_length_tokens}) for embedding"
    all_tokens = [_tokenizer.encode(text) for text in texts]

    result = []
    for tokens in all_tokens:
        if len(tokens) > context_length - 2:
            num_overlaps = (len(tokens) - context_length_tokens) // stride + 2
            for k in range(num_overlaps):
                start_idx = k * stride
                additional_zeros = (
                    0
                    if k < num_overlaps - 1
                    else context_length_tokens - len(tokens[start_idx:])
                )
                _tokens = (
                    [sot_token]
                    + tokens[start_idx : start_idx + context_length_tokens]
                    + [eot_token]
                    + [0 for _ in range(additional_zeros)]
                )
                result.append(torch.tensor(_tokens, dtype=torch.int))
        else:
            additional_zeros = context_length - 2 - len(tokens)
            _tokens = (
                [sot_token]
                + tokens
                + [eot_token]
                + +[0 for _ in range(additional_zeros)]
            )
            result.append(torch.tensor(_tokens, dtype=torch.int))

    return torch.stack(result)


def get_dataset_url(
    dataset: str, model_quality: Optional[Qualities], output_modality: Modalities
) -> str:
    data_folder = None
    if output_modality == Modalities.IMAGE:
        data_folder = 'jpeg'
    elif output_modality == Modalities.TEXT:
        data_folder = 'text'
    elif output_modality == Modalities.MUSIC:
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
