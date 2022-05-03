import base64
import os
import random
import uuid
from copy import deepcopy
from os.path import join as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from docarray import Document, DocumentArray
from yaspin import yaspin

from now.data_loading.convert_datasets_to_jpeg import to_thumbnail_jpg
from now.dialog import QUALITY_MAP, Modalities
from now.utils import download, sigmap

_tokenizer = _Tokenizer()


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


def remove_duplicates(da: DocumentArray):
    """Some da"""
    # known_set = set()
    # unique_dataset = DocumentArray()
    # for i, d in enumerate(da):
    #     d.id = str(uuid.uuid4())
    #     l = d.tags['finetuner_label']
    #     if d.text and l in known_set:
    #         continue
    #     unique_dataset.append(d)
    #     known_set.add(l)
    # return unique_dataset
    # da_text = DocumentArray(d for d in da if d.text)
    # da_img = DocumentArray(d for d in da if not d.text)
    # da_text.embeddings = da_text.embeddings - da_text.embeddings.mean(0)
    # da_img.embeddings = da_img.embeddings - da_img.embeddings.mean(0)

    new_da = DocumentArray()
    for i, d in enumerate(da):
        new_doc = deepcopy(d)
        new_doc.id = str(uuid.uuid4())
        new_da.append(new_doc)
    return new_da


def load_data(
    output_modality: str,
    data: str,
    model_quality: str,
    is_custom: bool,
    custom_type: str,
    secret: Optional[str],
    url: Optional[str],
    path: Optional[str],
) -> Tuple[DocumentArray, str]:

    data_folder = None
    if not is_custom:
        print('⬇  Download data')
        if output_modality == Modalities.IMAGE:
            data_folder = 'jpeg'
        elif output_modality == Modalities.TEXT:
            data_folder = 'text'
        elif output_modality == Modalities.MUSIC:
            data_folder = 'music'
        url = (
            'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets/'
            f'{data_folder}/{data}.{QUALITY_MAP[model_quality][0]}.bin'
        )
        da = _fetch_da_from_url(url)
        ds_type = 'demo'

    else:
        if custom_type == 'docarray':
            print('⬇  pull docarray')
            try:
                da = DocumentArray.pull(token=secret, show_progress=True)
                ds_type = 'docarray_pull'
            except Exception:
                print(
                    '💔 oh no, the secret of your docarray is wrong, or it was deleted after 14 days'
                )
                exit(1)
        elif custom_type == 'url':
            print('⬇  Download data')
            da = _fetch_da_from_url(url)
            ds_type = 'url'
        else:
            if os.path.isfile(path):
                try:
                    da = DocumentArray.load_binary(path)
                    ds_type = 'local_da'
                except Exception as e:
                    print('Failed to load the binary file provided')
                    exit(1)
            else:
                with yaspin(
                    sigmap=sigmap, text="Loading and pre-processing data", color="green"
                ) as spinner:
                    if output_modality == Modalities.IMAGE:
                        da = _load_images_from_folder(path)
                    elif output_modality == Modalities.TEXT:
                        da = _load_texts_from_folder(path)
                    spinner.ok('🏭')
                ds_type = 'local_folder'

                # for d in da:
                #     d.tags['finetuner_label'] = os.path.dirname(d.uri).split('/')[-1]

    da = da.shuffle(seed=42)
    da = remove_duplicates(da)
    return da, ds_type


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


# def load_all_data(dataset):
#     for k, v in dataset.items():
#         if v is not None:
#             dataset[k] = load_data(v)


def fill_missing(ds, train_val_split_ratio, num_default_val_queries, is_debug):
    # ds['index'] = deepcopy(DocumentArray(d for d in ds['index'] if d.tensor is not None))
    if ds['train'] is None:
        ds['train'] = ds['index']
    if ds['val'] is None:
        # TODO makes split based on classes rather than instances
        split_index = max(
            int(len(ds['train']) * train_val_split_ratio),
            len(ds['train']) - 5000,
        )
        train = ds['train']
        ds['train'], ds['val'] = train[:split_index], train[split_index:]

    if ds['val_index'] is None:
        ds['val_index'] = deepcopy(ds['val'])
    if ds['val_query'] is None:
        if is_debug:
            num_queries = 10
        else:
            num_queries = 100

        ds['val_query'] = DocumentArray(
            [deepcopy(doc) for doc in random.sample(ds['val_index'], num_queries)]
        )
        # for d in ds['val_query']:
        #     ds['val_index'].remove(d)

    if ds['val_index_image'] is None:
        ds['val_index_image'] = deepcopy(
            # DocumentArray(d for d in ds['val'] if d.blob is not None)
            DocumentArray(d for d in ds['val'] if d.blob != b'')
        )
    if ds['val_query_image'] is None:
        ds['val_query_image'] = DocumentArray(
            [
                deepcopy(doc)
                for doc in random.sample(ds['val_index_image'], num_default_val_queries)
            ]
        )
