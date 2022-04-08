import io
import math
from typing import TYPE_CHECKING, Callable

import torch
from docarray import Document
from jina import DocumentArray
from PIL import Image
from tqdm import tqdm


def _text_collate_fn(batch: DocumentArray):
    if TYPE_CHECKING:
        import clip
    return clip.tokenize([doc.text.lower() for doc in batch], truncate=True)


class _ImageCollateFn:
    def __init__(self, now_preprocess: Callable):
        self._now_preprocess = now_preprocess

    def __call__(self, batch: DocumentArray):
        return torch.stack(
            [
                self._now_preprocess(
                    Image.fromarray(doc.tensor)
                    if doc.tensor
                    else Image.open(io.BytesIO(doc.blob))
                )
                for doc in batch
            ],
            dim=0,
        )


def embed_docs(
    model: Callable,
    docs: DocumentArray,
    collate_fn: Callable,
    device: str = 'cpu',
    batch_size: int = 128,
):
    total = math.ceil(len(docs) / batch_size)

    with torch.no_grad():
        for batch in tqdm(docs.batch(batch_size=batch_size), total=total):
            x = collate_fn(batch)
            x = x.to(device)
            y = model(x)
            batch.embeddings = y.cpu().detach().numpy()


def to_jpg(image_docs):
    def pil2bytes(img: Image):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=75)
        return img_byte_arr.getvalue()

    def convert_to_jpeg(d: Document):
        if not d.blob:
            d.convert_image_tensor_to_blob(image_format='jpeg')
            # TODO: Why we need separate conversion if same is done above?
            # im = Image.fromarray(d.tensor)
            # d.tensor = None
            # d.blob = pil2bytes(im)
        return d

    image_docs.apply(convert_to_jpeg)


def embed_dataset(
    dataset: str,
    model: str,
    project: str,
    bucket: str,
    location: str,
    batch_size: int = 128,
    num_workers: int = 8,
):
    if TYPE_CHECKING:
        import clip
    path = f'{dataset}.bin'

    print(f'===> {dataset} - {model}')
    print(f'  Loading {dataset} dataset from {path} ...')
    docs = DocumentArray.load_binary(path)
    print(f'  Dataset size: {len(docs)}')

    print(f'  Loading {model} CLIP model ...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = clip.load(model, device)
    clip_model.eval()
    clip_model.to(device)

    print(f'  Splitting text and image docs ...')
    text_docs = DocumentArray(
        [doc for doc in docs if doc.tags['content_type'] == 'text']
    )
    image_docs = DocumentArray(
        [doc for doc in docs if doc.tags['content_type'] == 'image']
    )
    print(f'  Num text docs: {len(text_docs)}')
    print(f'  Num image docs: {len(image_docs)}')

    print(f'  Embedding text docs ...')
    embed_docs(
        clip_model.encode_text,
        text_docs,
        collate_fn=_text_collate_fn,
        device=device,
        batch_size=batch_size,
    )
    print(f'  Done!')

    print(f'  Embedding image docs ...')
    embed_docs(
        clip_model.encode_image,
        image_docs,
        collate_fn=_ImageCollateFn(preprocess),
        device=device,
        batch_size=batch_size,
    )
    print(f'  Done!')

    print(f' convert images to jpeg for smaller datasets')
    to_jpg(image_docs)

    print(f'  Saving embedded docs ...')
    docs = text_docs + image_docs
    docs = docs.shuffle(42)
    out = f'{dataset}.{model.replace("/", "")}.bin'
    docs.save_binary(out)
    print(f'  Saved embedded docs to {out} ...')

    # print(f'  Uploading dataset ...')
    # upload_to_gcloud_bucket(project, bucket, location, out)
    # print(f'  Uploaded dataset to gs://{bucket}/{location}/{out}')


def main():
    """
    Main method.
    """
    project = 'jina-simpsons-florian'
    bucket = 'jina-fashion-data'
    location = 'data/one-line/datasets'
    datasets = [
        'tll',
        'nft-monkey',
        'deepfashion',
        'nih-chest-xrays',
        'geolocation-geoguessr',
        'stanford-cars',
        'bird-species',
        'best-artworks',
    ]
    models = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    batch_size = 128

    for dataset in datasets:
        for model in models:
            embed_dataset(dataset, model, project, bucket, location, batch_size)


if __name__ == '__main__':
    main()
