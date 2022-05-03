import csv
import json
import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field
from random import shuffle
from typing import Any, Dict, Optional

import pandas as pd
from jina import Document, DocumentArray
from tqdm import tqdm

from now.data_loading.utils import upload_to_gcloud_bucket

IMAGE_SHAPE = (224, 224)


@dataclass
class _DataPoint:
    # id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    content_type: str = 'image'
    label: str = ''
    split: str = 'none'
    tags: Dict[str, Any] = field(default_factory=lambda: {})


def _build_doc(datapoint: _DataPoint) -> Document:
    # doc = Document(id=datapoint.id)
    doc = Document()
    if datapoint.content_type == 'image':
        doc.uri = datapoint.image_path
        doc.load_uri_to_image_tensor()
        doc.set_image_tensor_shape(IMAGE_SHAPE)
    else:
        doc.text = datapoint.text
    doc.tags = {'finetuner_label': datapoint.label, 'split': datapoint.split}
    doc.tags.update(datapoint.tags)
    doc.tags.update({'content_type': datapoint.content_type})
    return doc


def _build_deepfashion(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the deepfashion dataset.
    Download the raw dataset from
    https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.jpg'
    imagedir = os.path.join(root, 'Img')
    fsplit = os.path.join(root, 'Eval', 'list_eval_partition.txt')
    fcolors = os.path.join(root, 'Anno', 'attributes', 'list_color_cloth.txt')

    # read list_eval_partition.txt
    img2split = {}
    with open(fsplit, 'r') as f:
        for line in f.read().splitlines()[2:]:
            img, _, split, _ = re.split(r' +', line)
            img2split[img] = split

    # read list_color_cloth.txt
    img2color = {}
    with open(fcolors, 'r') as f:
        for line in f.read().splitlines()[2:]:
            img, color, *_ = re.split(r'  +', line)
            img2color[img] = color

    # add image docs
    data = []
    for rootdir, _, fnames in os.walk(imagedir):
        labels = []
        productid = os.path.relpath(rootdir, imagedir)
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(rootdir, fname)
                imgid = os.path.relpath(path, imagedir)
                split = img2split[imgid]
                color = img2color[imgid]
                label = productid + '/' + color
                labels.append(label)
                data.append(
                    _DataPoint(
                        id=imgid,
                        image_path=path,
                        label=label,
                        split=split,
                        tags={'color': color},
                    )
                )

        # add text doc
        if len(labels) > 0:
            for label in set(labels):
                _, gender, category, _, color = label.split('/')
                text_elements = [category, gender, color]
                shuffle(text_elements)
                text = (
                    f'{" ".join(text_elements)}'.lower()
                    .replace('-', ' ')
                    .replace('_', ' ')
                )
                data.append(
                    _DataPoint(
                        id=rootdir,
                        text=text,
                        content_type='text',
                        label=label,
                        tags={'color': color},
                    )
                )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_nih_chest_xrays(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the NIH chest xrays dataset.
    Download the raw dataset from
    https://www.kaggle.com/nih-chest-xrays/data
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.png'
    flabels = 'Data_Entry_2017.csv'
    ftrain = 'train_val_list.txt'
    ftest = 'test_list.txt'

    # read Data_Entry_2017.csv
    # labels - fname: (finding, patient id)
    with open(os.path.join(root, flabels), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        labels = {row[0]: (row[1], row[3]) for row in reader}

    # read train_val_list.txt
    with open(os.path.join(root, ftrain), 'r') as f:
        train_list = f.read().splitlines()

    # read test_list.txt
    with open(os.path.join(root, ftest), 'r') as f:
        test_list = f.read().splitlines()

    # add image docs
    data = []
    for rootdir, _, fnames in os.walk(root):
        for fname in fnames:
            if fname.endswith(extension):

                path = os.path.join(rootdir, fname)
                label = labels.get(fname)[0]  # or labels[1]
                if fname in train_list:
                    split = 'train'
                elif fname in test_list:
                    split = 'test'
                else:
                    raise ValueError(
                        f'Doc with fname: {fname} not in train or test splits'
                    )
                data.append(
                    _DataPoint(id=fname, image_path=path, label=label, split=split)
                )

    # add text docs
    labelnames = {label for _, (label, __) in labels.items()}
    for label in labelnames:
        data.append(
            _DataPoint(
                id=label,
                text=label.lower()
                .replace('|', ' ')
                .replace('_', ' ')
                .replace('-', ' '),
                content_type='text',
                label=label,
            )
        )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_geolocation_geoguessr(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the geolocation-geoguessr dataset.
    Download the raw dataset from
    https://www.kaggle.com/ubitquitin/geolocation-geoguessr-images-50k
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.jpg'

    # add image docs
    data = []
    for rootdir, _, fnames in os.walk(root):
        label = os.path.relpath(rootdir, root)
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(rootdir, fname)
                data.append(_DataPoint(id=fname, image_path=path, label=label))

        # add text doc
        if len(fnames) > 0:
            data.append(
                _DataPoint(
                    id=label, text=label.lower(), content_type='text', label=label
                )
            )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_stanford_cars(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the stanford cars dataset.
    Download the raw dataset from
    https://www.kaggle.com/jessicali9530/stanford-cars-dataset
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.jpg'
    train_data = os.path.join(root, 'car_data', 'train')
    test_data = os.path.join(root, 'car_data', 'test')

    # add image docs
    data = []
    labels = []
    for split, root in [('train', train_data), ('test', test_data)]:
        for rootdir, _, fnames in os.walk(root):
            if len(fnames) > 0:
                label = os.path.relpath(rootdir, root)
                labels.append(label)
                for fname in fnames:
                    if fname.endswith(extension) and 'cropped' not in fname:
                        path = os.path.join(rootdir, fname)
                        data.append(
                            _DataPoint(
                                id=fname, image_path=path, label=label, split=split
                            )
                        )

    # add text docs
    labels = set(labels)
    for label in labels:
        data.append(
            _DataPoint(id=label, text=label.lower(), content_type='text', label=label)
        )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_bird_species(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the bird species dataset.
    Download the raw dataset from
    https://www.kaggle.com/veeralakrishna/200-bird-species-with-11788-images
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.jpg'
    root = os.path.join(root, 'CUB_200_2011', 'CUB_200_2011')
    fimages = os.path.join(root, 'images.txt')
    fclasses = os.path.join(root, 'classes.txt')
    flabels = os.path.join(root, 'image_class_labels.txt')
    fsplit = os.path.join(root, 'train_test_split.txt')
    contentdir = os.path.join(root, 'images')

    # read images.txt
    image2id = {}
    with open(fimages, 'r') as f:
        for line in f.read().splitlines():
            iid, fname = line.split()
            iid = int(iid)
            image2id[fname] = iid

    # read classes.txt
    id2class = {}
    with open(fclasses, 'r') as f:
        for line in f.read().splitlines():
            iid, classname = line.split()
            iid = int(iid)
            id2class[iid] = classname

    # read image_class_labels.txt
    imageid2classid = {}
    with open(flabels, 'r') as f:
        for line in f.read().splitlines():
            iid, cid = line.split()
            iid, cid = int(iid), int(cid)
            imageid2classid[iid] = cid

    # read train_test_split.txt
    imageid2split = {}
    with open(fsplit, 'r') as f:
        for line in f.read().splitlines():
            iid, split = line.split()
            iid, split = int(iid), int(split)
            imageid2split[iid] = split

    # add image docs
    data = []
    for rootdir, _, fnames in os.walk(contentdir):
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(rootdir, fname)
                image = os.path.relpath(path, contentdir)
                iid = image2id[image]
                cid = imageid2classid[iid]
                label = id2class[cid]
                split = imageid2split[iid]
                split = 'train' if split else 'test'
                data.append(
                    _DataPoint(id=fname, image_path=path, label=label, split=split)
                )

    # add text docs
    labels = {label for _, label in id2class.items()}
    for label in labels:
        data.append(
            _DataPoint(
                id=label,
                text=label[4:].lower().replace('_', ' '),
                content_type='text',
                label=label,
            )
        )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_best_artworks(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the best artworks dataset.
    Download the raw dataset from
    https://www.kaggle.com/ikarus777/best-artworks-of-all-time
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    extension = '.jpg'
    fartists = os.path.join(root, 'artists.csv')
    contentdir = os.path.join(root, 'images', 'images')

    # read artists.csv
    with open(fartists, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        label2genre = {row[1]: row[3] for row in reader}

    # add image docs
    data = []
    for rootdir, _, fnames in os.walk(contentdir):
        label = os.path.relpath(rootdir, contentdir).replace('_', ' ')
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(rootdir, fname)
                data.append(_DataPoint(id=fname, image_path=path, label=label))
        if len(fnames) > 0:
            if label == 'Albrecht Dürer':
                genre = 'Northern Renaissance'
            else:
                genre = label2genre[label]
            text = genre.lower().replace(',', ' ').replace('"', '')
            data.append(
                _DataPoint(id=genre, text=text, label=label, content_type='text')
            )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def create_file_to_text_map(dict_list):
    file_to_text = {}
    for d in dict_list:
        meta = d['metadata']
        file = meta['image'].split('//')[-1]
        attributes = meta['attributes']
        values = [d['value'] for d in attributes]
        shuffle(values)
        text = ' '.join(values)
        file_to_text[file] = text.lower()
    return file_to_text


def _build_nft(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the nft dataset.
    Download the raw dataset from
    https://github.com/skogard/apebase
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """
    f_labels = os.path.join(root, 'db')
    contentdir = os.path.join(root, 'ipfs')

    # read artists.csv
    with open(f_labels, 'r') as f:
        lines = f.readlines()
    dict_list = [json.loads(line) for line in lines]
    file_to_text = create_file_to_text_map(dict_list)

    data = []
    for file, text in file_to_text.items():
        data.append(_DataPoint(id=file, image_path=f'{contentdir}/{file}', label=file))
        data.append(
            _DataPoint(
                id=file + '_text',
                text=file_to_text[file],
                label=file,
                content_type='text',
            )
        )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_tll(root: str, num_workers: int = 8) -> DocumentArray:
    """
    Build the tll dataset.
    Download the raw dataset from
    https://sites.google.com/view/totally-looks-like-dataset
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :return: DocumentArray
    """

    def transform(d: Document):
        d.load_uri_to_blob()
        d.tags['content_type'] = 'image'
        return d

    da = DocumentArray.from_files(root + '/**')
    da.apply(lambda d: transform(d))
    return da


def _build_lyrics(
    root: str, num_workers: int = 8, genre: str = '', max_size: int = 0
) -> DocumentArray:
    """
    Builds lyrics dataset of given size and genre if specified, else the entire dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param genre: if genre isn't empty string this will only select subset of artist with this genre
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    artists_path = os.path.join(root, 'artists-data.csv')
    lyrics_path = os.path.join(root, 'lyrics-data.csv')

    artists_df = pd.read_csv(artists_path).dropna()
    lyrics = pd.read_csv(lyrics_path).dropna()

    # select English lyrics with <= 100 sentences
    lyrics = lyrics.query("language == 'en'")
    lyrics['num_sentences'] = lyrics.apply(
        lambda x: len(x['Lyric'].split('\n')), axis=1
    )
    lyrics = lyrics.query('num_sentences <= 100')

    lyrics = pd.merge(lyrics, artists_df, left_on='ALink', right_on='Link')

    lyrics = lyrics[lyrics['Genres'].str.contains(genre)]

    if 0 < max_size:
        lyrics = lyrics.sample(frac=1)

    # create sentences from lyrics
    data, all_sentences = [], []
    for idx, row in tqdm(lyrics.iterrows()):
        if 0 < max_size <= len(data):
            break
        row = row.to_dict()
        _sentences = row.pop('Lyric').split('\n')
        # filter empty, duplicate and one-word sentences and the ones containing special characters in beginning and end
        _sentences = set(
            filter(
                lambda s: len(s) > 0
                and not re.fullmatch(r"\W+[\s\w]*\W+", s)
                and not re.fullmatch(r"\W", s)
                and not re.fullmatch(r"\w+", s)
                and not re.fullmatch(r"\w+[.]+", s)
                and s not in all_sentences,
                _sentences,
            )
        )
        for _sentence in _sentences:
            if 0 < max_size <= len(data):
                break
            all_sentences.append(_sentence)
            if re.fullmatch(r".*\w", _sentence):
                _sentence += "."
            data.append(
                _DataPoint(
                    text=_sentence,
                    content_type='text',
                    tags={
                        # 'artist': row['Artist'],
                        # 'artist_genres': row['Genres'],
                        # 'song': row['SName'],
                        'additional_info': [row['SName'], row['Artist']],
                    },
                )
            )

    # build docs
    with mp.Pool(processes=num_workers) as pool:
        docs = list(tqdm(pool.imap(_build_doc, data)))

    return DocumentArray(docs)


def _build_rock_lyrics(
    root: str, num_workers: int = 8, max_size: int = 200000
) -> DocumentArray:
    """
    Builds the rock lyrics dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    return _build_lyrics(
        genre='Rock',
        root=root.replace('rock-lyrics', 'lyrics'),
        num_workers=num_workers,
        max_size=max_size,
    )


def _build_pop_lyrics(
    root: str, num_workers: int = 8, max_size: int = 200000
) -> DocumentArray:
    """
    Builds the pop lyrics dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    return _build_lyrics(
        genre='Pop',
        root=root.replace('pop-lyrics', 'lyrics'),
        num_workers=num_workers,
        max_size=max_size,
    )


def _build_rap_lyrics(
    root: str, num_workers: int = 8, max_size: int = 200000
) -> DocumentArray:
    """
    Builds the rap lyrics dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    return _build_lyrics(
        genre='Rap',
        root=root.replace('rap-lyrics', 'lyrics'),
        num_workers=num_workers,
        max_size=max_size,
    )


def _build_indie_lyrics(
    root: str, num_workers: int = 8, max_size: int = 200000
) -> DocumentArray:
    """
    Builds the indie lyrics dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    return _build_lyrics(
        genre='Indie',
        root=root.replace('indie-lyrics', 'lyrics'),
        num_workers=num_workers,
        max_size=max_size,
    )


def _build_metal_lyrics(
    root: str, num_workers: int = 8, max_size: int = 200000
) -> DocumentArray:
    """
    Builds the indie lyrics dataset. Download the CSV files from:
    https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
    :param root: the dataset root folder.
    :param num_workers: the number of parallel workers to use.
    :param max_size: used to randomly subsample from dataset if greater than 0
    :return: DocumentArray
    """
    return _build_lyrics(
        genre='Metal',
        root=root.replace('metal-lyrics', 'lyrics'),
        num_workers=num_workers,
        max_size=max_size,
    )


def process_dataset(
    datadir: str,
    name: str,
    project: str,
    bucket: str,
    location: str,
    sample_k: bool = True,
    k: int = 10,
) -> None:
    """
    Build, save and upload a dataset.
    """
    root = f'{datadir}/{name}'
    out = f'{name}.bin'
    out_img10 = f'{name}.img{k}.bin'
    out_txt10 = f'{name}.txt{k}.bin'

    print(f'===> {name}')
    print(f'  Building {name} from {root} ...')
    docs = globals()[f'_build_{name.replace("-", "_")}'](root)
    docs = docs.shuffle(42)
    image_docs = DocumentArray(
        [doc for doc in docs if doc.tags['content_type'] == 'image']
    )
    text_docs = DocumentArray(
        [doc for doc in docs if doc.tags['content_type'] == 'text']
    )
    print(f'  Dataset size: {len(docs)}')
    print(f'  Num image docs: {len(image_docs)}')
    print(f'  Num text docs: {len(text_docs)}')

    if sample_k:
        print(f'  Sampling {k} image and {k} text docs ...')
        image_docs = image_docs[:k]
        text_docs = text_docs[:k]

    print('  Saving datasets ...')
    docs.save_binary(out)
    print(f'  Saved dataset to {out}')
    if sample_k:
        if len(image_docs) > 0:
            image_docs.save_binary(out_img10)
            print(f'  Saved dataset to {out_img10}')
        if len(text_docs) > 0:
            text_docs.save_binary(out_txt10)
            print(f'  Saved dataset to {out_txt10}')

    print('  Uploading datasets ...')
    upload_to_gcloud_bucket(project, bucket, location, out)
    print(f'  Uploaded dataset to gs://{bucket}/{location}/{out}')
    if sample_k:
        if len(image_docs) > 0:
            upload_to_gcloud_bucket(project, bucket, location, out_img10)
            print(f'  Uploaded dataset to gs://{bucket}/{location}/{out_img10}')
        if len(text_docs) > 0:
            upload_to_gcloud_bucket(project, bucket, location, out_txt10)
            print(f'  Uploaded dataset to gs://{bucket}/{location}/{out_txt10}')


def main():
    """
    Main method.
    """
    localdir = 'data'
    project = 'jina-simpsons-florian'
    bucket = 'jina-fashion-data'
    location = 'data/one-line/datasets/jpeg'
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
    for name in datasets:
        process_dataset(localdir, name, project, bucket, location)
    location = 'data/one-line/datasets/text'
    datasets = [
        'rock-lyrics',
        'pop-lyrics',
        'rap-lyrics',
        'indie-lyrics',
        'metal-lyrics',
        'lyrics',
    ]
    for name in datasets:
        process_dataset(localdir, name, project, bucket, location)


if __name__ == '__main__':
    main()
