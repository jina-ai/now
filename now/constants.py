from typing import List


class Modalities:
    IMAGE = 'image'
    MUSIC = 'music'
    TEXT = 'text'

    @classmethod
    def as_list(cls) -> List[str]:
        return [cls.IMAGE, cls.MUSIC, cls.TEXT]


class DatasetTypes:
    DEMO = 'demo'
    PATH = 'path'
    URL = 'url'
    DOCARRAY = 'docarray'

    @classmethod
    def as_list(cls) -> List[str]:
        return [cls.DEMO, cls.PATH, cls.URL, cls.DOCARRAY]


class Qualities:
    MEDIUM = 'medium'
    GOOD = 'good'
    EXCELLENT = 'excellent'

    @classmethod
    def as_list(cls) -> List[str]:
        return [cls.MEDIUM, cls.GOOD, cls.EXCELLENT]


BASE_STORAGE_URL = (
    'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets'
)

IMAGE_MODEL_QUALITY_MAP = {
    Qualities.MEDIUM: ('ViT-B32', 'openai/clip-vit-base-patch32'),
    Qualities.GOOD: ('ViT-B16', 'openai/clip-vit-base-patch16'),
    Qualities.EXCELLENT: ('ViT-L14', 'openai/clip-vit-large-patch14'),
}


AVAILABLE_DATASET = {
    Modalities.IMAGE: [
        'best-artworks',
        'nft-monkey',
        'tll',
        'bird-species',
        'stanford-cars',
        'deepfashion',
        'nih-chest-xrays',
        'geolocation-geoguessr',
    ],
    Modalities.MUSIC: [
        'music-genres-mid',
        'music-genres-large',
    ],
    Modalities.TEXT: [
        'rock-lyrics',
        'pop-lyrics',
        'rap-lyrics',
        'indie-lyrics',
        'metal-lyrics',
    ],
}

JC_SECRET = '~/.cache/jina-now/wolf.json'

SURVEY_LINK = 'https://docs.google.com/forms/d/e/1FAIpQLSex5gMN4wuQc63TriwRqREBfdijwOrATPe7RotcPaT1SSPfEw/viewform?usp=pp_url&entry.1126738320=Jina+NOW+pip+package'
