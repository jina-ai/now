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


class DemoDatasets:
    BEST_ARTWORKS = 'best-artworks'
    NFT_MONKEY = 'nft-monkey'
    TLL = 'tll'
    BIRD_SPECIES = 'bird-species'
    STANFORD_CARS = 'stanford-cars'
    DEEP_FASHION = 'deepfashion'
    NIH_CHEST_XRAYS = 'nih-chest-xrays'
    GEOLOCATION_GEOGUESSR = 'geolocation-geoguessr'
    MUSIC_GENRES_MID = 'music-genres-mid'
    MUSIC_GENRES_LARGE = 'music-genres-large'
    MUSIC_GENRES_EXTRA_LARGE = 'music-genres-xl'
    MUSIC_MID = 'music-mid'
    MUSIC_LARGE = 'music-large'
    MUSIC_EXTRA_LARGE = 'music-xl'
    ROCK_LYRICS = 'rock-lyrics'
    POP_LYRICS = 'pop-lyrics'
    RAP_LYRICS = 'rap-lyrics'
    INDIE_LYRICS = 'indie-lyrics'
    METAL_LYRICS = 'metal-lyrics'


AVAILABLE_DATASET = {
    Modalities.IMAGE: [
        DemoDatasets.BEST_ARTWORKS,
        DemoDatasets.NFT_MONKEY,
        DemoDatasets.TLL,
        DemoDatasets.BIRD_SPECIES,
        DemoDatasets.STANFORD_CARS,
        DemoDatasets.DEEP_FASHION,
        DemoDatasets.NIH_CHEST_XRAYS,
        DemoDatasets.GEOLOCATION_GEOGUESSR,
    ],
    Modalities.MUSIC: [
        DemoDatasets.MUSIC_GENRES_MID,
        DemoDatasets.MUSIC_GENRES_LARGE,
    ],
    Modalities.TEXT: [
        DemoDatasets.ROCK_LYRICS,
        DemoDatasets.POP_LYRICS,
        DemoDatasets.RAP_LYRICS,
        DemoDatasets.INDIE_LYRICS,
        DemoDatasets.METAL_LYRICS,
    ],
}

JC_SECRET = '~/.cache/jina-now/wolf.json'

SURVEY_LINK = 'https://docs.google.com/forms/d/e/1FAIpQLSex5gMN4wuQc63TriwRqREBfdijwOrATPe7RotcPaT1SSPfEw/viewform?usp=pp_url&entry.1126738320=Jina+NOW+pip+package'


PRE_TRAINED_LINEAR_HEADS_MUSIC = {
    DemoDatasets.MUSIC_GENRES_MID: 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/'
    'linear_head_encoder_music_2k',
    DemoDatasets.MUSIC_GENRES_LARGE: 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/'
    'linear_head_encoder_music_10k',
    DemoDatasets.MUSIC_GENRES_EXTRA_LARGE: 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/'
    'linear_head_encoder_music_40k',
}
