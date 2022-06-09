from now.constants import Qualities

# make sure for all options to have `name` and `description` attribute as this
# will also show up on the terminal as arguments to Jina NOW CLI
QUALITY_CLIP = {
    'name': 'quality',
    'choices': [
        {'name': '🦊 medium (≈3GB mem, 15q/s)', 'value': Qualities.MEDIUM},
        {'name': '🐻 good (≈3GB mem, 2.5q/s)', 'value': Qualities.GOOD},
        {
            'name': '🦄 excellent (≈4GB mem, 0.5q/s)',
            'value': Qualities.EXCELLENT,
        },
    ],
    'prompt_message': 'What quality do you expect?',
    'prompt_type': 'list',
    'description': 'Choose the quality of the model that you would like to finetune',
}

# DATASET_DESCRIPTION = [
#     {'name': '🖼  artworks (≈8K docs)', 'value': DemoDatasets.BEST_ARTWORKS},
#     {'name': '💰 nft - bored apes (10K docs)', 'value': DemoDatasets.NFT_MONKEY},
#     {'name': '👬 totally looks like (≈12K docs)', 'value': DemoDatasets.TLL},
#     {'name': '🦆 birds (≈12K docs)', 'value': DemoDatasets.BIRD_SPECIES},
#     {'name': '🚗 cars (≈16K docs)', 'value': DemoDatasets.STANFORD_CARS},
#     {'name': '🏞 geolocation (≈50K docs)', 'value': DemoDatasets.GEOLOCATION_GEOGUESSR},
#     {'name': '👕 fashion (≈53K docs)', 'value': DemoDatasets.DEEP_FASHION},
#     {'name': '☢️ chest x-ray (≈100K docs)', 'value': DemoDatasets.NIH_CHEST_XRAYS},
# ]
