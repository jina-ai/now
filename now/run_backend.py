import os
import pickle
import warnings
from os.path import join as osp

from yaspin import yaspin

from now.data_loading.data_loading import load_data
from now.deployment.flow import deploy_flow
from now.dialog import UserInput
from now.finetuning.finetuning import fill_missing
from now.utils import sigmap


def save_mean(da, tmpdir):
    mean = da.embeddings.mean(0)
    if not os.path.exists(osp(tmpdir, 'now/hub/head_encoder/')):
        os.makedirs(osp(tmpdir, 'now/hub/head_encoder/'))
    with open(osp(tmpdir, 'now/hub/head_encoder/mean.bin'), 'wb') as f:
        pickle.dump(mean, f)


def is_finetuning(dataset_name, dataset):
    # finetuning for some datasets is deactivated since the generalization ability is lost
    # they can be reactivated once this ticket is done:
    # https://github.com/jina-ai/now/issues/76
    if dataset_name in [
        'tll',
        'nft-monkey',
        # 'deepfashion',
        'nih-chest-xrays',
        'geolocation-geoguessr',
        'stanford-cars',
        # 'bird-species',
        'best-artworks',
        'lyrics',
        'lyrics-10000',
        'rock-lyrics',
        'pop-lyrics',
        'rap-lyrics',
        'indie-lyrics',
        'metal-lyrics',
    ]:
        return False
    for d in dataset:
        if 'finetuner_label' in d.tags:
            return True
    return False


def run(user_input: UserInput, is_debug, tmpdir, kubectl_path: str):
    """
    Args:
        user_input: User input arguments
        is_debug: if True it also works on small datasets
    """
    # bring back at some point to make this configurable by the user
    (
        final_layer_output_dim,
        embedding_size,
        batch_size,
        train_val_split_ratio,
        num_default_val_queries,
    ) = parse_user_input(user_input.quality, is_debug)

    dataset = load_data(user_input)

    finetuning = is_finetuning(user_input.data, dataset)

    if not finetuning:
        embedding_size = int(final_layer_output_dim / 2)
    dataset = {
        'index': dataset,
        'train': None,
        'val': None,
        'val_query': None,
        'val_index': None,
        'val_query_image': None,
        'val_index_image': None,
    }

    if finetuning:
        from now.finetuning.finetuning import add_clip_embeddings, finetune_layer
        from now.hub.head_encoder.head_encoder import extend_embeddings
        from now.hub.hub import push_to_hub
        from now.improvements.improvements import show_improvement

        add_clip_embeddings(
            dataset,
            user_input.model_variant,
            tmpdir,
            kubectl_path=kubectl_path,
        )
        extend_embeddings(dataset['index'], final_layer_output_dim)
        save_mean(dataset['index'], tmpdir)
        fill_missing(dataset, train_val_split_ratio, num_default_val_queries, is_debug)

        # if False:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            finetuned_model_path = finetune_layer(
                dataset, batch_size, final_layer_output_dim, embedding_size, tmpdir
            )

        with yaspin(sigmap=sigmap, text="Create overview", color="green") as spinner:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    show_improvement(
                        user_input.data,
                        user_input.quality,
                        dataset['val_query_image'],
                        dataset['val_index_image'],
                        dataset['val_query'],
                        dataset['val_index'],
                        final_layer_output_dim,
                        embedding_size,
                        finetuned_model_path,
                        class_label='finetuner_label',
                    )
            except Exception as e:
                pass
            spinner.ok('🖼')
        print(
            f'before-after comparison result is saved in the current working directory as image'
        )
        executor_name = push_to_hub(tmpdir)
    else:
        executor_name = None
    # executor_name = 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/
    # linear_head_encoder_2022-02-20_20-35-15'
    # print('###executor_name', executor_name)
    # executor_name = 'FineTunedLinearHeadEncoder:93ea59dbd1ee3fe0bdc44252c6e86a87/
    # deleteme_2022-02-06_13-20-37'
    (
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = deploy_flow(
        executor_name=executor_name,
        output_modality=user_input.output_modality,
        index=dataset['index'],
        vision_model=user_input.model_variant,
        final_layer_output_dim=final_layer_output_dim,
        embedding_size=embedding_size,
        tmpdir=tmpdir,
        finetuning=finetuning,
        sandbox=user_input.sandbox,
        kubectl_path=kubectl_path,
    )
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal


def parse_user_input(quality, is_debug):
    if is_debug:
        train_val_split_ratio = 0.5
        batch_size = 10
    else:
        batch_size = 128
        train_val_split_ratio = 0.9
    num_default_val_queries = 10

    if quality == 'ViT-L14':
        final_layer_output_dim = 768 * 2
    else:
        final_layer_output_dim = 512 * 2
    embedding_size = 128

    return (
        final_layer_output_dim,
        embedding_size,
        batch_size,
        train_val_split_ratio,
        num_default_val_queries,
    )
