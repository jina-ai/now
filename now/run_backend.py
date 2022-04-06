import os
import pickle
import warnings
from os.path import join as osp

from yaspin import yaspin

from now.data_loading.data_loading import fill_missing, load_data
from now.deployment.flow import deploy_flow
from now.dialog import UserInput


def save_mean(da, tmpdir):
    mean = da.embeddings.mean(0)
    if not os.path.exists(osp(tmpdir, 'now/hub/head_encoder/')):
        os.makedirs(osp(tmpdir, 'now/hub/head_encoder/'))
    with open(osp(tmpdir, 'now/hub/head_encoder/mean.bin'), 'wb') as f:
        pickle.dump(mean, f)


def run(user_input: UserInput, is_debug, tmpdir, **kwargs):
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
    ) = parse_user_input(user_input.model_quality, is_debug)

    dataset, ds_type = load_data(
        user_input.dataset,
        user_input.model_quality,
        user_input.is_custom_dataset,
        user_input.custom_dataset_type,
        user_input.dataset_secret,
        user_input.dataset_url,
        user_input.dataset_path,
    )

    finetuning = False
    for d in dataset:
        if 'finetuner_label' in d.tags:
            finetuning = True
            break
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

        add_clip_embeddings(dataset, user_input.model_variant, tmpdir, **kwargs)
        extend_embeddings(dataset['index'], final_layer_output_dim)
        save_mean(dataset['index'], tmpdir)
        fill_missing(dataset, train_val_split_ratio, num_default_val_queries, is_debug)

        # if False:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            finetuned_model_path = finetune_layer(
                dataset, batch_size, final_layer_output_dim, embedding_size, tmpdir
            )

        with yaspin(text="Create overview", color="green") as spinner:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    show_improvement(
                        user_input.dataset,
                        user_input.model_quality,
                        dataset['val_query_image'],
                        dataset['val_index_image'],
                        dataset['val_query'],
                        dataset['val_index'],
                        final_layer_output_dim,
                        embedding_size,
                        finetuned_model_path,
                        class_label='finetuner_label',
                    )
            except Exception:
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
        executor_name,
        dataset['index'],
        user_input.model_variant,
        final_layer_output_dim,
        embedding_size,
        tmpdir,
        finetuning,
        **kwargs,
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
