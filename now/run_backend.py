from now.data_loading.data_loading import load_data
from now.deployment.flow import deploy_flow
from now.dialog import UserInput
from now.finetuning.run_finetuning import finetune_now
from now.finetuning.settings import parse_finetune_settings


def run(user_input: UserInput, tmpdir, kubectl_path: str):
    """
    TODO: Write docs

    :param user_input:
    :param tmpdir:
    :param kubectl_path:
    :return:
    """
    dataset = load_data(user_input)

    finetune_settings = parse_finetune_settings(user_input, dataset)
    if finetune_settings.perform_finetuning:
        print(f'🔧 Perform finetuning!')
        finetune_settings.finetuned_model_name = finetune_now(
            user_input, dataset, finetune_settings, kubectl_path
        )
    (
        gateway_host,
        gateway_port,
        gateway_host_internal,
        gateway_port_internal,
    ) = deploy_flow(
        user_input=user_input,
        finetune_settings=finetune_settings,
        index=dataset,
        tmpdir=tmpdir,
        kubectl_path=kubectl_path,
    )
    return gateway_host, gateway_port, gateway_host_internal, gateway_port_internal
