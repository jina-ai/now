import abc
import os
from typing import Dict, List, Optional

from docarray import DocumentArray

from now.constants import AVAILABLE_DATASET, Modalities, Qualities
from now.dataclasses import UserInput
from now.datasource.datasource import DemoDatasource


class JinaNOWApp:
    """
    Interface for Jina NOW applications.
    To create a new application, you need to inherit from this class and override the methods.
    Example:
        class TextToImageApp(JinaNowApp):
            pass
    """

    def __init__(self):
        self.flow_yaml = ''

        self.set_flow_yaml()

    @property
    def app(self) -> str:
        """
        Name of the app. Should be an enum value set in now.constants.Apps
        """
        raise NotImplementedError()

    @property
    def is_enabled(self) -> bool:
        """
        Set to True if this app is enabled for the end user.
        """
        raise NotImplementedError()

    @property
    def description(self) -> str:
        """
        Short description of the app.
        """
        return 'Jina NOW app'

    @property
    @abc.abstractmethod
    def input_modality(self) -> Modalities:
        """
        Modality used for running search queries
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def output_modality(self) -> Modalities:
        """
        Modality used for indexing data
        """
        raise NotImplementedError()

    def set_flow_yaml(self, **kwargs):
        """Used to configure the flow yaml in the Jina NOW app."""
        flow_dir = os.path.abspath(os.path.join(__file__, '..'))
        self.flow_yaml = os.path.join(flow_dir, 'flow.yml')

    @property
    def bff(self) -> Optional[str]:
        """
        TODO This function is currently not used but already introduces the concept of custom bff
        Used to configure the bff which is used to define input and output format.
        :return: the path to the file where the bff routs are configured
        """
        return None

    @property
    def playground(self) -> Optional[str]:
        """
        TODO This function is currently not used but already introduces the concept of custom playground
        Used to configure the playground(streamlit app) where the user can run example queries
        :return: the path to the streamlit file.
        """
        return None

    @property
    def options(self) -> List[Dict]:
        """
        Get the options which are used to configure the app.
        On CLI the user will get a prompt and at the storefront, a GUI will be generated accordingly.
        Example:
        return [
            {
                name='quality',
                choices=[
                    {'name': '🦊 medium (≈3GB mem, 15q/s)', 'value': 'openai/clip-vit-base-patch32'},
                    {'name': '🐻 good (≈3GB mem, 2.5q/s)', 'value': 'openai/clip-vit-base-patch16'},
                    {'name': '🦄 excellent (≈4GB mem, 0.5q/s)','value': 'openai/clip-vit-large-patch14',},
                ],
                prompt_message='What quality do you expect?',
                prompt_type='list'
            }
        ]
        :return:
        """
        return []

    @property
    def pre_trained_embedding_size(self) -> Dict[Qualities, int]:
        """
        Returns a dictionary which maps given quality to embedding size of pretrained model.
        """
        return {}

    @property
    def example_datasource(self) -> List[DemoDatasource]:
        """
        # TODO just a prototype - needs to be implemented in the future
        Get a list of example datasets for the app.

        """
        if self.output_modality in AVAILABLE_DATASET:
            return [
                DemoDatasource(
                    id_=ds[0], display_name=ds[1], modality_folder=self.output_modality
                )
                for ds in AVAILABLE_DATASET[self.output_modality]
            ]
        else:
            return []

    def set_app_parser(self, parser, formatter) -> None:
        """
        This parser reads from the `options` property and parses it
        to form the command line arguments for app
        """
        if self.is_enabled:
            parser = parser.add_parser(
                self.app,
                help=self.description,
                description=f'Create an {self.app} app.',
                formatter_class=formatter,
            )
            for option in self.options:
                parser.add_argument(
                    f'--{option["name"]}',
                    help=option['description'],
                    type=str,
                )

    def check_requirements(self) -> bool:
        """
        Returns true if all requirements on the system are satisfied. Else False.
        """
        return True

    # TODO Remove kubectl_path. At the moment, the setup function needs kubectl because of finetuning a custom
    #  dataset with local deployment. In that case, inference is done on the k8s cluster.
    def setup(
        self, da: DocumentArray, user_config: UserInput, kubectl_path: str
    ) -> Dict:
        """
        Runs before the flow is deployed.
        Common use cases:
            - create a database
            - finetune a model + push the artifact
            - notify other services
            - check if starting the app is currently possible
        :param da:
        :param user_config: user configuration based on the given options
        :return: dict used to replace variables in flow yaml and to clean up resources after the flow is terminated
        """
        return {}

    def cleanup(self, app_config: dict) -> None:
        """
        Runs after the flow is terminated.
        Cleans up the resources created during setup.
        Common examples are:
            - delete a database
            - remove artifact
            - notify other services
        :param app_config: contains all information needed to clean up the allocated resources
        """
        pass
