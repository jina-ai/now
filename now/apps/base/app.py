import abc
import os
from typing import Dict, List, Optional

from docarray import DocumentArray

from now.constants import Modalities
from now.dataclasses import UserInput
from now.datasource.datasource import Datasource


class JinaNOWApp:
    """
    Interface for Jina NOW applications.
    To create a new application, you need to inherit from this class and override the methods.
    Example:
        class TextToImageApp(JinaNowApp):
            pass
    """

    def __init__(self):
        curdir = os.path.realpath(__file__)
        self._flow_yaml = os.path.join(curdir, 'flow.yml')

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

    @property
    def flow_yaml(self) -> str:
        """
        Used to configure the flow yaml in the Jina NOW app.
        :return: either the path, to the yaml or the yaml content.
        """
        return self._flow_yaml

    @flow_yaml.setter
    def flow_yaml(self, value: str):
        """
        Set the flow yaml. Either the path to the yaml or the yaml content.
        """
        self._flow_yaml = value

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
    def example_datasource(self) -> List[Datasource]:
        """
        # TODO just a prototype - needs to be implemented in the future
        Get a list of example datasets for the app.

        """
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
        pass

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
