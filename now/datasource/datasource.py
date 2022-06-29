import abc
from typing import List, Optional

from docarray import DocumentArray

from now.constants import Modalities, Qualities
from now.data_loading.data_loading import _fetch_da_from_url, get_dataset_url
from now.utils import BetterEnum


class DatasourceType(BetterEnum):
    DEMO = 'demo'
    CUSTOM_DOCARRAY = 'custom-docarray'
    LOCAL_FILES = 'local-files'
    DATABASE_CONNECTOR = 'db-connector'


class Datasource:
    @property
    @abc.abstractmethod
    def type(self) -> DatasourceType:
        raise NotImplementedError('type')

    @property
    @abc.abstractmethod
    def modalities(self) -> List[Modalities]:
        raise NotImplementedError('modalities')

    @abc.abstractmethod
    def get_data(self, *args, **kwargs) -> DocumentArray:
        raise NotImplementedError('get_data')


class DemoDatasource(Datasource):
    def __init__(
        self,
        id_: str,
        display_name: str,
        modality_folder: Modalities,
    ):
        self.id = id_
        self.display_name = display_name
        self.modalities_folder = modality_folder

    @property
    def type(self) -> DatasourceType:
        return DatasourceType.DEMO

    @property
    def modalities(self) -> List[Modalities]:
        return [Modalities.TEXT, Modalities.IMAGE]

    def get_data(self, quality: Optional[Qualities] = None) -> DocumentArray:
        url = get_dataset_url(self.id, quality, self.modalities_folder)
        return _fetch_da_from_url(url)


# example_datasources = [
#     DocarrayURLDatasource('bird-species', '🦆 birds (≈12K docs)', Modalities.IMAGE),
#     ...,
# ]
