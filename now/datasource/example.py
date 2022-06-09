from docarray import DocumentArray

from now.constants import BASE_STORAGE_URL, Modalities
from now.datasource.datasource import Datasource


# TODO just a prototype - needs to be implemented in the future
class DocarrayURLDatasource(Datasource):
    def __init__(self, ds_id, display_name, modality):
        self.ds_id = ds_id
        self.display_name = display_name
        self.modality = modality

    def get_data(self, quality: str) -> DocumentArray:
        # TODO comment from sebastian:
        #  This signature misses the mapping arg from the baseclass.
        #  The baseclass interface is not really guiding the user a lot at the moment.
        #  Maybe we should start with an explicit ObjectStorageDataset or something like that
        #  because all our app are using that atm. and abstract the interface furhter once we work on
        #  the database table column mapping stuff?

        url = f'{BASE_STORAGE_URL}/{self.modality}/{self.ds_id}{("." + quality) if quality is not None else ""}.bin'
        # TODO  return document array from url
        raise NotImplementedError()


class DocarrayPullDatasource(Datasource):
    def __init__(self, name):
        self.name = name

    def get_data(self, quality: str) -> DocumentArray:
        pass


example_datasources = [
    ExampleDatasource('bird-species', '🦆 birds (≈12K docs)', Modalities.IMAGE),
    ...,
]
