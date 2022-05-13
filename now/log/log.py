import os

from yaspin.core import Yaspin


def yaspin_extended(*args, **kwargs):
    return YaspinExtended(*args, **kwargs)


class YaspinExtended(Yaspin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        if 'NOW_CI_RUN' in os.environ:
            return self
        else:
            return super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        if 'NOW_CI_RUN' not in os.environ:
            return super().__exit__(exc_type, exc_val, traceback)
