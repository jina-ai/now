from yaspin.core import Yaspin

TEST = False


def yaspin_extended(*args, **kwargs):
    return YaspinExtended(*args, **kwargs)


class YaspinExtended(Yaspin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        if not TEST:
            return super().__enter__()
        else:
            return self

    def __exit__(self, exc_type, exc_val, traceback):
        if not TEST:
            return super().__exit__(exc_type, exc_val, traceback)
