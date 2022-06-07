import datetime
import os
import sys
import time
from functools import wraps

from yaspin.core import Yaspin


def yaspin_extended(*args, **kwargs):
    return YaspinExtended(*args, **kwargs)


class YaspinExtended(Yaspin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        if 'NOW_CI_RUN' in os.environ:
            self.now_ci_run_t0 = time.time()
            return self
        else:
            return super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        if 'NOW_CI_RUN' not in os.environ:
            return super().__exit__(exc_type, exc_val, traceback)
        else:
            # inspired from Yaspin._freeze and Yaspin._compose_out
            elapsed_time = time.time() - self.now_ci_run_t0
            sec, fsec = divmod(round(100 * elapsed_time), 100)
            text = self._text + " ({}.{:02.0f})\n".format(
                datetime.timedelta(seconds=sec), fsec
            )
            with self._stdout_lock:
                sys.stdout.write(text)

    def ok(self, text="OK"):
        if 'NOW_CI_RUN' not in os.environ:
            return super().ok(text=text)
        else:
            if text:
                self._text = text + ' ' + self._text

    def fail(self, text="FAIL"):
        if 'NOW_CI_RUN' not in os.environ:
            return super().fail(text=text)
        else:
            if text:
                self._text = text + ' ' + self._text


def time_profiler(fun):
    @wraps(fun)
    def profiled_fun(*args, **kwargs):
        if 'NOW_CI_RUN' in os.environ:
            t0 = time.time()
        result = fun(*args, **kwargs)
        if 'NOW_CI_RUN' in os.environ:
            elapsed_time = time.time() - t0
            sec, fsec = divmod(round(100 * elapsed_time), 100)
            print(
                "Time to execute {}.{}: ({}.{:02.0f})".format(
                    fun.__module__, fun.__name__, datetime.timedelta(seconds=sec), fsec
                )
            )
        return result

    return profiled_fun
