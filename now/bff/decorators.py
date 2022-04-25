import inspect
import logging
import time
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from typing import Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)


@contextmanager
def _count_elapsed(func: Callable):
    """
    A private context manager used for decoupling the timing logic in the timed
    decorator that comes in asyncio and non-asyncio variants.
    """
    start = time.time()
    yield
    elapsed = str(timedelta(seconds=(time.time() - start)))
    logger.debug(f'Execution of .{func.__name__}() took {elapsed}')


def timed(func: Callable):
    """
    Decorator to count and log elapsed time of a function call.
    """

    @wraps(func)
    def surround(*args, **kwargs):
        with _count_elapsed(func):
            return func(*args, **kwargs)

    return surround


def async_timed(func: Callable):
    """
    Decorator to count and log elapsed time of an async function call.
    """

    @wraps(func)
    async def surround(*args, **kwargs):
        with _count_elapsed(func):
            return await func(*args, **kwargs)

    return surround


def api_method(func: Callable):
    """Decorator to log api call info and handle exceptions."""

    @wraps(func)
    def surround(*args, **kwargs):
        func_name = inspect.getmodule(func).__name__ + f':{func.__name__}'
        logger.info(f'--- Calling api method {func_name} ...')
        try:
            response = func(*args, **kwargs)
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            logger.error(
                f'An error occurred while running {func.__name__}(), '
                f'details: {str(exc)}'
            )
            logger.exception(exc)
            raise HTTPException(status_code=500, detail='Unknown error')
        return response

    return surround
