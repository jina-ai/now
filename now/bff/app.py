import logging.config
import sys

import uvicorn
from fastapi import FastAPI

import now.bff.settings as api_settings
from now.bff import __author__, __email__, __summary__, __title__, __version__
from now.bff.decorators import api_method, timed
from now.bff.v1.api import v1_router

logging.config.dictConfig(api_settings.DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger('bff.app')
logger.setLevel(api_settings.DEFAULT_LOGGING_LEVEL)


def build_app():
    """Build FastAPI app."""
    app = FastAPI(
        title=__title__,
        description=__summary__,
        version=__version__,
        contact={
            'author': __author__,
            'email': __email__,
        },
    )

    @app.get('/ping')
    @api_method
    @timed
    def check_liveness() -> str:
        """
        Sanity check - this will let the caller know that the service is operational.
        """
        return 'pong!'

    @app.get('/')
    @api_method
    @timed
    def read_root() -> str:
        """
        Root path welcome message.
        """
        return (
            f'{__title__} v{__version__} 🚀 {__summary__} ✨ '
            f'author: {__author__} email: {__email__} 📄  '
            'Check out /docs or /redoc for the API documentation!'
        )

    @app.on_event('startup')
    def startup():
        logger.info(
            f'Jina NOW v{__version__} started! '
            f'Listening to [::]:{api_settings.DEFAULT_PORT}'
        )

    app.include_router(v1_router, prefix='/api/v1')

    return app


def run_server():
    """Run server."""
    app = build_app()

    # start the server!
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8080,
        loop='uvloop',
        http='httptools',
    )


if __name__ == '__main__':
    try:
        run_server()
    except Exception as exc:
        logger.critical(str(exc))
        logger.exception(exc)
        sys.exit(1)
