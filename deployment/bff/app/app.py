import logging.config
import sys

import uvicorn
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.routing import Mount

import deployment.bff.app.settings as api_settings
from deployment.bff.app.decorators import api_method, timed
from deployment.bff.app.v1.routers import image, text

logging.config.dictConfig(api_settings.DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger('bff.app')
logger.setLevel(api_settings.DEFAULT_LOGGING_LEVEL)

TITLE = 'Jina NOW'
DESCRIPTION = 'The Jina NOW service API'
AUTHOR = 'Jina AI'
EMAIL = 'hello@jina.ai'
__version__ = "latest"


def get_app_instance():
    """Build FastAPI app."""
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        contact={
            'author': AUTHOR,
            'email': EMAIL,
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
            f'{TITLE} v{__version__} 🚀 {DESCRIPTION} ✨ '
            f'author: {AUTHOR} email: {EMAIL} 📄  '
            'Check out /docs or /redoc for the API documentation!'
        )

    @app.on_event('startup')
    def startup():
        logger.info(
            f'Jina NOW started! ' f'Listening to [::]:{api_settings.DEFAULT_PORT}'
        )

    return app


def build_app():
    # Image router
    image_mount = "/api/v1/image"
    image_app = get_app_instance()
    image_app.include_router(image.router, tags=['Image'])

    # Text router
    text_mount = "/api/v1/text"
    text_app = get_app_instance()
    text_app.include_router(text.router, tags=['Text'])

    # Mount them - for other modalities just add an app instance
    app = Starlette(routes=[Mount(image_mount, image_app), Mount(text_mount, text_app)])

    return app


application = build_app()


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
