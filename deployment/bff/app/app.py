import logging.config
import sys

import uvicorn
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.routing import Mount

import deployment.bff.app.settings as api_settings
from deployment.bff.app.decorators import api_method, timed
from deployment.bff.app.v1.routers import (
    img2img,
    img2txt,
    music2music,
    text2text,
    txt2img,
    txt2video,
)

logging.config.dictConfig(api_settings.DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger('bff.app')
logger.setLevel(api_settings.DEFAULT_LOGGING_LEVEL)

TITLE = 'Jina NOW'
DESCRIPTION = 'The Jina NOW service API'
AUTHOR = 'Jina AI'
EMAIL = 'hello@jina.ai'
__version__ = 'latest'


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
    # Image2Image router
    img2img_mount = '/api/v1/image-to-image'
    img2img_app = get_app_instance()
    img2img_app.include_router(img2img.router, tags=['Image-To-Image'])

    # Image2Text router
    img2txt_mount = '/api/v1/image-to-text'
    img2txt_app = get_app_instance()
    img2txt_app.include_router(img2txt.router, tags=['Image-To-Text'])

    # Text2Image router
    txt2img_mount = '/api/v1/text-to-image'
    txt2img_app = get_app_instance()
    txt2img_app.include_router(txt2img.router, tags=['Text-To-Image'])

    # Text2Text router
    text2text_mount = '/api/v1/text-to-text'
    text2text_app = get_app_instance()
    text2text_app.include_router(text2text.router, tags=['Text-To-Text'])

    # Music2Music router
    music2music_mount = '/api/v1/music-to-music'
    music2music_app = get_app_instance()
    music2music_app.include_router(music2music.router, tags=['Music-To-Music'])

    # Text2Video router
    text2video_mount = '/api/v1/text-to-video'
    text2video_app = get_app_instance()
    text2video_app.include_router(txt2video.router, tags=['Text-To-Video'])

    # Mount them - for other modalities just add an app instance
    app = Starlette(
        routes=[
            Mount(img2img_mount, img2img_app),
            Mount(img2txt_mount, img2txt_app),
            Mount(txt2img_mount, txt2img_app),
            Mount(text2text_mount, text2text_app),
            Mount(music2music_mount, music2music_app),
            Mount(text2video_mount, text2video_app),
        ]
    )

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
