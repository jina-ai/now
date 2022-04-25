import logging

logger = logging.getLogger(__name__)


ENV_PREFIX = 'JINA_NOW_'

# server
DEFAULT_WORKERS = 1
DEFAULT_PORT = 8080
DEFAULT_BACKLOG = 2048

# debug flag
DEFAULT_DEBUG = True

# logging
DEFAULT_LOGGING_LEVEL = 'DEBUG'
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'basic': {
            'class': 'logging.Formatter',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'format': '%(asctime)s  %(name)-30s  %(levelname)8s  ::  %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'basic',
            'stream': 'ext://sys.stdout',
        }
    },
    'root': {'level': 'DEBUG', 'handlers': ['console']},
}
