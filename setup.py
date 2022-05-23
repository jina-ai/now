import sys
from os import path
import platform
from setuptools import find_packages
from setuptools import setup
import pkg_resources

if sys.version_info < (3, 7, 0):
    raise OSError(f'Jina NOW requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'jina-now'
    libinfo_py = path.join('now', '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

import pathlib


with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]
sys_platform = platform.system().lower()
# if sys_platform == 'darwin':
torch_requirement = ['torch==1.10.2']
# else:
#     torch_requirement = [
#         'torch==1.10.2+cpu',
#     ]


install_requires.extend(torch_requirement)

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='Jina NOW - get your neural search case up and running in minutes.',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='Apache 2.0',
    url='https://github.com/jina-ai/now/',
    download_url='https://github.com/jina-ai/now/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=install_requires,
    entry_points={"console_scripts": ["jina-now = now.cli:cli"]},
    extras_require={
        'test': [
            'pytest',
            'pytest-timeout',
            'pytest-mock',
            'pytest-cov',
            'pytest-repeat',
            'pytest-reraise',
            'mock',
            'pytest-custom_exit_code',
            'black==22.3.0',
            'pytest-split==0.8.0',
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Documentation': 'https://docs.jina.ai',
        'Source': 'https://github.com/jina-ai/now',
        'Tracker': 'https://github.com/jina-ai/now/issues',
    },
    keywords='jina neural-search neural-network deep-learning now private data democratization',
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
)
