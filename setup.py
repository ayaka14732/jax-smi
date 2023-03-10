from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='jax-smi',
    version='1.0.3',
    description='JAX Synergistic Memory Inspector',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ayaka14732/jax-smi',
    author='Ayaka Mikazuki',
    author_email='ayaka@mail.shn.hk',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='jax machine-learning',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='>=3.8, <4',
    install_requires=['jax>=0.2.16', 'fire'],
    entry_points = {
        'console_scripts': ['jax-smi=jax_smi.cli_tool:main'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/ayaka14732/jax-smi/issues',
        'Source': 'https://github.com/ayaka14732/jax-smi',
    },
    zip_safe=False,
)
