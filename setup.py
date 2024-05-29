#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['typer', 'numpy', 'bionumpy', 'pandas', 'plotly', 'scikit-learn', 'matplotlib', 'diskcache', 'geopy',
                'pooch',
                'python-dateutil', 'meteostat', 'cyclopts', 'requests', 'pydantic', 'pyyaml',
                'geopandas', 'libpysal', 'docker', 'jax', 'jaxlib', 'blackjax', 'fastapi']

test_requirements = ['pytest>=3', "hypothesis"]

setup(
    author="Sandvelab",
    author_email='sandvelab',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MITupdate proje License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="Main repository for joint programming project on climate health",
    entry_points={
        'console_scripts': [
            'climate_health=climate_health.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='climate_health',
    name='climate_health',
    packages=find_packages(include=['climate_health', 'climate_health.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sandvelab/climate_health',
    version='0.0.1',
    zip_safe=False,
)
