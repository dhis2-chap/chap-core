#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "virtualenv",
    "numpy<2.0",
    "bionumpy",
    "pandas",
    "plotly",
    "scikit-learn",
    "matplotlib",
    "diskcache",
    "geopy",
    "pooch",
    "python-dateutil",
    "meteostat",
    "cyclopts",
    "requests",
    "fastapi",
    "pydantic>=2.0",
    "pyyaml",
    "geopandas",
    "libpysal",
    "docker",
    "scipy",
    "gitpython",
    "earthengine-api",
    "python-dotenv",
    "rq",
    "python-multipart",
    "uvicorn",
    "pydantic-geojson",
    "annotated_types",
    "pycountry",
    "unidecode",
    "httpx",
    "earthengine-api",
    "mlflow",
    "gluonts",
    "xarray",
]

test_requirements = ["pytest>=3", "hypothesis"]

setup(
    author="Sandvelab",
    author_email="knutdrand@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="Main repository for joint programming project on climate health",
    entry_points={
        "console_scripts": [
            "chap=chap_core.cli:main",
            "chap-cli=chap_core.chap_cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description="Chap Core",
    include_package_data=True,
    keywords="chap_core",
    name="chap_core",
    packages=find_packages(include=["chap_core", "chap_core.*", "chap_core.*.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dhis2/chap-core",
    version="0.0.8",
    zip_safe=False,
)
