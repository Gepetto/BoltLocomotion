"""Installation script for the 'cat_envs' python package."""

import os
import tomli  # Use tomli instead of toml

from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_FILE = os.path.join(EXTENSION_PATH, "config", "extension.toml")

with open(EXTENSION_TOML_FILE, "rb") as f:
    EXTENSION_TOML_DATA = tomli.load(f)

INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="cat_envs",
    packages=["cat_envs"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
