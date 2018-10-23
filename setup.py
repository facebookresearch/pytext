#!/usr/bin/env python3

import os
from setuptools import setup


DIR = os.path.dirname(__file__)
REQUIREMENTS = os.path.join(DIR, "requirements.txt")


with open(REQUIREMENTS) as f:
    reqs = f.read()

setup(
    name="pytext",
    version="0.1",
    description="pytorch modeling framework and model zoo for text models",
    url="https://github.com/facebookresearch/PyText",
    author="Facebook",
    license="BSD",
    packages=["pytext"],
    install_requires=reqs.strip().split("\n"),
    dependency_links=[
        "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",
    ]
)
