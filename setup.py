#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import setuptools


DIR = os.path.dirname(__file__)
REQUIREMENTS = os.path.join(DIR, "requirements.txt")


with open(REQUIREMENTS) as f:
    reqs = f.read()

setuptools.setup(
    name="pytext-nlp",
    version="0.3.3",
    description="pytorch modeling framework and model zoo for text models",
    url="https://github.com/facebookresearch/PyText",
    author="Facebook",
    license="BSD",
    packages=setuptools.find_packages(),
    install_requires=reqs.strip().split("\n"),
    entry_points={"console_scripts": ["pytext = pytext.main:main"]},
)
