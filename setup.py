#!/usr/bin/env python3

from setuptools import setup


with open("requirements.txt") as f:
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
)
