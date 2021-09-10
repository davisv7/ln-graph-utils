#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='ln_graph_utils',
    version='1.0',
    packages=find_packages(include=['ln_graph_utils']),
    description='Common LN Graph Utils I use.',
    author='davis7',
    install_requires=["networkx"],
)
