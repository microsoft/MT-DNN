# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import os
import re
from os.path import dirname, join

from setuptools import setup

from mtdnn import AUTHOR, LICENSE, TITLE, VERSION


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()

setup(
    name="mtdnn",
    version=VERSION,
    license=LICENSE,
    description="Multi-Task Deep Neural Networks for Natural Language Understanding. Developed by Microsoft Research AI",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CONTRIBUTION.md")),
    ),
    author=AUTHOR,
    author_email="xiaodl@microsoft.com",
    url="https://github.com/microsoft/mt-dnn",
    packages=["mtdnn"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",
    ],
    project_urls={
        "Documentation": "https://github.com/microsoft/mt-dnn/",
        "Issue Tracker": "https://github.com/microsoft/mt-dnn/issues",
    },
    keywords=[
        "Microsoft NLP",
        "Microsoft MT-DNN",
        "Mutli-Task Deep Neural Network for Natual Language Understanding",
        "Natural Language Processing",
        "Text Processing",
        "Word Embedding",
        "Multi-Task DNN",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch==1.4.0",
        "tqdm",
        "colorlog",
        "boto3",
        "pytorch-pretrained-bert==0.6.0",
        "regex",
        "scikit-learn",
        "pyyaml",
        "pytest",
        "sentencepiece",
        "tensorboardX",
        "tensorboard",
        "future",
        "fairseq==0.8.0",
        "seqeval==0.0.12",
        "transformers==2.9.0",
    ],
    dependency_links=[],
    extras_require={},
    use_scm_version=False,
    setup_requires=[],
)
