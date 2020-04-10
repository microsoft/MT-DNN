#!/usr/bin/python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Code adapted from https://github.com/microsoft/nlp-recipes/blob/master/tools/generate_conda_file.py

# This script creates yaml files to build conda environments
# For generating a conda file for running only python code:
# $ python generate_conda_file.py
#
# For generating a conda file for running python gpu:
# $ python generate_conda_file.py --gpu


import argparse
import textwrap
from sys import platform


HELP_MSG = """
To create the conda environment:
$ conda env create -f {conda_env}.yaml

To update the conda environment:
$ conda env update -f {conda_env}.yaml

To register the conda environment in Jupyter:
$ conda activate {conda_env}
$ python -m ipykernel install --user --name {conda_env} \
--display-name "Python ({conda_env})"
"""


CHANNELS = ["defaults", "conda-forge", "pytorch"]

CONDA_BASE = {
    "python": "python==3.6.8",
    "pip": "pip>=19.1.1",
    "ipykernel": "ipykernel>=4.6.1",
    "jupyter": "jupyter>=1.0.0",
    "matplotlib": "matplotlib>=2.2.2",
    "numpy": "numpy>=1.16.2",
    "pandas": "pandas>=0.24.2",
    "pytest": "pytest>=3.6.4",
    "pytorch": "pytorch-cpu>=1.0.0",
    "scipy": "scipy>=1.0.0",
    "h5py": "h5py>=2.8.0",
    "tensorflow": "tensorflow==1.15.2",
    "tensorflow-hub": "tensorflow-hub==0.7.0",
    "dask": "dask[dataframe]==1.2.2",
    "papermill": "papermill>=1.0.1",
}

CONDA_GPU = {
    "numba": "numba>=0.38.1",
    "cudatoolkit": "cudatoolkit==10.2.89",
}

PIP_BASE = {
    "allennlp": "allennlp==0.8.4",
    "black": "black>=18.6b4",
    "cached-property": "cached-property==1.5.1",
    "jsonlines": "jsonlines>=1.2.0",
    "nteract-scrapbook": "nteract-scrapbook>=0.2.1",
    "pytorch-pretrained-bert": "pytorch-pretrained-bert>=0.6",
    "tqdm": "tqdm==4.32.2",
    "pyemd": "pyemd==0.5.1",
    "ipywebrtc": "ipywebrtc==0.4.3",
    "pre-commit": "pre-commit>=1.14.4",
    "seaborn": "seaborn>=0.9.0",
    "sklearn-crfsuite": "sklearn-crfsuite>=0.3.6",
    "spacy": "spacy==2.1.8",
    "spacy-models": (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz"
    ),
    "transformers": "transformers==2.3.0",
    "gensim": "gensim>=3.7.0",
    "nltk": "nltk>=3.4",
    "seqeval": "seqeval==0.0.12",
    "bertsum": "git+https://github.com/daden-ms/BertSum.git@030c139c97bc57d0c31f6515b8bf9649f999a443#egg=BertSum",
    "pyrouge": "pyrouge>=0.1.3",
    "py-rouge": "py-rouge>=1.1",
    "torchtext": "torchtext>=0.4.0",
    "multiprocess": "multiprocess==0.70.9",
    "tensorboardX": "tensorboardX==1.8",
    "tensorboard": "tensorboard",
    "colorlog": "colorlog",
    "boto3": "boto3",
    "regex": "regex",
    "scikit-learn": "scikit-learn",
    "pyyaml": "pyyaml",
    "future": "future",
    "fairseq": "fairseq==0.8.0",
    "sentencepiece": "sentencepiece",    
}

PIP_GPU = {
    "torch": "torch==1.4.0",
}

PIP_DARWIN = {}
PIP_DARWIN_GPU = {}

PIP_LINUX = {}
PIP_LINUX_GPU = {}

PIP_WIN32 = {}
PIP_WIN32_GPU = {}

CONDA_DARWIN = {}
CONDA_DARWIN_GPU = {}

CONDA_LINUX = {}
CONDA_LINUX_GPU = {}

CONDA_WIN32 = {}
CONDA_WIN32_GPU = {"pytorch": "pytorch==1.0.0", "cudatoolkit": "cuda90"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
        This script generates a conda file for different environments.
        Plain python is the default,
        but flags can be used to support GPU functionality."""
        ),
        epilog=HELP_MSG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", help="specify name of conda environment")
    parser.add_argument(
        "--gpu", action="store_true", help="include packages for GPU support"
    )
    args = parser.parse_args()

    # set name for environment and output yaml file
    conda_env = "mtdnn_cpu"
    if args.gpu:
        conda_env = "mtdnn_gpu"

    # overwrite environment name with user input
    if args.name is not None:
        conda_env = args.name

    # add conda and pip base packages
    conda_packages = CONDA_BASE
    pip_packages = PIP_BASE

    # update conda and pip packages based on flags provided
    if args.gpu:
        conda_packages.update(CONDA_GPU)
        pip_packages.update(PIP_GPU)

    # update conda and pip packages based on os platform support
    if platform == "darwin":
        conda_packages.update(CONDA_DARWIN)
        pip_packages.update(PIP_DARWIN)
        if args.gpu:
            conda_packages.update(CONDA_DARWIN_GPU)
            pip_packages.update(PIP_DARWIN_GPU)
    elif platform.startswith("linux"):
        conda_packages.update(CONDA_LINUX)
        pip_packages.update(PIP_LINUX)
        if args.gpu:
            conda_packages.update(CONDA_LINUX_GPU)
            pip_packages.update(PIP_LINUX_GPU)
    elif platform == "win32":
        conda_packages.update(CONDA_WIN32)
        pip_packages.update(PIP_WIN32)
        if args.gpu:
            conda_packages.update(CONDA_WIN32_GPU)
            pip_packages.update(PIP_WIN32_GPU)
    else:
        raise Exception("Unsupported platform. Must be Windows, Linux, or macOS")

    # write out yaml file
    conda_file = "{}.yaml".format(conda_env)
    with open(conda_file, "w") as f:
        for line in HELP_MSG.format(conda_env=conda_env).split("\n"):
            f.write("# {}\n".format(line))
        f.write("name: {}\n".format(conda_env))
        f.write("channels:\n")
        for channel in CHANNELS:
            f.write("- {}\n".format(channel))
        f.write("dependencies:\n")
        for conda_package in conda_packages.values():
            f.write("- {}\n".format(conda_package))
        f.write("- pip:\n")
        for pip_package in pip_packages.values():
            f.write("  - {}\n".format(pip_package))

    print("Generated conda file: {}".format(conda_file))
    print(HELP_MSG.format(conda_env=conda_env))
