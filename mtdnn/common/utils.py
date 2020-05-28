# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# Some code referenced from https://github.com/microsoft/nlp-recipes
import json
import logging
import math
import os
import random
import subprocess
import sys
import tarfile
import zipfile
from contextlib import contextmanager
from logging import Logger
from tempfile import TemporaryDirectory

import numpy
import requests
import torch
from tqdm import tqdm
from time import gmtime, strftime


class MTDNNCommonUtils:
    @staticmethod
    def set_environment(seed, set_cuda=False):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and set_cuda:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def patch_var(v, cuda=True):
        if cuda:
            v = v.cuda(non_blocking=True)
        return v

    @staticmethod
    def get_gpu_memory_map():
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    @staticmethod
    def get_pip_env():
        result = subprocess.call(["pip", "freeze"])
        return result

    @staticmethod
    def load_pytorch_model(local_model_path: str = ""):
        state_dict = None
        assert os.path.exists(local_model_path), "Model File path doesn't exist"
        state_dict = torch.load(local_model_path)
        return state_dict

    @staticmethod
    def dump(path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def generate_decoder_opt(enable_san, max_opt):
        opt_v = 0
        if enable_san and max_opt < 3:
            opt_v = max_opt
        return opt_v

    @staticmethod
    def create_logger(name, silent=False, to_disk=False, log_file="run.log"):
        """ Logger wrapper """
        # setup logger
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.propagate = False
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S",
        )
        if not silent:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            log.addHandler(ch)
        if to_disk:
            log_file = (
                log_file
                if log_file is not None
                else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
            )
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        return log

    @staticmethod
    def create_directory_if_not_exists(dir_path: str):
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    @contextmanager
    def download_path(path=None):
        tmp_dir = TemporaryDirectory()
        if not path:
            path = tmp_dir.name
        else:
            path = os.path.realpath(path)

        try:
            yield path
        finally:
            tmp_dir.cleanup()

    @staticmethod
    def maybe_download(
        url, filename=None, work_directory=".", expected_bytes=None, log: Logger = None
    ):
        """Download a file if it is not already downloaded.

        Args:
            filename (str): File name.
            work_directory (str): Working directory.
            url (str): URL of the file to download.
            expected_bytes (int): Expected file size in bytes.
        Returns:
            str: File path of the file downloaded.
        """
        if filename is None:
            filename = url.split("/")[-1]
        os.makedirs(work_directory, exist_ok=True)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            if not os.path.isdir(work_directory):
                os.makedirs(work_directory)
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)

            with open(filepath, "wb") as file:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
                ):
                    file.write(data)
        else:
            log.debug("File {} already downloaded".format(filepath))
        if expected_bytes is not None:
            statinfo = os.stat(filepath)
            if statinfo.st_size != expected_bytes:
                os.remove(filepath)
                raise IOError("Failed to verify {}".format(filepath))

        return filepath
