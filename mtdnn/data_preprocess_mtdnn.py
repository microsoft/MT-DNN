# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os

from common.glue.glue_utils import *
from mtdnn.tasks.config import MTDNNTaskDefs

GLUE_TAKS_LOADER


class MTDNNDataPreprocess:
    def __init__(
        self,
        task_defs: MTDNNTaskDefs,
        data_dir: str = "data",
        is_old_glue: bool = False,
        seed: int = 13,
    ):
        assert os.path.exists(data_dir)
        self.task_defs = task_defs
        self.data_dir = data_dir
        self.is_old_glue = is_old_glue
        self.seed = seed

    def load_data(self):
        pass

    def build_data(self):
        pass

