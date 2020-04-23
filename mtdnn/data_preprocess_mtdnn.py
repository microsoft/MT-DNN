# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
from collections import ChainMap

from mtdnn.common.types import DataFormat
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tasks.utils import (dump_processed_rows, load_cola,
                               load_conll_chunk, load_conll_ner,
                               load_conll_pos, load_mnli, load_mrpc, load_qnli,
                               load_qnnli, load_qqp, load_rte, load_scitail,
                               load_snli, load_sst, load_stsb, load_wnli)

# Map of supported tasks
GLUE_SUPPORTED_TASKS_LOADER_MAP = {
    "cola": load_cola,
    "mnli": load_mnli,
    "mrpc": load_mrpc,
    "qnli": load_qnli,
    "qqnli": load_qnnli,
    "qqp": load_qqp,
    "rte": load_rte,
    "scitail": load_scitail,
    "snli": load_snli,
    "sst": load_sst,
    "stsb": load_stsb,
    "wnli": load_wnli,
}
NER_SUPPORTED_TASKS_LOADER_MAP = {
    "ner": load_conll_ner,
    "pos": load_conll_pos,
    "chunk": load_conll_chunk,
}
SUPPORTED_TASKS_LOADER_MAP = ChainMap(
    GLUE_SUPPORTED_TASKS_LOADER_MAP, NER_SUPPORTED_TASKS_LOADER_MAP
)


class MTDNNTaskDataFileLoader:
    supported_tasks_loader_map = SUPPORTED_TASKS_LOADER_MAP

    def __init__(
        self, data_dir: str, canonical_data_suffix: str, task_defs: MTDNNTaskDefs, seed: int = 13
    ):
        self.data_dir = data_dir
        self.task_defs = task_defs
        self.seed = seed 
        self.canonical_data_dir = os.path.join(self.data_dir, canonical_data_suffix)
        if not os.path.isdir(self.canonical_data_dir):
            os.mkdir(self.canonical_data_dir)

    def load_and_build_data(self, is_old_glue: bool = False):

        """
            data_opts_map[name] = {
                "data_paths": ['train', 'test', 'dev1', 'dev2']",
                "opts": {
                    "header": task.header or True,
                    "is_train": task.is_train or True,
                    "multi_snli": task.multi_snli or False,
                },
            }
            """
        datasets_map: dict = self.task_defs.data_paths_map

        # For each task, load file and build out data
        for name, params in datasets_map.items():

            # TODO - standardize parameters for all loaders to use opts
            opts = params.pop("opts")
            for dataset, path in params.items():
                in_file_path = os.path.join(self.data_dir, path)
                out_file_name = f"{name}_{dataset.split("_")[0]}.tsv"
                out_file_path = os.path.join(self.canonical_data_dir, out_file_name)

                if name not in ["mnli", "qnli"]:
                    try:
                        # Load and dump file
                        data = self.supported_tasks_loader_map[name](in_file_path, **kwargs)
                        dump_processed_rows(data, out_file_path, self.task_defs.data_type_map[name])
                    except expression as ex:
                        raise IOError(ex)
                elif name in ["mnli", "qnli"]:
                    if name == 'mnli':
                        # Do the dev, train, test match, test mismatch
                        pass 
                    elif name == 'qnli':
                        if is_old_glue:
                            random.seed(self.seed)
                             try:
                                # Load and dump file
                                data = self.supported_tasks_loader_map['qnnli'](in_file_path, **kwargs)
                                dump_processed_rows(data, out_file_path, DataFormat.PremiseAndMultiHypothesis)
                            except expression as ex:
                                raise IOError(ex)
                        else:
                            pass



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


# OTHER_SUPPORTED_TASKS_LOADER_MAP = {
#     "squad": SQUADTaskConfig,
#     "squad-v2": SQUADTaskConfig,
# }
