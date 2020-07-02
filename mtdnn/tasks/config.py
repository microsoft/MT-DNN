# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

# This script reuses some code from https://github.com/huggingface/transformers


""" Model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from mtdnn.common.san import SANClassifier

from mtdnn.common.loss import LossCriterion
from mtdnn.common.metrics import Metric
from mtdnn.common.types import DataFormat, EncoderModelType, TaskDefType, TaskType
from mtdnn.common.utils import MTDNNCommonUtils
from mtdnn.common.vocab import Vocabulary


logger = MTDNNCommonUtils.create_logger(__name__, to_disk=True)

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


class TaskConfig(object):
    """Base Class for Task Configurations

    Handles parameters that are common to all task configurations

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, **kwargs: dict):
        """ Define a generic task configuration """
        logger.info("Mapping Task attributes")

        # assert data exists for preprocessing
        assert (
            "data_source_dir" in kwargs
        ), "[ERROR] - Source data directory with data splits not provided"
        assert (
            kwargs["data_source_dir"] and type(kwargs["data_source_dir"]) == str
        ), "[ERROR] - Source data directory path must be a string"
        assert kwargs[
            "data_source_dir"
        ], "[ERROR] - Source data directory path cannot be empty"
        assert os.path.isdir(
            kwargs["data_source_dir"]
        ), "[ERROR] - Source data directory path does not exist"

        assert all(
            os.path.exists(os.path.join(kwargs["data_source_dir"], f"{split}.tsv"))
            for split in kwargs["split_names"]
        ), f"[ERROR] - All data splits do not exist in path - {kwargs['data_source_dir']}"

        assert kwargs[
            "data_process_opts"
        ], "[ERROR] - Source data processing options must be set"

        # Mapping attributes
        for key, value in kwargs.items():
            try:
                if key == "data_source_dir":
                    data_paths = [
                        os.path.join(kwargs["data_source_dir"], f"{split}.tsv")
                        for split in kwargs["split_names"]
                    ]
                    setattr(self, "data_paths", data_paths)
                else:
                    setattr(self, key, value)
            except AttributeError as err:
                logger.error(
                    f"[ERROR] - Unable to set {key} with value {value} for {self}"
                )
                raise err

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy(self.__dict__)


class COLATaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "cola",
                "data_format": "PremiseOnly",
                "encoder_type": "BERT",
                "dropout_p": 0.05,
                "enable_san": False,
                "metric_meta": ["ACC", "MCC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(COLATaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.05)


class MNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "mnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "dropout_p": 0.3,
                "enable_san": True,
                "labels": ["contradiction", "neutral", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 3,
                "split_names": [
                    "train",
                    "matched_dev",
                    "mismatched_dev",
                    "matched_test",
                    "mismatched_test",
                ],
                "task_type": "Classification",
            }
        super(MNLITaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.3)
        self.split_names = kwargs.pop(
            "split_names",
            [
                "train",
                "matched_dev",
                "mismatched_dev",
                "matched_test",
                "mismatched_test",
            ],
        )


class MRPCTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "mrpc",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC", "F1"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(MRPCTaskConfig, self).__init__(**kwargs)


class QNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "qnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["not_entailment", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(QNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class QQPTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "qqp",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC", "F1"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(QQPTaskConfig, self).__init__(**kwargs)


class RTETaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "rte",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["not_entailment", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(RTETaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class SCITAILTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "scitail",
                "encoder_type": "BERT",
                "data_format": "PremiseAndOneHypothesis",
                "enable_san": True,
                "labels": ["neutral", "entails"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(SCITAILTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["neutral", "entails"])


class SNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "snli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["contradiction", "neutral", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 3,
                "task_type": "Classification",
            }
        super(SNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["contradiction", "neutral", "entailment"])


class SSTTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "sst",
                "data_format": "PremiseOnly",
                "encoder_type": "BERT",
                "enable_san": False,
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(SSTTaskConfig, self).__init__(**kwargs)


class STSBTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "stsb",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": False,
                "metric_meta": ["Pearson", "Spearman"],
                "n_class": 1,
                "loss": "MseCriterion",
                "kd_loss": "MseCriterion",
                "task_type": "Regression",
            }
        super(STSBTaskConfig, self).__init__(**kwargs)


class WNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "wnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(WNLITaskConfig, self).__init__(**kwargs)


class NERTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "ner",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.3,
                "enable_san": False,
                "labels": [
                    "O",
                    "B-MISC",
                    "I-MISC",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 12,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(NERTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class POSTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "pos",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "labels": [
                    ",",
                    "\\",
                    ":",
                    ".",
                    "''",
                    '"',
                    "(",
                    ")",
                    "$",
                    "CC",
                    "CD",
                    "DT",
                    "EX",
                    "FW",
                    "IN",
                    "JJ",
                    "JJR",
                    "JJS",
                    "LS",
                    "MD",
                    "NN",
                    "NNP",
                    "NNPS",
                    "NNS",
                    "NN|SYM",
                    "PDT",
                    "POS",
                    "PRP",
                    "PRP$",
                    "RB",
                    "RBR",
                    "RBS",
                    "RP",
                    "SYM",
                    "TO",
                    "UH",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                    "WDT",
                    "WP",
                    "WP$",
                    "WRB",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 49,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(POSTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                ",",
                "\\",
                ":",
                ".",
                "''",
                '"',
                "(",
                ")",
                "$",
                "CC",
                "CD",
                "DT",
                "EX",
                "FW",
                "IN",
                "JJ",
                "JJR",
                "JJS",
                "LS",
                "MD",
                "NN",
                "NNP",
                "NNPS",
                "NNS",
                "NN|SYM",
                "PDT",
                "POS",
                "PRP",
                "PRP$",
                "RB",
                "RBR",
                "RBS",
                "RP",
                "SYM",
                "TO",
                "UH",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "WDT",
                "WP",
                "WP$",
                "WRB",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class CHUNKTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "chunk",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "labels": [
                    "B-ADJP",
                    "B-ADVP",
                    "B-CONJP",
                    "B-INTJ",
                    "B-LST",
                    "B-NP",
                    "B-PP",
                    "B-PRT",
                    "B-SBAR",
                    "B-VP",
                    "I-ADJP",
                    "I-ADVP",
                    "I-CONJP",
                    "I-INTJ",
                    "I-LST",
                    "I-NP",
                    "I-PP",
                    "I-SBAR",
                    "I-VP",
                    "O",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 23,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(CHUNKTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                "B-ADJP",
                "B-ADVP",
                "B-CONJP",
                "B-INTJ",
                "B-LST",
                "B-NP",
                "B-PP",
                "B-PRT",
                "B-SBAR",
                "B-VP",
                "I-ADJP",
                "I-ADVP",
                "I-CONJP",
                "I-INTJ",
                "I-LST",
                "I-NP",
                "I-PP",
                "I-SBAR",
                "I-VP",
                "O",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class SQUADTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "squad",
                "data_format": "MRC",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "metric_meta": ["EmF1"],
                "n_class": 2,
                "task_type": "Span",
                "loss": "SpanCeCriterion",
                "split_names": ["train", "dev"],
            }
        super(SQUADTaskConfig, self).__init__(**kwargs)
        self.split_names = kwargs.pop("split_names", ["train", "dev"])
        self.dropout_p = kwargs.pop("dropout_p", 0.1)


class MaskLMTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "MaskLM",
                "data_format": "MLM",
                "encoder_type": "BERT",
                "enable_san": False,
                "metric_meta": ["ACC"],
                "n_class": 30522,
                "task_type": "MaskLM",
                "loss": "MlmCriterion",
                "split_names": ["train", "dev"],
            }
        super(MaskLMTaskConfig, self).__init__(**kwargs)


# Map of supported tasks
SUPPORTED_TASKS_MAP = {
    "cola": COLATaskConfig,
    "mnli": MNLITaskConfig,
    "mrpc": MRPCTaskConfig,
    "qnli": QNLITaskConfig,
    "qqp": QQPTaskConfig,
    "rte": RTETaskConfig,
    "scitail": SCITAILTaskConfig,
    "snli": SNLITaskConfig,
    "sst": SSTTaskConfig,
    "stsb": STSBTaskConfig,
    "wnli": WNLITaskConfig,
    "ner": NERTaskConfig,
    "pos": POSTaskConfig,
    "chunk": CHUNKTaskConfig,
    "squad": SQUADTaskConfig,
    "squad-v2": SQUADTaskConfig,
    "masklm": MaskLMTaskConfig,
}


class MTDNNTaskConfig:
    supported_tasks_map = SUPPORTED_TASKS_MAP

    def from_dict(self, task_name: str, opts: dict = {}):
        """ Create Task configuration from dictionary of configuration """
        assert opts, "Configuration dictionary cannot be empty"
        task = self.supported_tasks_map[task_name]
        opts.update({"task_name": f"{task_name}"})
        return task(kwargs=opts)

    def get_supported_tasks(self) -> list:
        """Return list of supported tasks

        Returns:
            list -- Supported list of tasks
        """
        return self.supported_tasks_map.keys()


class MTDNNTaskDefs:
    """Definition of single or multiple tasks to train. Can take a single task name or a definition yaml or json file
        
        Arguments:
            task_dict_or_def_file {str or dict} -- Task dictionary or definition file (yaml or json)  
            Example:

            JSON:
            {
                "cola": {
                    "data_format": "PremiseOnly",
                    "encoder_type": "BERT",
                    "dropout_p": 0.05,
                    "enable_san": false,
                    "metric_meta": [
                        "ACC",
                        "MCC"
                    ],
                    "loss": "CeCriterion",
                    "kd_loss": "MseCriterion",
                    "n_class": 2,
                    "split_names": ["train", "test", "dev"],
                    "data_paths": ["CoLA/train.tsv","CoLA/dev.tsv","CoLA/test.tsv"],
                    "data_opts": {
                        "header": True,
                        "is_train": True,
                        "multi_snli": False,
                    },
                    "task_type": "Classification",
                }
                ...
            }
            or 
            
            Python dict:
                { 
                    "cola": {
                        "data_format": "PremiseOnly",
                        "encoder_type": "BERT",
                        "dropout_p": 0.05,
                        "enable_san": False,
                        "metric_meta": [
                            "ACC",
                            "MCC"
                        ],
                        "loss": "CeCriterion",
                        "kd_loss": "MseCriterion",
                        "n_class": 2,
                        "split_names": ["train", "test", "dev"],
                        "data_paths": ["CoLA/train.tsv","CoLA/dev.tsv","CoLA/test.tsv"],
                        "data_opts": {
                            "header": True,
                            "is_train": True,
                            "multi_snli": False,
                        },
                        "task_type": "Classification",
                }
                ...
            }

        """

    def __init__(self, task_dict_or_file: Union[str, dict]):

        assert (
            task_dict_or_file
        ), "Please pass in a task dict or definition file in yaml or json"
        self._task_def_dic = {}
        self._configured_tasks = []  # list of configured tasks
        if isinstance(task_dict_or_file, dict):
            self._task_def_dic = task_dict_or_file
        elif isinstance(task_dict_or_file, str):
            assert os.path.exists(
                task_dict_or_file
            ), "Task definition file does not exist"
            assert os.path.isfile(task_dict_or_file), "Task definition must be a file"

            task_def_filepath, ext = os.path.splitext(task_dict_or_file)
            ext = ext[1:].lower()
            assert ext in [
                "json",
                "yml",
                "yaml",
            ], "Definition file must be in JSON or YAML format"

            self._task_def_dic = (
                yaml.safe_load(open(task_dict_or_file))
                if ext in ["yaml", "yml"]
                else json.load(open(task_dict_or_file))
            )

        global_map = {}
        n_class_map = {}
        data_type_map = {}
        task_type_map = {}
        metric_meta_map = {}
        enable_san_map = {}
        dropout_p_map = {}
        encoderType_map = {}
        loss_map = {}
        kd_loss_map = {}
        data_paths_map = {}
        split_names_map = {}

        # Create an instance of task creator singleton
        task_creator = MTDNNTaskConfig()

        uniq_encoderType = set()
        for name, params in self._task_def_dic.items():
            assert (
                "_" not in name
            ), f"task name should not contain '_', current task name: {name}"

            # Create a singleton to create tasks
            task = task_creator.from_dict(task_name=name, opts=params)

            n_class_map[name] = task.n_class
            data_type_map[name] = DataFormat[task.data_format]
            task_type_map[name] = TaskType[task.task_type]
            metric_meta_map[name] = tuple(
                Metric[metric_name] for metric_name in task.metric_meta
            )
            enable_san_map[name] = task.enable_san
            uniq_encoderType.add(EncoderModelType[task.encoder_type])

            if hasattr(task, "labels"):
                labels = task.labels
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[name] = label_mapper

            # split names
            if hasattr(task, "split_names"):
                split_names_map[name] = task.split_names

            # dropout
            if hasattr(task, "dropout_p"):
                dropout_p_map[name] = task.dropout_p

            # loss map
            if hasattr(task, "loss"):
                t_loss = task.loss
                loss_crt = LossCriterion[t_loss]
                loss_map[name] = loss_crt
            else:
                loss_map[name] = None

            if hasattr(task, "kd_loss"):
                t_loss = task.kd_loss
                loss_crt = LossCriterion[t_loss]
                kd_loss_map[name] = loss_crt
            else:
                kd_loss_map[name] = None

            # Map train, test (and dev) data paths
            data_paths_map[name] = {
                "data_paths": task.data_paths or [],
                "data_opts": task.data_process_opts
                or {"header": True, "is_train": True, "multi_snli": False,},
            }

            # Track configured tasks for downstream
            self._configured_tasks.append(task.to_dict())

        logger.info(
            f"Configured task definitions - {[obj['task_name'] for obj in self.get_configured_tasks()]}"
        )

        assert len(uniq_encoderType) == 1, "The shared encoder has to be the same."
        self.global_map = global_map
        self.n_class_map = n_class_map
        self.data_type_map = data_type_map
        self.task_type_map = task_type_map
        self.metric_meta_map = metric_meta_map
        self.enable_san_map = enable_san_map
        self.dropout_p_map = dropout_p_map
        self.encoderType = uniq_encoderType.pop()
        self.loss_map = loss_map
        self.kd_loss_map = kd_loss_map
        self.data_paths_map = data_paths_map
        self.split_names_map = split_names_map

    def get_configured_tasks(self):
        """Returns a list of configured tasks objects by TaskDefs class from the input configuration file
        
        Returns:
            list -- List of configured task classes
        """
        return self._configured_tasks

    def get_task_names(self):
        """ Returns a list of configured task names
        
        Returns:
            list -- List of configured task classes
        """
        return self.task_type_map.keys()

    def get_task_def(self, task_name: str = ""):
        """Returns a dictionary of parameters for specified task

        Keyword Arguments:
            task_name {str} -- Task name for definition to get (default: {""})

        Returns:
            dict -- Task definition for specified task
        """
        assert task_name in self.task_type_map, "[ERROR] - Task is not configured"
        # return {
        #     k: v
        #     for ele in self.get_configured_tasks()
        #     for k, v in ele.items()
        #     if ele["task_name"] == task_name
        # }
        return TaskDefType(
            self.global_map.get(task_name, None),
            self.n_class_map[task_name],
            self.data_type_map[task_name],
            self.task_type_map[task_name],
            self.metric_meta_map[task_name],
            self.split_names_map[task_name],
            self.enable_san_map[task_name],
            self.dropout_p_map.get(task_name, None),
            self.loss_map[task_name],
            self.kd_loss_map[task_name],
            self.data_paths_map[task_name],
        )


class MTDNNTask:
    def __init__(self, task_def):
        self._task_def = task_def

    def input_parse_label(self, label: str):
        raise NotImplementedError()

    @staticmethod
    def input_is_valid_sample(sample, max_len):
        return len(sample["token_id"]) <= max_len

    @staticmethod
    def train_prepare_label(labels):
        raise NotImplementedError()

    @staticmethod
    def train_prepare_soft_label(softlabels):
        raise NotImplementedError()

    @staticmethod
    def train_build_task_layer(decoder_opt, hidden_size, lab, opt, prefix, dropout):
        if decoder_opt == 1:
            out_proj = SANClassifier(
                hidden_size, hidden_size, lab, opt, prefix, dropout=dropout
            )
        else:
            out_proj = nn.Linear(hidden_size, lab)
        return out_proj

    @staticmethod
    def train_forward(
        sequence_output,
        pooled_output,
        premise_mask,
        hyp_mask,
        decoder_opt,
        dropout_layer,
        task_layer,
    ):
        if decoder_opt == 1:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:, :max_query, :]
            logits = task_layer(sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            pooled_output = dropout_layer(pooled_output)
            logits = task_layer(pooled_output)
        return logits

    @staticmethod
    def test_prepare_label(batch_info, labels):
        batch_info["label"] = labels

    @staticmethod
    def test_predict(score):
        raise NotImplementedError()


def register_task(name):
    """
        @register_task('Classification')
        class ClassificationTask(MTDNNTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~MTDNNTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, MTDNNTask):
            raise ValueError(
                "Task ({}: {}) must extend MTDNNTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


def get_task_obj(task_def):
    task_name = task_def.task_type.name
    task_cls = TASK_REGISTRY.get(task_name, None)
    if task_cls is None:
        return None

    return task_cls(task_def)


@register_task("Regression")
class RegressionTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label: str):
        return float(label)

    @staticmethod
    def train_prepare_label(labels):
        return torch.FloatTensor(labels)

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def test_predict(score):
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict


@register_task("Classification")
class ClassificationTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label: str):
        label_dict = self._task_def.label_vocab
        if label_dict is not None:
            return label_dict[label]
        else:
            return int(label)

    @staticmethod
    def train_prepare_label(labels):
        return torch.LongTensor(labels)

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def test_predict(score):
        score = F.softmax(score, dim=1)
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict
