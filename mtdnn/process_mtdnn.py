# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import logging
import os
import random
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import BatchSampler, DataLoader, Dataset

from mtdnn.common.types import TaskType
from mtdnn.common.utils import MTDNNCommonUtils
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.dataset_mtdnn import (
    MTDNNCollater,
    MTDNNMultiTaskBatchSampler,
    MTDNNMultiTaskDataset,
    MTDNNSingleTaskDataset,
)
from mtdnn.modeling_mtdnn import MTDNNModel
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tasks.utils import submit

logger = MTDNNCommonUtils.create_logger(__name__, to_disk=True)


class MTDNNDataProcess:
    def __init__(
        self,
        config: MTDNNConfig,
        task_defs: MTDNNTaskDefs,
        vectorized_data: dict,
        glue_format: bool = False,
        data_sort: bool = False,
    ):
        assert vectorized_data, "[ERROR] - Vectorized data cannot be None"

        # Initialize class members
        self.config = config
        self.task_defs = task_defs
        self.train_datasets_list = list(
            filter(lambda file_name: "train" in file_name, vectorized_data.keys())
        )
        self.test_dev_datasets_list = list(
            filter(
                lambda file_name: "dev" in file_name or "test" in file_name,
                vectorized_data.keys(),
            )
        )
        self.vectorized_data = vectorized_data
        self.glue_format = glue_format
        self.data_sort = data_sort
        self.tasks = {}
        self.tasks_class = {}
        self.nclass_list = []
        self.decoder_opts = []
        self.task_types = []
        self.dropout_list = []
        self.loss_types = []
        self.kd_loss_types = []
        self._multitask_train_dataloader = self._process_train_datasets()
        (
            self._dev_dataloaders_list,
            self._test_dataloaders_list,
        ) = self._process_dev_test_datasets()
        self._num_all_batches = (
            self.config.epochs
            * len(self._multitask_train_dataloader)
            // self.config.grad_accumulation_step
        )

    def _process_train_datasets(self):
        """Preprocess the training sets and generate decoding and task specific training options needed to update config object
        
        Returns:
            [DataLoader] -- Multiple tasks train data ready for training
        """
        logger.info("Starting to process the training data sets")

        train_datasets = []
        for dataset in self.train_datasets_list:
            prefix = dataset.split("_")[0]
            if prefix in self.tasks:
                continue
            assert (
                prefix in self.task_defs.n_class_map
            ), f"[ERROR] - {prefix} does not exist in {self.task_defs.n_class_map}"
            assert (
                prefix in self.task_defs.data_type_map
            ), f"[ERROR] - {prefix} does not exist in {self.task_defs.data_type_map}"
            data_type = self.task_defs.data_type_map[prefix]
            nclass = self.task_defs.n_class_map[prefix]
            task_id = len(self.tasks)
            if self.config.mtl_opt > 0:
                task_id = (
                    self.tasks_class[nclass]
                    if nclass in self.tasks_class
                    else len(self.tasks_class)
                )

            task_type = self.task_defs.task_type_map[prefix]

            dopt = self.generate_decoder_opt(
                self.task_defs.enable_san_map[prefix], self.config.answer_opt
            )
            if task_id < len(self.decoder_opts):
                self.decoder_opts[task_id] = min(self.decoder_opts[task_id], dopt)
            else:
                self.decoder_opts.append(dopt)
            self.task_types.append(task_type)
            self.loss_types.append(self.task_defs.loss_map[prefix])
            self.kd_loss_types.append(self.task_defs.kd_loss_map[prefix])

            if prefix not in self.tasks:
                self.tasks[prefix] = len(self.tasks)
                if self.config.mtl_opt < 1:
                    self.nclass_list.append(nclass)

            if nclass not in self.tasks_class:
                self.tasks_class[nclass] = len(self.tasks_class)
                if self.config.mtl_opt > 0:
                    self.nclass_list.append(nclass)

            dropout_p = self.task_defs.dropout_p_map.get(prefix, self.config.dropout_p)
            self.dropout_list.append(dropout_p)

            train_data = self.vectorized_data[dataset]
            assert (
                train_data
            ), f"[ERROR] - Training dataset for {dataset} does not exist."

            logger.info(f"Loading {dataset} as task {task_id}")
            train_data_set = MTDNNSingleTaskDataset(
                train_data,
                True,
                maxlen=self.config.max_seq_len,
                task_id=task_id,
                task_type=task_type,
                data_type=data_type,
            )
            train_datasets.append(train_data_set)
        train_collater = MTDNNCollater(
            dropout_w=self.config.dropout_w, encoder_type=self.config.encoder_type
        )
        multitask_train_dataset = MTDNNMultiTaskDataset(train_datasets)
        multitask_batch_sampler = MTDNNMultiTaskBatchSampler(
            train_datasets,
            self.config.batch_size,
            self.config.mix_opt,
            self.config.ratio,
        )
        multitask_train_data = DataLoader(
            multitask_train_dataset,
            batch_sampler=multitask_batch_sampler,
            collate_fn=train_collater.collate_fn,
            pin_memory=self.config.cuda,
        )
        return multitask_train_data

    def _process_dev_test_datasets(self):
        """Preprocess the test sets 
        
        Returns:
            [List] -- Multiple tasks test data ready for inference
        """
        logger.info("Starting to process the testing data sets")
        dev_dataloaders_list = []
        test_dataloaders_list = []
        test_collater = MTDNNCollater(
            is_train=False, encoder_type=self.config.encoder_type
        )
        for dataset in self.test_dev_datasets_list:
            prefix = dataset.split("_")[0]
            task_id = (
                self.tasks_class[self.task_defs.n_class_map[prefix]]
                if self.config.mtl_opt > 0
                else self.tasks[prefix]
            )
            task_type = self.task_defs.task_type_map[prefix]

            pw_task = False
            if task_type == TaskType.Ranking:
                pw_task = True

            assert prefix in self.task_defs.data_type_map
            data_type = self.task_defs.data_type_map[prefix]

            # Process datasets for the dev and tests
            data_rows = self.vectorized_data[dataset]
            assert (
                data_rows
            ), f"[ERROR] - Test/Dev dataset for {dataset} does not exist."
            logger.info(f"Loading {dataset} as task {task_id}")
            data_set = MTDNNSingleTaskDataset(
                data_rows,
                False,
                maxlen=self.config.max_seq_len,
                task_id=task_id,
                task_type=task_type,
                data_type=data_type,
            )
            data_loader = DataLoader(
                data_set,
                batch_size=self.config.batch_size_eval,
                collate_fn=test_collater.collate_fn,
                pin_memory=self.config.cuda,
            )

            dev_dataloaders_list.append(
                data_loader
            ) if "dev" in dataset else test_dataloaders_list.append(data_loader)

        # Return tuple of dev and test dataloaders
        return dev_dataloaders_list, test_dataloaders_list

    def get_train_dataloader(self) -> DataLoader:
        """Returns a dataloader for mutliple tasks
        
        Returns:
            DataLoader -- Multiple tasks batch dataloader
        """
        return self._multitask_train_dataloader

    def get_dev_dataloaders(self) -> list:
        """Returns a list of dev dataloaders for multiple tasks
        
        Returns:
            list -- List of dev dataloaders
        """
        return self._dev_dataloaders_list

    def get_test_dataloaders(self) -> list:
        """Returns a list of test dataloaders for multiple tasks
        
        Returns:
            list -- List of test dataloaders
        """
        return self._test_dataloaders_list

    def generate_decoder_opt(self, enable_san, max_opt):
        return max_opt if enable_san and max_opt < 3 else 0

    # Getters for Model training configuration
    def get_decoder_options_list(self) -> list:
        return self.decoder_opts

    def get_task_types_list(self) -> list:
        return self.task_types

    def get_tasks_dropout_prob_list(self) -> list:
        return self.dropout_list

    def get_loss_types_list(self) -> list:
        return self.loss_types

    def get_kd_loss_types_list(self) -> list:
        return self.kd_loss_types

    def get_task_nclass_list(self) -> list:
        return self.nclass_list

    def get_num_all_batches(self) -> int:
        return self._num_all_batches
