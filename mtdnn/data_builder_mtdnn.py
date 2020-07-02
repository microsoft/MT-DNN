# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import os
from collections import ChainMap, defaultdict
from typing import Dict, List

from tqdm import tqdm

from mtdnn.common import squad_utils
from mtdnn.common.tokenization_utils import load_task_data
from mtdnn.common.types import DataFormat, EncoderModelType
from mtdnn.common.utils import MTDNNCommonUtils
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tasks.utils import (
    load_cola,
    load_conll_chunk,
    load_conll_ner,
    load_conll_pos,
    load_mnli,
    load_mrpc,
    load_qnli,
    load_qqp,
    load_rte,
    load_scitail,
    load_snli,
    load_sst,
    load_stsb,
    load_wnli,
    process_data_and_dump_rows,
)
from mtdnn.tokenizer_mtdnn import MTDNNTokenizer


logger = MTDNNCommonUtils.create_logger(__name__, to_disk=True)

# Map of supported tasks
GLUE_SUPPORTED_TASKS_LOADER_MAP = {
    "cola": load_cola,
    "mnli": load_mnli,
    "mrpc": load_mrpc,
    "qnli": load_qnli,
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
        self,
        task_defs: MTDNNTaskDefs,
        data_dir: str = "data",
        canonical_data_suffix: str = "canonical_data",
    ):
        self.data_dir = data_dir
        self.task_defs = task_defs
        self.canonical_data_dir = os.path.join(self.data_dir, canonical_data_suffix)
        if not os.path.isdir(self.canonical_data_dir):
            os.makedirs(self.canonical_data_dir)

    def load_and_build_data(self, dump_rows: bool = False) -> dict:
        """
        Load and build out the GLUE and NER Tasks. 
        
        Data Options format in Task Definitions will look like:
            data_opts_map[name] = {
                "data_paths": ['train', 'test', 'dev1', 'dev2']",
                "data_opts": {
                    "header": task.header or True,
                    "is_train": task.is_train or True,
                    "multi_snli": task.multi_snli or False,
                },
            }

        Keyword Arguments:
            dump_rows {bool} -- Dump processed rows to disk after processing (default: {False})

        Raises:
            IOError: IO Error from reading the files from disk

        Returns:
            dict -- Dictionary of processed data rows with key as task name
        """

        datasets_map: dict = self.task_defs.data_paths_map
        processed_data = defaultdict(lambda: [])
        # For each task, load file and build out data

        for name, params in datasets_map.items():

            # TODO - standardize parameters for all loaders to use opts
            data_opts: dict = params.get("data_opts", None)
            assert data_opts, "[ERROR] - Data opts cannot be None"

            # For each task, we process the provided data files into MT-DNN format
            # Format of input is of the form MNLI/{train.tsv, dev_matched.tsv, dev_mismatched.tsv, ...}
            for path in params["data_paths"]:
                in_file_path = os.path.join(self.data_dir, path)
                in_file = os.path.split(path)[-1]
                out_file_name = f"{name}_{in_file}"
                out_file_path = os.path.join(self.canonical_data_dir, out_file_name)

                ####################################################################
                #                  HANDLE SPECIAL CASES                            #
                ####################################################################

                # Set data processing option for test data
                if "test" in in_file:
                    data_opts["is_train"] = False

                # No header for cola train and dev data
                if name == "cola" and ("train" in in_file or "dev" in in_file):
                    data_opts["header"] = False
                try:
                    # Load and dump file
                    task_load_func = self.supported_tasks_loader_map[name]
                    data = task_load_func(in_file_path, data_opts)
                    processed_rows = process_data_and_dump_rows(
                        rows=data,
                        out_path=out_file_path,
                        data_format=self.task_defs.data_type_map[name],
                        dump_rows=dump_rows,
                    )
                    # Format - cola_dev: [processed_rows]
                    processed_data.update(
                        {os.path.splitext(out_file_name)[0]: processed_rows}
                    )
                    logger.info(
                        f"Sucessfully loaded and built {len(data)} samples for {name} at {out_file_path}"
                    )
                except Exception as ex:
                    raise IOError(ex)
        return processed_data


class MTDNNDataBuilder:

    DEBUG_MODE = False
    MAX_SEQ_LEN = 512
    DOC_STRIDE = 180
    MAX_QUERY_LEN = 64
    MRC_MAX_SEQ_LEN = 384

    def __init__(
        self,
        tokenizer: MTDNNTokenizer = None,
        task_defs: MTDNNTaskDefs = None,
        do_lower_case: bool = False,
        data_dir: str = "data",
        canonical_data_suffix: str = "canonical_data",
        dump_rows: bool = False,
    ):
        assert tokenizer, "[ERROR] - MTDNN Tokenizer is required"
        assert task_defs, "[ERROR] - MTDNN Task Definition is required"
        self.tokenizer = tokenizer
        self.task_defs = task_defs
        self.save_to_file = dump_rows
        self.model_name = (
            self.tokenizer.get_model_name()
        )  # ensure model name is same as tokenizer
        self.literal_model_name = self.model_name.split("-")[0]
        self.model_type = EncoderModelType[
            self.literal_model_name.upper()
        ]  # BERT = 1, ROBERTA = 2ll
        mt_dnn_model_name_fmt = self.model_name.replace(
            "-", "_"
        )  # format to mt-dnn format
        self.mt_dnn_suffix = (
            f"{mt_dnn_model_name_fmt}_lower"
            if do_lower_case
            else f"{mt_dnn_model_name_fmt}"
        )
        self.canonical_data_dir: str = f"{data_dir}/{canonical_data_suffix}"
        self.mt_dnn_root = os.path.join(self.canonical_data_dir, self.mt_dnn_suffix)
        if not os.path.isdir(self.mt_dnn_root):
            os.makedirs(self.mt_dnn_root)

        # Load and process data
        self.task_data_loader = MTDNNTaskDataFileLoader(
            self.task_defs, data_dir, canonical_data_suffix,
        )
        self.processed_tasks_data = self.task_data_loader.load_and_build_data(
            self.save_to_file
        )

    def build_data_premise_only(
        self, data: List, max_seq_len: int = 0,
    ):
        """ Build data of single sentence tasks """
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = []
        for idx, sample in tqdm(enumerate(data), desc="Building Data For Premise Only"):
            ids = sample["uid"]
            premise = sample["premise"]
            label = sample["label"]
            if len(premise) > self.MAX_SEQ_LEN - 2:
                premise = premise[: self.MAX_SEQ_LEN - 2]
            input_ids, input_mask, type_ids = self.tokenizer.encode(
                text=premise, max_length=self.MAX_SEQ_LEN,
            )
            features = {
                "uid": ids,
                "label": label,
                "token_id": input_ids,
                "type_id": type_ids,
            }
            rows.append(features)
        return rows

    def build_data_premise_and_one_hypo(
        self, data: List, max_seq_len: int = 0,
    ):
        """ Build data of sentence pair tasks """
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = []
        for idx, sample in tqdm(
            enumerate(data), desc="Building Data For Premise and One Hypothesis"
        ):
            ids = sample["uid"]
            premise = sample["premise"]
            hypothesis = sample["hypothesis"]
            label = sample["label"]
            input_ids, input_mask, type_ids = self.tokenizer.encode(
                text=premise, text_pair=hypothesis, max_length=max_seq_len,
            )
            features = {
                "uid": ids,
                "label": label,
                "token_id": input_ids,
                "type_id": type_ids,
            }
            rows.append(features)
        return rows

    def build_data_premise_and_multi_hypo(
        self, data: List, max_seq_len: int = 0,
    ):
        """ Build QNLI as a pair-wise ranking task """
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = []
        for idx, sample in tqdm(
            enumerate(data), desc="Building Data For Premise and Multi Hypothesis"
        ):
            ids = sample["uid"]
            premise = sample["premise"]
            hypothesis_list = sample["hypothesis"]
            label = sample["label"]
            input_ids_list = []
            type_ids_list = []
            for hypothesis in hypothesis_list:
                input_ids, mask, type_ids = self.tokenizer.encode(
                    text=premise, text_pair=hypothesis, max_length=max_seq_len,
                )
                input_ids_list.append(input_ids)
                type_ids_list.append(type_ids)
            features = {
                "uid": ids,
                "label": label,
                "token_id": input_ids_list,
                "type_id": type_ids_list,
                "ruid": sample["ruid"],
                "olabel": sample["olabel"],
            }
            rows.append(features)
        return rows

    def build_data_sequence(
        self, data: List, max_seq_len: int = 0, label_mapper: Dict = None,
    ):
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = []
        for idx, sample in tqdm(enumerate(data), desc="Building Data For Sequence"):
            ids = sample["uid"]
            premise = sample["premise"]
            tokens = []
            labels = []
            for i, word in tqdm(enumerate(premise), desc="Building Sequence Premise"):
                subwords = tokenizer.tokenize(word)
                tokens.extend(subwords)
                for j in range(len(subwords)):
                    if j == 0:
                        labels.append(sample["label"][i])
                    else:
                        labels.append(label_mapper["X"])
            if len(premise) > max_seq_len - 2:
                tokens = tokens[: max_seq_len - 2]
                labels = labels[: max_seq_len - 2]

            label = [label_mapper["CLS"]] + labels + [label_mapper["SEP"]]
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            )
            assert len(label) == len(input_ids)
            type_ids = [0] * len(input_ids)
            features = {
                "uid": ids,
                "label": label,
                "token_id": input_ids,
                "type_id": type_ids,
            }
            rows.append(features)
        return rows

    def build_data_mrc(
        self,
        data: List,
        max_seq_len: int = 0,
        label_mapper: Dict = None,
        is_training: bool = True,
    ):
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = []
        unique_id = 1000000000  # TODO: this is from BERT, needed to remove it...
        for example_index, sample in tqdm(
            enumerate(data), desc="Building Data For MRC"
        ):
            ids = sample["uid"]
            doc = sample["premise"]
            query = sample["hypothesis"]
            label = sample["label"]
            doc_tokens, cw_map = squad_utils.token_doc(doc)
            (
                answer_start,
                answer_end,
                answer,
                is_impossible,
            ) = squad_utils.parse_squad_label(label)
            answer_start_adjusted, answer_end_adjusted = squad_utils.recompute_span(
                answer, answer_start, cw_map
            )
            is_valid = squad_utils.is_valid_answer(
                doc_tokens, answer_start_adjusted, answer_end_adjusted, answer
            )
            if not is_valid:
                continue
            """
            TODO --xiaodl: support RoBERTa
            """
            feature_list = squad_utils.mrc_feature(
                self.tokenizer,
                unique_id,
                example_index,
                query,
                doc_tokens,
                answer_start_adjusted,
                answer_end_adjusted,
                is_impossible,
                max_seq_len,
                self.MAX_QUERY_LEN,
                self.DOC_STRIDE,
                answer_text=answer,
                is_training=True,
            )
            unique_id += len(feature_list)
            for feature in feature_list:
                feature_obj = {
                    "uid": ids,
                    "token_id": feature.input_ids,
                    "mask": feature.input_mask,
                    "type_id": feature.segment_ids,
                    "example_index": feature.example_index,
                    "doc_span_index": feature.doc_span_index,
                    "tokens": feature.tokens,
                    "token_to_orig_map": feature.token_to_orig_map,
                    "token_is_max_context": feature.token_is_max_context,
                    "start_position": feature.start_position,
                    "end_position": feature.end_position,
                    "label": feature.is_impossible,
                    "doc": doc,
                    "doc_offset": feature.doc_offset,
                    "answer": [answer],
                }
                rows.append(feature_obj)
        return rows

    def _build_data_from_format(
        self,
        data: List,
        dump_path: str = "",
        data_format: int = DataFormat.Init,
        max_seq_len: int = 0,
        label_mapper: Dict = None,
        dump_rows: bool = False,
    ):
        max_seq_len = max_seq_len if max_seq_len else self.MAX_SEQ_LEN
        rows = None

        # Process the data depending on the data format set from the config
        if data_format == DataFormat.PremiseOnly:
            rows = self.build_data_premise_only(data, max_seq_len)
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            rows = self.build_data_premise_and_one_hypo(data, max_seq_len)
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            rows = self.build_data_premise_and_multi_hypo(data, max_seq_len)
        elif data_format == DataFormat.Sequence:
            rows = self.build_data_sequence(data, max_seq_len, label_mapper)
        elif data_format == DataFormat.MRC:
            rows = self.build_data_mrc(data, max_seq_len)
        else:
            raise ValueError(data_format)

        # Save file to disk
        if self.save_to_file:
            with open(dump_path, "w", encoding="utf-8") as writer:
                logger.info(f"Saving data to {dump_path}")
                for row in tqdm(rows, desc=f"Saving Data For {data_format.name}"):
                    writer.write(f"{json.dumps(row)}\n")
        return rows

    def vectorize(self):
        """ Tokenize and build data for the tasks """
        mtdnn_featurized_data = {}
        for task_split_name, task_data in self.processed_tasks_data.items():
            print(task_split_name)
            split_name = task_split_name.split("_")
            task = split_name[0]
            task_def = self.task_defs.get_task_def(task)
            dump_path = f"{os.path.join(self.mt_dnn_root, task_split_name)}.json"
            logger.info(f"Building Data For '{' '.join(split_name).upper()}' Task")
            loaded_data = load_task_data(task_data, task_def)
            rows = self._build_data_from_format(
                data=loaded_data,
                dump_path=dump_path,
                data_format=task_def.data_format,
                label_mapper=task_def.label_vocab,
                dump_rows=self.save_to_file,
            )
            mtdnn_featurized_data.update({task_split_name: rows})
        return mtdnn_featurized_data
