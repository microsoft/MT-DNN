# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
from collections import ChainMap

from mtdnn.common.types import DataFormat, EncoderModelType
from mtdnn.common.utils import MTDNNCommonUtils
from mtdnn.modeling_mtdnn import MODEL_CLASSES
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tasks.utils import (dump_processed_rows, load_cola,
                               load_conll_chunk, load_conll_ner,
                               load_conll_pos, load_mnli, load_mrpc, load_qnli,
                               load_qnnli, load_qqp, load_rte, load_scitail,
                               load_snli, load_sst, load_stsb, load_wnli)

logger = MTDNNCommonUtils.setup_logging(filename="preprocessor.log")

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
        canonical_data_suffix: str = "canonical",
    ):
        self.data_dir = data_dir
        self.task_defs = task_defs
        self.canonical_data_dir = os.path.join(self.data_dir, canonical_data_suffix)
        if not os.path.isdir(self.canonical_data_dir):
            os.mkdir(self.canonical_data_dir)

    def load_and_build_data(self):

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
        """
        datasets_map: dict = self.task_defs.data_paths_map

        # For each task, load file and build out data
        for name, params in datasets_map.items():

            # TODO - standardize parameters for all loaders to use opts
            data_opts = params.pop("data_opts")

            # For each task, we process the provided data files into MT-DNN format
            # Format of input is of the form MNLI/{train.tsv, dev_matched.tsv, dev_mismatched.tsv, ...}
            for path in params["data_paths"]:
                in_file_path = os.path.join(self.data_dir, path)
                in_file = os.path.split(path)[-1]
                out_file_name = f"{name}_{in_file}"
                out_file_path = os.path.join(self.canonical_data_dir, out_file_name)

                try:
                    # Load and dump file
                    data = self.supported_tasks_loader_map[name](in_file_path, **kwargs)
                    dump_processed_rows(
                        data, out_file_path, self.task_defs.data_type_map[name]
                    )
                    logger.info(
                        f"Sucessfully loaded and built {len(data)} samples for {name} at {out_file_path}"
                    )
                except expression as ex:
                    raise IOError(ex)


class MTDNNTokenizer:
    "Preprocessing GLUE/SNLI/SciTail datasets."
    DEBUG_MODE = False
    MAX_SEQ_LEN = 512
    DOC_STRIDE = 180
    MAX_QUERY_LEN = 64
    MRC_MAX_SEQ_LEN = 384

    def __init__(self, model_name: str = 'bert-base-uncased', do_lower_case: bool = False, task_defs: MTDNNTaskDefs, canonical_data_dir: str = "data/canonical_data"):
        self.literal_model_name = model_name.split('-')[0]
        self.model_type = EncoderModelType[self.literal_model_name.upper()] # BERT = 1, ROBERTA = 2
        mt_dnn_model_name_fmt = model_name.replace('-', '_') # format to mt-dnn format
        self.mt_dnn_suffix = f"{mt_dnn_model_name_fmt}_lower" if do_lower_case else f"{mt_dnn_model_name_fmt}"
        self.task_defs = task_defs
        self.canonical_data_dir = canonical_data_dir
        self.config, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.literal_model_name]
        self.tokenizer = self.tokenizer_class.from_pretrained(model_name, do_lower_case=do_lower_case)

    
    def feature_extractor(
        self,
        text_a: str,
        text_b: str = '',
        max_length=512,
        model_type=None,
        enable_padding=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=False,
    ):  
    # set mask_padding_with_zero default value as False to keep consistent with original setting
        inputs = self.tokenizer.encode_plus(
            text_a, text_b, add_special_tokens=True, max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        if enable_padding:
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
                len(input_ids), max_length
            )
            assert (
                len(attention_mask) == max_length
            ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert (
                len(token_type_ids) == max_length
            ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if model_type.lower() in ["bert", "roberta"]:
            attention_mask = None

        if model_type.lower() not in ["distilbert", "bert", "xlnet"]:
            token_type_ids = [0] * len(token_type_ids)

        return (
            input_ids,
            attention_mask,
            token_type_ids,
        )  # input_ids, input_mask, segment_id

    def load_data(self, file_path, task_def):
        data_format = task_def.data_type
        task_type = task_def.task_type
        label_dict = task_def.label_vocab
        if task_type == TaskType.Ranking:
            assert data_format == DataFormat.PremiseAndMultiHypothesis

        rows = []
        for line in open(file_path, encoding="utf-8"):
            fields = line.strip("\n").split("\t")
            if data_format == DataFormat.PremiseOnly:
                assert len(fields) == 3
                row = {"uid": fields[0], "label": fields[1], "premise": fields[2]}
            elif data_format == DataFormat.PremiseAndOneHypothesis:
                assert len(fields) == 4
                row = {
                    "uid": fields[0],
                    "label": fields[1],
                    "premise": fields[2],
                    "hypothesis": fields[3]}
            elif data_format == DataFormat.PremiseAndMultiHypothesis:
                assert len(fields) > 5
                row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                        "hypothesis": fields[4:]}
            elif data_format == DataFormat.Seqence:
                row = {"uid": fields[0], "label": eval(fields[1]),  "premise": eval(fields[2])}

            elif data_format == DataFormat.MRC:
                row = {
                    "uid": fields[0],
                    "label": fields[1],
                    "premise": fields[2],
                    "hypothesis": fields[3]}
            else:
                raise ValueError(data_format)

            task_obj = tasks.get_task_obj(task_def)
            if task_obj is not None:
                row["label"] = task_obj.input_parse_label(row["label"])
            elif task_type == TaskType.Ranking:
                labels = row["label"].split(",")
                if label_dict is not None:
                    labels = [label_dict[label] for label in labels]
                else:
                    labels = [float(label) for label in labels]
                row["label"] = int(np.argmax(labels))
                row["olabel"] = labels
            elif task_type == TaskType.Span:
                pass  # don't process row label
            elif task_type == TaskType.SeqenceLabeling:
                assert type(row["label"]) is list
                row["label"] = [label_dict[label] for label in row["label"]]

            rows.append(row)
        return rows

    def load_score_file(self, score_path, n_class):
        sample_id_2_pred_score_seg_dic = {}
        score_obj = json.loads(open(score_path, encoding="utf-8").read())
        assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and \
            (len(score_obj["scores"]) / len(score_obj["uids"]) == n_class), \
            "scores column size should equal to sample count or multiple of sample count (for classification problem)"

        scores = score_obj["scores"]
        score_segs = [scores[i * n_class: (i+1) * n_class] for i in range(len(score_obj["uids"]))]
        for sample_id, pred, score_seg in zip(score_obj["uids"], score_obj["predictions"], score_segs):
            sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
        return sample_id_2_pred_score_seg_dic

    def build_data_premise_only(self,
        data,
        dump_path,
        max_seq_len=MAX_SEQ_LEN,
        tokenizer=None,
        encoderModelType=EncoderModelType.BERT,
    ):
        """Build data of single sentence tasks
        """
        with open(dump_path, "w", encoding="utf-8") as writer:
            for idx, sample in enumerate(data):
                ids = sample["uid"]
                premise = sample["premise"]
                label = sample["label"]
                if len(premise) > max_seq_len - 2:
                    premise = premise[: max_seq_len - 2]
                input_ids, input_mask, type_ids = feature_extractor(
                    tokenizer,
                    premise,
                    max_length=max_seq_len,
                    model_type=encoderModelType.name,
                )
                features = {
                    "uid": ids,
                    "label": label,
                    "token_id": input_ids,
                    "type_id": type_ids,
                }
                writer.write("{}\n".format(json.dumps(features)))

    def build_data_premise_and_one_hypo(self,
        data,
        dump_path,
        max_seq_len=MAX_SEQ_LEN,
        tokenizer=None,
        encoderModelType=EncoderModelType.BERT,
    ):
        """Build data of sentence pair tasks
        """
        with open(dump_path, "w", encoding="utf-8") as writer:
            for idx, sample in enumerate(data):
                ids = sample["uid"]
                premise = sample["premise"]
                hypothesis = sample["hypothesis"]
                label = sample["label"]
                input_ids, input_mask, type_ids = feature_extractor(
                    tokenizer,
                    premise,
                    text_b=hypothesis,
                    max_length=max_seq_len,
                    model_type=encoderModelType.name,
                )
                features = {
                    "uid": ids,
                    "label": label,
                    "token_id": input_ids,
                    "type_id": type_ids,
                }
                writer.write("{}\n".format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(self,
        data,
        dump_path,
        max_seq_len=MAX_SEQ_LEN,
        tokenizer=None,
        encoderModelType=EncoderModelType.BERT,
    ):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, "w", encoding="utf-8") as writer:
            for idx, sample in enumerate(data):
                ids = sample["uid"]
                premise = sample["premise"]
                hypothesis_list = sample["hypothesis"]
                label = sample["label"]
                input_ids_list = []
                type_ids_list = []
                for hypothesis in hypothesis_list:
                    input_ids, mask, type_ids = feature_extractor(
                        tokenizer,
                        premise,
                        hypothesis,
                        max_length=max_seq_len,
                        model_type=encoderModelType.name,
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
                writer.write("{}\n".format(json.dumps(features)))

    def build_data_sequence(self,
        data,
        dump_path,
        max_seq_len=MAX_SEQ_LEN,
        tokenizer=None,
        encoderModelType=EncoderModelType.BERT,
        label_mapper=None,
    ):
        with open(dump_path, "w", encoding="utf-8") as writer:
            for idx, sample in enumerate(data):
                ids = sample["uid"]
                premise = sample["premise"]
                tokens = []
                labels = []
                for i, word in enumerate(premise):
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
                writer.write("{}\n".format(json.dumps(features)))

    def build_data_mrc(
        self,
        data,
        dump_path,
        max_seq_len=MRC_MAX_SEQ_LEN,
        tokenizer=None,
        label_mapper=None,
        is_training=True,
    ):
        with open(dump_path, "w", encoding="utf-8") as writer:
            unique_id = 1000000000  # TODO: this is from BERT, needed to remove it...
            for example_index, sample in enumerate(data):
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
                    tokenizer,
                    unique_id,
                    example_index,
                    query,
                    doc_tokens,
                    answer_start_adjusted,
                    answer_end_adjusted,
                    is_impossible,
                    max_seq_len,
                    MAX_QUERY_LEN,
                    DOC_STRIDE,
                    answer_text=answer,
                    is_training=True,
                )
                unique_id += len(feature_list)
                for feature in feature_list:
                    so = json.dumps(
                        {
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
                    )
                    writer.write("{}\n".format(so))