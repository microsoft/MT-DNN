# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import json
from typing import Union

import numpy as np

from mtdnn.common.types import DataFormat, TaskType, TaskDefType
from mtdnn.tasks.config import get_task_obj


def load_task_data(
    file_path_or_processed_data_list: Union[str, list], task_def: TaskDefType
):
    """Load data in MT-DNN Format

    Arguments:
        file_path_or_processed_data_list {Union[str, list]} -- File path or processed rows object
        task_def {dict} -- Task Definition to be loaded

    Raises:
        ValueError: Invalid Task requested

    Returns:
        list -- list of processed data in MT-DNN Format
    """
    assert task_def, "[ERROR] - Task Definition cannot be none"
    data_format = task_def.data_format
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis
    if isinstance(file_path_or_processed_data_list, str):
        processed_data = open(file_path_or_processed_data_list, encoding="utf-8")
    elif isinstance(file_path_or_processed_data_list, list):
        processed_data = file_path_or_processed_data_list

    rows = []
    for line in processed_data:
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
                "hypothesis": fields[3],
            }
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {
                "uid": fields[0],
                "ruid": fields[1].split(","),
                "label": fields[2],
                "premise": fields[3],
                "hypothesis": fields[4:],
            }
        elif data_format == DataFormat.Sequence:
            row = {
                "uid": fields[0],
                "label": eval(fields[1]),
                "premise": eval(fields[2]),
            }

        elif data_format == DataFormat.MRC:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3],
            }
        else:
            raise ValueError(data_format)

        task_obj = get_task_obj(task_def)
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
        elif task_type == TaskType.SequenceLabeling:
            assert type(row["label"]) is list
            row["label"] = [label_dict[label] for label in row["label"]]

        rows.append(row)
    return rows


def load_score_file(score_path: str = "", n_class: int = 1):
    sample_id_2_pred_score_seg_dic = {}
    score_obj = json.loads(open(score_path, encoding="utf-8").read())
    assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and (
        len(score_obj["scores"]) / len(score_obj["uids"]) == n_class
    ), "[ERROR] - scores column size should equal to sample count or multiple of sample count (for classification problem)"

    scores = score_obj["scores"]
    score_segs = [
        scores[i * n_class : (i + 1) * n_class] for i in range(len(score_obj["uids"]))
    ]
    for sample_id, pred, score_seg in zip(
        score_obj["uids"], score_obj["predictions"], score_segs
    ):
        sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
    return sample_id_2_pred_score_seg_dic
