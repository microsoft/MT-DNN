# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import pdb
from random import shuffle
from sys import path

from mtdnn.common.metrics import calc_metrics
from mtdnn.common.types import DataFormat


def process_data_and_dump_rows(
    rows: list,
    out_path: str,
    data_format: DataFormat,
    write_mode: str = "w",
    dump_rows: bool = False,
) -> None:
    """
        Output files should have following format
        :param rows: data
        :param out_path: output file path
        :return: processed_rows: List of string rows 
    """
    processed_rows = []
    for row in rows:
        data = ""
        if data_format in [DataFormat.PremiseOnly, DataFormat.Sequence]:
            for col in ["uid", "label", "premise"]:
                if "\t" in str(row[col]):
                    pdb.set_trace()
            data = f"{row['uid']}\t{row['label']}\t{row['premise']}\n"
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            for col in ["uid", "label", "premise", "hypothesis"]:
                if "\t" in str(row[col]):
                    pdb.set_trace()
            data = (
                f"{row['uid']}\t{row['label']}\t{row['premise']}\t{row['hypothesis']}\n"
            )
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            for col in ["uid", "label", "premise"]:
                if "\t" in str(row[col]):
                    pdb.set_trace()
            hypothesis = row["hypothesis"]
            for one_hypo in hypothesis:
                if "\t" in str(one_hypo):
                    pdb.set_trace()
            hypothesis = "\t".join(hypothesis)
            data = f"{row['uid']}\t{row['ruid']}\t{row['label']}\t{row['premise']}\t{hypothesis}\n"
        else:
            raise ValueError(data_format)
        processed_rows.append(data)

    # Save data if dump_rows is true
    if dump_rows:
        with open(out_path, mode=write_mode, encoding="utf-8") as out_f:
            out_f.writelines(processed_rows)
    return processed_rows


def load_scitail(file_path, kwargs: dict = {}):
    """ Loading scitail """

    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            blocks = line.strip().split("\t")
            assert len(blocks) > 2
            if blocks[0] == "-":
                continue
            sample = {
                "uid": str(cnt),
                "premise": blocks[0],
                "hypothesis": blocks[1],
                "label": blocks[2],
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_snli(file_path, kwargs: dict = {}):
    """ Load SNLI """
    header = kwargs.get("header", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 10
            if blocks[-1] == "-":
                continue
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[7],
                "hypothesis": blocks[8],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_mnli(file_path, kwargs: dict = {}):
    """ Load MNLI """
    header = kwargs.get("header", True)
    multi_snli = kwargs.get("multi_snli", False)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 9
            if blocks[-1] == "-":
                continue
            lab = "contradiction"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[8],
                "hypothesis": blocks[9],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_mrpc(file_path, kwargs: dict = {}):
    """ Load MRPC """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 4
            lab = 0
            if is_train:
                lab = int(blocks[0])
            sample = {
                "uid": cnt,
                "premise": blocks[-2],
                "hypothesis": blocks[-1],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_qnli(file_path, kwargs: dict = {}):
    """ Load QNLI for classification"""

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 2
            lab = "not_entailment"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[1],
                "hypothesis": blocks[2],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


def load_qqp(file_path, kwargs: dict = {}):
    """ Load QQP """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    skipped = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if is_train and len(blocks) < 6:
                skipped += 1
                continue
            if not is_train:
                assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {
                    "uid": cnt,
                    "premise": blocks[-3],
                    "hypothesis": blocks[-2],
                    "label": lab,
                }
            else:
                sample = {
                    "uid": int(blocks[0]),
                    "premise": blocks[-2],
                    "hypothesis": blocks[-1],
                    "label": lab,
                }
            rows.append(sample)
            cnt += 1
    return rows


def load_rte(file_path, kwargs: dict = {}):
    """ Load RTE """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if is_train and len(blocks) < 4:
                continue
            if not is_train:
                assert len(blocks) == 3
            lab = "not_entailment"
            if is_train:
                lab = blocks[-1]
                sample = {
                    "uid": int(blocks[0]),
                    "premise": blocks[-3],
                    "hypothesis": blocks[-2],
                    "label": lab,
                }
            else:
                sample = {
                    "uid": int(blocks[0]),
                    "premise": blocks[-2],
                    "hypothesis": blocks[-1],
                    "label": lab,
                }
            rows.append(sample)
            cnt += 1
    return rows


def load_wnli(file_path, kwargs: dict = {}):
    """ Load WNLI """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if is_train and len(blocks) < 4:
                continue
            if not is_train:
                assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {
                    "uid": cnt,
                    "premise": blocks[-3],
                    "hypothesis": blocks[-2],
                    "label": lab,
                }
            else:
                sample = {
                    "uid": cnt,
                    "premise": blocks[-2],
                    "hypothesis": blocks[-1],
                    "label": lab,
                }
            rows.append(sample)
            cnt += 1
    return rows


def load_sst(file_path, kwargs: dict = {}):
    """ Load SST """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if is_train and len(blocks) < 2:
                continue
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {"uid": cnt, "premise": blocks[0], "label": lab}
            else:
                sample = {"uid": int(blocks[0]), "premise": blocks[1], "label": lab}

            cnt += 1
            rows.append(sample)
    return rows


def load_cola(file_path, kwargs: dict = {}):
    """ Load COLA """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)

    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            if is_train and len(blocks) < 2:
                continue
            lab = 0
            if is_train:
                lab = int(blocks[1])
                sample = {"uid": cnt, "premise": blocks[-1], "label": lab}
            else:
                sample = {"uid": cnt, "premise": blocks[-1], "label": lab}
            rows.append(sample)
            cnt += 1
    return rows


def load_stsb(file_path, kwargs: dict = {}):
    """ Load STSB """

    header = kwargs.get("header", True)
    is_train = kwargs.get("is_train", True)
    rows = []
    cnt = 0
    with open(file_path, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 8
            score = "0.0"
            if is_train:
                score = blocks[-1]
                sample = {
                    "uid": cnt,
                    "premise": blocks[-3],
                    "hypothesis": blocks[-2],
                    "label": score,
                }
            else:
                sample = {
                    "uid": cnt,
                    "premise": blocks[-2],
                    "hypothesis": blocks[-1],
                    "label": score,
                }
            rows.append(sample)
            cnt += 1
    return rows


def load_conll_ner(file_path, kwargs: dict = {}):
    """ Load NER """

    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows


def load_conll_pos(file_path, kwargs: dict = {}):
    """ Load POS """

    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[1])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows


def load_conll_chunk(file_path, kwargs: dict = {}):
    """ Load CHUNK """

    rows = []
    cnt = 0
    sentence = []
    label = []
    with open(file_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                if len(sentence) > 0:
                    sample = {"uid": cnt, "premise": sentence, "label": label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(" ")
            sentence.append(splits[0])
            label.append(splits[2])
        if len(sentence) > 0:
            sample = {"uid": cnt, "premise": sentence, "label": label}
    return rows


def submit(path, data, label_dict=None):
    header = "index\tprediction"
    with open(path, "w") as writer:
        predictions, uids = data["predictions"], data["uids"]
        writer.write("{}\n".format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write("{}\t{}\n".format(uid, pred))
            else:
                assert type(pred) is int
                writer.write("{}\t{}\n".format(uid, label_dict[pred]))
