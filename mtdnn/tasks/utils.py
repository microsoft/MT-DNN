# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import pdb

from mtdnn.common.types import DataFormat


def dump_processed_rows(
    rows: list, out_path: str, data_format: DataFormat, write_mode: str = "w"
):
    """
    output files should have following format
    :param rows:
    :param out_path:
    :return:
    """
    with open(out_path, mode=write_mode, encoding="utf-8") as out_f:
        for row in rows:
            if data_format in [DataFormat.PremiseOnly, DataFormat.Sequence]:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        pdb.set_trace()
                out_f.write(f"{row['uid']}\t{row['label']}\t{row['premise']}\n")
            elif data_format == DataFormat.PremiseAndOneHypothesis:
                for col in ["uid", "label", "premise", "hypothesis"]:
                    if "\t" in str(row[col]):
                        pdb.set_trace()
                out_f.write(
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
                out_f.write(
                    f"{row['uid']}\t{row['ruid']}\t{row['label']}\t{row['premise']}\t{hypothesis}\n"
                )
            else:
                raise ValueError(data_format)
