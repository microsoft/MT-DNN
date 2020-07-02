# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4
    SequenceLabeling = 5
    MaskLM = 6


class DataFormat(IntEnum):
    Init = 0
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Sequence = 5
    MLM = 6


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4


class TaskDefType(dict):
    def __init__(
        self,
        label_vocab,
        n_class,
        data_format,
        task_type,
        metric_meta,
        split_names,
        enable_san,
        dropout_p,
        loss,
        kd_loss,
        data_paths,
    ):
        """
            :param label_vocab: map string label to numbers.
                only valid for Classification task or ranking task.
                For ranking task, better label should have large number
        """
        super().__init__(
            **{k: repr(v) for k, v in locals().items()}
        )  # ensure the class is JSON serializable
        self.label_vocab = label_vocab
        self.n_class = n_class
        self.data_format = data_format
        self.task_type = task_type
        self.metric_meta = metric_meta
        self.split_names = split_names
        self.enable_san = enable_san
        self.dropout_p = dropout_p
        self.loss = loss
        self.kd_loss = kd_loss
        self.data_paths = data_paths

    @classmethod
    def from_dict(cls, dict_rep):
        return cls(**dict_rep)
