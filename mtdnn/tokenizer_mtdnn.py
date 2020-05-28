# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from mtdnn.common.types import EncoderModelType
from mtdnn.modeling_mtdnn import MODEL_CLASSES
from mtdnn.tasks.config import MTDNNTaskDefs


class MTDNNTokenizer:
    """Wraps the hugging face transformer tokenizer for Preprocessing GLUE/SNLI/SciTail datasets."""

    def __init__(
        self, model_name: str = "bert-base-uncased", do_lower_case: bool = False,
    ):
        self._model_name = model_name
        self.literal_model_name = model_name.split("-")[0]
        self.model_type = EncoderModelType[
            self.literal_model_name.upper()
        ].name  # BERT = 1, ROBERTA = 2
        mt_dnn_model_name_fmt = model_name.replace("-", "_")  # format to mt-dnn format
        self.mt_dnn_suffix = (
            f"{mt_dnn_model_name_fmt}_lower"
            if do_lower_case
            else f"{mt_dnn_model_name_fmt}"
        )
        _, _, tokenizer_class = MODEL_CLASSES[self.literal_model_name]
        self._tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=do_lower_case
        )

    def encode(
        self,
        text: str = "",
        text_pair: str = "",
        max_length: int = 512,
        enable_padding: bool = False,
        pad_on_left: bool = False,
        pad_token: int = 0,
        pad_token_segment_id: int = 0,
        mask_padding_with_zero: bool = False,
    ):
        """
            Returns a tuple containing the encoded sequence or sequence pair and additional informations:
            the input mask and segment id
        """
        # set mask_padding_with_zero default value as False to keep consistent with original setting
        inputs = self._tokenizer.encode_plus(
            text, text_pair, add_special_tokens=True, max_length=max_length,
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
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            assert (
                len(input_ids) == max_length
            ), f"[ERROR] - Input Ids length: {len(input_ids)} does not match max length: {max_length}"

            assert (
                len(attention_mask) == max_length
            ), f"[ERROR] - Attention mask length: {len(attention_mask)} does not match max length: {max_length}"

            assert (
                len(token_type_ids) == max_length
            ), f"[ERROR] - Token types id length: {len(token_type_ids)} does not match max length: {max_length}"

        if self.model_type.lower() in ["bert", "roberta"]:
            attention_mask = None

        if self.model_type.lower() not in ["distilbert", "bert", "xlnet"]:
            token_type_ids = [0] * len(token_type_ids)

        # input_ids, input_mask, segment_id
        return (
            input_ids,
            attention_mask,
            token_type_ids,
        )

    def get_model_name(self) -> str:
        return self._model_name
