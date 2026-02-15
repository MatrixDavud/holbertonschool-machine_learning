#!/usr/bin/env python3
"""
Dataset module for Portuguese to English translation
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares the TED Portuguese to English dataset
    """

    def __init__(self):
        """
        Class constructor

        Loads:
        - ted_hrlr_translate/pt_to_en train split
        - ted_hrlr_translate/pt_to_en validation split

        Initializes pretrained tokenizers.
        """

        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )

        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates pretrained sub-word tokenizers

        Args:
            data: tf.data.Dataset containing (pt, en) pairs

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
