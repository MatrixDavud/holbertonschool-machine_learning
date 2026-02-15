#!/usr/bin/env python3
"""
Load, tokenize tensorflow Dataset
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and
        validation datasets. Also initializes tokenizers for
        Portuguese and English.
        """
        # Try to load with try_gcs flag (use Google Cloud Storage)
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True,
            try_gcs=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True,
            try_gcs=True
        )

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained
        models.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en)
            sentences.

        Returns:
            tokenizer_pt: Tokenizer for Portuguese.
            tokenizer_en: Tokenizer for English.
        """
        # Load the pre-trained tokenizers directly
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        return tokenizer_pt, tokenizer_en
