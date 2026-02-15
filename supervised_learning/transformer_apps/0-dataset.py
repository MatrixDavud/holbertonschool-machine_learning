#!/usr/bin/env python3
"""
Module for loading and preprocessing dataset for machine translation
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for loading and preparing translation data
    """

    def __init__(self):
        """
        Initialize the Dataset class with train/valid splits and tokenizers
        """
        # Load the ted_hrlr_translate/pt_to_en dataset
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )

        # Get train and validation splits
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Create tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset with examples as tuple (pt, en)
                pt: tf.Tensor containing Portuguese sentence
                en: tf.Tensor containing English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load pre-trained tokenizers
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        return tokenizer_pt, tokenizer_en
