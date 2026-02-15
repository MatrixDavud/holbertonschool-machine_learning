#!/usr/bin/env python3
"""
creates a dataset
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    loads and preps a dataset for machine translation
    """
    def __init__(self):
        """constructor"""
        train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train',
            as_supervised=True)
        validate = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
           train)
        self.data_valid = validate.map(self.tf_encode)
        self.data_train = train.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            clean_up_tokenization_spaces=True, use_fast=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            clean_up_tokenization_spaces=True, use_fast=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, 2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, 2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        """
        en_size = self.tokenizer_en.vocab_size
        pt_size = self.tokenizer_pt.vocab_size
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)
        pt_tokens = [pt_size] + pt_tokens + [pt_size + 1]
        en_tokens = [en_size] + en_tokens + [en_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
