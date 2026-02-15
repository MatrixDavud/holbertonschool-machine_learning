#!/usr/bin/env python3
"""Class dataset, loading dataset for machine translation"""


import transformers
import tensorflow_datasets as tfds


class Dataset:
    """loading dataset for machine translation"""
    def __init__(self):
        """class constructor for dataset class"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """creates subword  tokenizers"""
        training_corpus_en = []
        training_corpus_pt = []
        for pt, en in data:
            training_corpus_en.append(en.numpy().decode('utf-8'))
            training_corpus_pt.append(pt.numpy().decode('utf-8'))

        pretrained_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        pretrained_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        vocab_size = 2**13
        self.tokenizer_pt = pretrained_tokenizer_pt.train_new_from_iterator(
            training_corpus_en, vocab_size=vocab_size
        )
        self.tokenizer_en = pretrained_tokenizer_en.train_new_from_iterator(
            training_corpus_pt, vocab_size=vocab_size
        )
        return self.tokenizer_pt, self.tokenizer_en
