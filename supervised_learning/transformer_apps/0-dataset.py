#!/usr/bin/env python3
"""
Dataset loading for transformer.
"""
import tensorflow_datasets as tfds
import transformers



class Dataset:
    """
    asdsadsada
    """
    def __init__(self):
        """
            Init the object.
        """
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']
        a, b = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = a, b

    def tokenize_dataset(self, data):
        """
            Tokenize dataset.
        """

        abv = "neuralmind/bert-base-portuguese-cased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        tpt = tpt.train_new_from_iterator(
            pt_sentences,
            vocab_size=2**13
        )

        ten = ten.train_new_from_iterator(
            en_sentences,
            vocab_size=2**13
        )

        return tpt, ten