#!/usr/bin/env python3
""" task 0/1 """
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    class for machine translation task using the TED talk
    portuguese to English dataset
    """

    def __init__(self):
        """
        initialize the dataset with training and validation splits and
        tokenizers
        """
        # load portuguese to english dataset
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # create tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        create sub-word tokenizers for the dataset

        args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                 pt is the portuguese sentence
                 en is the english sentence
        returns:
            tokenizer_pt: portuguese tokenizer
            tokenizer_en: english tokenizer
        """
        # collect portuguese and english sentences
        pt_sentences = []
        en_sentences = []

        # convert dataset to lists of sentences
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode())
            en_sentences.append(en.numpy().decode())

        # load pretrained tokenizers
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes portuguese and english sentences into token sequences

        args:
            pt tf.tensor containing portuguese sentence
            en tf.tensor containing english sentence

        returns:
            tuple of two np.ndarrays
            portuguese tokens with start and end tokens
            english tokens with start and end tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy().decode()) + \
            [pt_vocab_size + 1]

        en_tokens = [en_vocab_size] + \
            self.tokenizer_en.encode(en.numpy().decode()) + \
            [en_vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)
