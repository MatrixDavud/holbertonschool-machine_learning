#!/usr/bin/env python3
"""
Module that provides a question answering function using BERT.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document
    that answers a question.

    Args:
        question (str): question to answer
        reference (str): reference document

    Returns:
        str: answer snippet or None
    """
    if not question or not reference:
        return None

    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    )

    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]

    outputs = model(
        input_word_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=segment_ids
    )

    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]

    start_index = tf.argmax(start_logits[0]).numpy()
    end_index = tf.argmax(end_logits[0]).numpy()

    if end_index < start_index:
        return None

    tokens = input_ids[0][start_index:end_index + 1]
    answer = tokenizer.decode(tokens)

    if not answer.strip():
        return None

    return answer
