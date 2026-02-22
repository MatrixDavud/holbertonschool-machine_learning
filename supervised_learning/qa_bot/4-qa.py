#!/usr/bin/env python3
"""
Multi-reference Question Answering system.
"""

semantic_search = __import__('3-semantic_search').semantic_search
single_qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions using multiple reference documents.

    Args:
        corpus_path (str): path to corpus directory
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        try:
            question = input("Q: ")
        except EOFError:
            print()
            break

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)

        if reference is None:
            print("A: Sorry, I do not understand your question.")
            continue

        answer = single_qa(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
