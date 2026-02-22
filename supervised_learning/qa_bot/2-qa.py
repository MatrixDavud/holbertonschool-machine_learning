#!/usr/bin/env python3
"""
Interactive Question Answering loop using a reference text.
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference document.

    Args:
        reference (str): reference text used to answer questions
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

        answer = question_answer(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
