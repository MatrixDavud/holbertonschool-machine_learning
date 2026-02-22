#!/usr/bin/env python3
"""
Interactive question loop.

Prompts the user with Q: and prints A: as a response.
If the user inputs exit keywords, the program exits.
"""


def main():
    """
    Runs the interactive question loop.
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

        print("A:")


if __name__ == "__main__":
    main()
