#!/usr/bin/env python3
"""Bar graph."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Create a bar graph representing number of fruit per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))

    colors = {
        'apples': 'red',
        'bananas': 'yellow',
        'oranges': '#ff8000',
        'peaches': '#ffe5b4'
    }

    bottom = np.zeros(len(people))

    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    for i, fruit_name in enumerate(fruits):
        plt.bar(x, fruit[i], width=0.5, bottom=bottom,
                color=colors[fruit_name], label=fruit_name)
        bottom += fruit[i]

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()

    plt.show()
