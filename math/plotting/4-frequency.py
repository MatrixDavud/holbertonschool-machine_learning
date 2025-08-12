#!/usr/bin/env python3
"""Bar graph plotting."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Draw a bar graph of grade frequencies."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.axis((0, 100, 0, 30))
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xticks(bins)
    plt.show()
