#!/usr/bin/env python3
"""Implementing Bayesian Probability concepts."""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that p (probability of severe
    side effects) lies within the range [p1, p2], given x and n and a
    uniform prior on p.

    Parameters:
        x (int): Number of observed severe side effects
        n (int): Total number of patients observed
        p1 (float): Lower bound of the interval
        p2 (float): Upper bound of the interval

    Returns:
        float: Posterior probability that p ∈ [p1, p2]
    """

    # --- Input validation ---
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
                )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # --- Posterior is Beta(x+1, n-x+1) due to uniform prior ---
    alpha = x + 1
    beta = n - x + 1

    # Beta cumulative distribution function: I_x(a, b)
    cdf_p1 = special.betainc(alpha, beta, p1)
    cdf_p2 = special.betainc(alpha, beta, p2)

    # Posterior probability that p is in [p1, p2]
    return cdf_p2 - cdf_p1
