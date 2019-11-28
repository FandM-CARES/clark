""" Utilities for input validation """
from typing import List

def check_X_y_same_size(X: List[List[int]], y: List[int]) -> None:
    """
    Validates that X and y contain the same number of samples

    """

    print("Not overriden")
    if X.shape[1] != y.shape[0]:
        raise ValueError(
            f"Differening number of samples provided: X has {X.shape[1]} samples and y has {y.shape[0]} samples")
