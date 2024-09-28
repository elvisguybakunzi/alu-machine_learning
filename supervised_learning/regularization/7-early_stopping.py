#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Args:
        cost: float, current validation cost of the network
        opt_cost: float, the lowest recorded validation cost of the network
        threshold: float, the threshold for considering an improvement
        patience: int, the patience count for early stopping
        count: int, how long the threshold has not been met

    Returns:
        A boolean indicating whether the
        network should stop early, and the updated count.
    """
    # Check if the improvement in cost is greater than the threshold
    if opt_cost - cost > threshold:
        # Reset count if there's an improvement
        count = 0
    else:
        # Increment count if there's no significant improvement
        count += 1

    # Determine if the patience has been exceeded
    if count >= patience:
        return True, count  # Stop early
    else:
        return False, count  # Continue training
