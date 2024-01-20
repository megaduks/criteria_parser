from typing import Set, List
import numpy as np


def jaccard_score(a: Set, b: Set, mode: str = "strict") -> float:
    """Computes different versions of the Jaccard score depending on the requested mode

    strict: |a & b| / |a + b|
    relaxed: |a & b| / min{|a|,|b|}
    left: |a & b| / |a|
    right: |a & b| / |b|

    Args:
        a (Set): Set a
        b (Set): Set b
        mode (str, optional): Mode. Defaults to "strict".

    Returns:
        float: Jaccard score
    """

    if (not a) or (not b):
        return 0.

    if mode == "strict":
        return len(a.intersection(b)) / len(a.union(b))
    elif mode == "relaxed":
        return len(a.intersection(b)) / min(len(a), len(b))
    elif mode == "left":
        return len(a.intersection(b)) / len(a)
    elif mode == "right":
        return len(a.intersection(b)) / len(b)


def entity_coverage_score(
        ents_true: List[str],
        ents_pred: List[str],
        jaccard_mode: str = "strict",
) -> float:
    """Computes the entity coverage score for a given mode

    Args:
        ents_true (List[str]): List of entities in the ground truth
        ents_pred (List[str]): List of entities in the prediction
        jaccard_mode (str, optional): Jaccard mode. Defaults to "strict".

    Returns:
        float: Average Jaccaard score of predicted entities
    """

    if not ents_true:
        return 0.

    if not ents_pred:
        return 0.

    # split each ent_pred and find maximum jaccard score among ents_true
    scores = [
        max([jaccard_score(set(e_true.split()), set(e_pred.split()), mode=jaccard_mode) for e_pred in ents_pred])
        for e_true
        in ents_true
    ]
    return np.mean(scores)
