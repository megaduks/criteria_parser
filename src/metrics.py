from typing import Set


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
