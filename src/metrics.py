from typing import Set, List
import numpy as np


def llm_none_cleaner(output: List[str]) -> List[str]:
    """Reads the output returned by the LLM and returns a cleaned version of the output
    by changing all instances of lists containing 'None' to an empty list.

    Args:
        output (List): Output returned by the LLM
    Returns:
        List: Empty list if the output contains only 'None', otherwise the original output
    """
    none_strings = ["None", "none", "NONE", "None.", "none.", "NONE."]

    # if the only string contained in the output is one of the none_strings, return an empty list
    if len(output) == 1 and output[0] in none_strings:
        return []
    else:
        return output


def str_to_BIO_entities(text: str, ent_name: str, entities: List[str]) -> List[str]:
    """Converts a list of entities into a list of BIO entities

    Args:
        text (str): Text
        ent_name (str): name of the entity
        entities (List[str]): List of entities
    Returns:
        List[str]: List of BIO entities
    """

    text = text.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?")

    # create a list of 'O' entities
    bio_entities = ["O"] * len(text.split())

    # for each entity, split it and find the index of the first word
    for entity in entities:

        # if entity is not in text, skip
        if entity not in text:
            continue

        entity = entity.split()

        start = text.split().index(entity[0])

        # mark the first word as 'B'
        bio_entities[start] = f"B-{ent_name}"

        # mark the rest of the words as 'I'
        for i in range(start + 1, start + len(entity)):
            bio_entities[i] = f"I-{ent_name}"

    return bio_entities


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
        return 0.0

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
    """Computes the entity coverage score for a given mode. Given a list of true entities and the list of
    predicted entities, the entity coverage score is the average Jaccard score of the predicted entities when
    each predicted entity is matched against the highest matching expected entity.

    Args:
        ents_true (List[str]): List of entities in the ground truth
        ents_pred (List[str]): List of entities in the prediction
        jaccard_mode (str, optional): Jaccard mode. Defaults to "strict".

    Returns:
        float: Average Jaccaard score of predicted entities
    """

    # raise TypeError when arguments are not lists
    if not (isinstance(ents_true, list) and isinstance(ents_pred, list)):
        raise TypeError("Entities must be a list")

    if not ents_true:
        return 0.0

    if not ents_pred:
        return 0.0

    # split each ent_pred and find maximum jaccard score among ents_true
    scores = [
        max(
            [
                jaccard_score(
                    set(e_true.split()), set(e_pred.split()), mode=jaccard_mode
                )
                for e_pred in ents_pred
            ]
        )
        for e_true in ents_true
    ]
    return np.mean(scores)


def entity_match_score(ents_true: List[List[str]], ents_pred: List[List[str]]) -> float:
    """Computes the entity match score for a given mode. Given a list of true entities and the list of
    predicted entities, the entity match score performs a pairwise comparison of the entities and returns 1 if
    there is a match and 0 otherwise.

    Args:
        ents_true (List[List[str]]): List of entities in the ground truth
        ents_pred (List[List[str]]): List of entities in the prediction

    Returns:
        float: Percentage of true entities that have a match in the predicted entities
    """

    if not ents_true:
        return 0.0

    if not ents_pred:
        return 0.0

    # transform ents_pred by changing every ['None'] to []
    ents_pred = [e if e != ["None"] else [] for e in ents_pred]

    # transform each list of entities into a set
    ents_true = [set(e) for e in ents_true]
    ents_pred = [set(e) for e in ents_pred]

    # make a pairwise comparison resulting in 1 if sets match, 0 otherwise
    matches = [
        1 if e_true == e_pred else 0 for e_true, e_pred in zip(ents_true, ents_pred)
    ]

    # return the percentage of matches
    return np.mean(matches)
