import pandas as pd
import json
from pathlib import Path
from typing import List, Dict


DATA_DIR = Path("../data")


def get_entities(clinical_trail_no: str, mode: str, entity_name: str) -> List:
    """Read annotations from .ann file and return a list of entities of type e

    Args:
        clinical_trail_no (str): Clinical trial number
        mode (str): Inclusion or exclusion criteria
        entity_name (str): Entity type

    Returns:
        List: List of entities of type e
    """

    entities = []

    with open(f"{DATA_DIR}/{clinical_trail_no}{mode}.ann", "rt") as f:
        data = f.read().splitlines()

    for row in data:
        if entity_name in row:
            entities.append(" ".join(row.split()[4:]))

    return entities


def load_chia() -> pd.DataFrame:
    """Exports Chia annotated dataset as a Pandas dataframe

    Returns:
        pd.DataFrame: Chia annotated dataset as a Pandas dataframe
    """

    _lst = []

    ent_map = {
        "drugs": "Drug",
        "persons": "Person",
        "procedures": "Proceure",
        "conditions": "Condition",
        "devices": "Device",
        "visits": "Visit",
        "scopes": "Scope",
        "observations": "Observation",
        "measurements": "Measurement",
    }

    for mode in ["_inc", "_exc"]:

        criteria_files = DATA_DIR.glob(f"*{mode}.txt")

        for f in criteria_files:
            clinical_trial_no = str(f).lstrip("data/").rstrip(f"{mode}.txt")

            with open(f, "rt") as f:
                criteria = " ".join(f.read().splitlines())

            _rec = {"ct_no": clinical_trial_no, "criteria": criteria,
                    "mode": "inclusion" if mode == "_inc" else "exclusion"}

            for entity in ent_map:
                ents = get_entities(clinical_trial_no, mode, ent_map[entity])
                _rec[entity] = ents if ents else None

            _lst.append(_rec)

    return pd.DataFrame(_lst)


def load_fb() -> pd.DataFrame:
    """Exports FB annotated dataset as a Pandas dataframe

    Returns:
        pd.DataFrame: FB annotated dataset as a Pandas dataframe
    """
    input_file = DATA_DIR / "fb_dset_preprocessed.json"

    with open(input_file) as f:
        _data = json.load(f)

    df = pd.json_normalize(_data)

    return df


def train_test_dev_split(df: pd.DataFrame, random_seed=42, ratio=(70, 20, 10)) -> Dict:
    """Splits the dataset into train, test and dev sets using the given ratios and random seed

    Args:
        df (pd.DataFrame): Input dataframe
        random_seed (int, optional): Random seed. Defaults to 42.
        ratio (tuple, optional): Train, test and dev ratios. Defaults to (70, 20, 10).

    Returns:
        Dict: Dictionary with train, test and dev sets
    """

    assert sum(ratio) == 100, "Sum of ratios must be 100"

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_ratio, test_ratio, dev_ratio = ratio

    train_size = int(len(df) * train_ratio / 100)
    test_size = int(len(df) * test_ratio / 100)
    dev_size = int(len(df) * dev_ratio / 100)

    train = df[:train_size]
    test = df[train_size:train_size+test_size]
    dev = df[train_size+test_size:]

    return {"train": train, "test": test, "dev": dev}
