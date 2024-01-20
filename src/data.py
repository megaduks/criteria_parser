import pandas as pd
from pathlib import Path
from typing import List


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
