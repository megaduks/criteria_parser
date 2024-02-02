import fire
import pandas as pd

from src.data import load_chia
from src.prompt import few_shot_entity_recognition


def process_chia(n: int = None, random: bool = False):
    """Processes the Chia dataset

    Args:
        n (int, optional): Number of rows to read. Defaults to None.
        random (bool, optional): Whether to read rows randomly. Defaults to False.
    """
    df = load_chia()

    if random:
        for _, row in df.sample(frac=1.)[:n].iterrows():
            print(row["criteria"])
            print("TRUE: ", row["drugs"], row["persons"], row["conditions"])
            print("PREDICTED: ", few_shot_entity_recognition(row["criteria"]))
            print("-" * 100)
    else:
        # iterate over rows of the dataframe
        for _, row in df[:n].iterrows():
            print(row["criteria"])
            print(row["drugs"], row["persons"], row["conditions"])
            print(few_shot_entity_recognition(row["criteria"]))
            print("-" * 100)


if __name__ == "__main__":
    fire.Fire()
