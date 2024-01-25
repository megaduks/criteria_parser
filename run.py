import fire
import pandas as pd

from src.data import load_chia


def process_chia(n: int = None, random: bool = False):
    """Processes the Chia dataset

    Args:
        n (int, optional): Number of rows to read. Defaults to None.
        random (bool, optional): Whether to read rows randomly. Defaults to False.
    """
    df = load_chia()

    if random:
        print(df.sample(frac=1.0)[:n])
    else:
        print(df[:n])


if __name__ == "__main__":
    fire.Fire()
