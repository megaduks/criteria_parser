import fire
import json
import pickle
from pathlib import Path
from src.data import load_chia, load_fb
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


def ner_fb(entity: str, n: int = None, random: bool = False, verbose: bool = False):
    """Applies the LLM prompting to extract NERs from the FB dataset

    Args:
        entity (str): Entity type
        n (int, optional): Number of rows to read. Defaults to None.
        random (bool, optional): Whether to read rows randomly. Defaults to False.
        verbose (bool, optional): Whether to print the results. Defaults to False.
    """
    df = load_fb()["test"]

    results = []

    few_shot_examples = Path("data/few-shots.json")
    with open(few_shot_examples, "r") as f:
        few_shot_examples = json.load(f)[entity]

    if random:
        for _, row in df.sample(frac=1.)[:n].iterrows():
            criterion = row["criterion"]
            ent_true = row[entity]
            ent_pred = few_shot_entity_recognition(few_shot_examples, criterion, entity)

            results.append((entity, criterion, ent_true, ent_pred))
    else:
        for _, row in df[:n].iterrows():
            criterion = row["criterion"]
            ent_true = row[entity]
            ent_pred = few_shot_entity_recognition(few_shot_examples, criterion, entity)

            results.append((entity, criterion, ent_true, ent_pred))

    output_file = Path(f"data/{entity}_ner_results.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    if verbose:
        for entity, criterion, ent_true, ent_pred in results:
            print(criterion)
            print("TRUE: ", ent_true)
            print("PREDICTED: ", ent_pred)
            print("-" * 100)


if __name__ == "__main__":
    fire.Fire()
