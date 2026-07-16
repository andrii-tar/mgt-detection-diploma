import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_config import DATA_ROOT


INPUT_PATH = DATA_ROOT / "raw/Training_Essay_Data.csv"
OUTPUT_HUMAN = DATA_ROOT / "raw/kaggle_train/kaggle_train_hg.csv"
OUTPUT_GENERATED = DATA_ROOT / "raw/kaggle_train/kaggle_train_mg.csv"


def main() -> None:
    OUTPUT_HUMAN.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df.reset_index(drop=True)
    df["id"] = df.index

    human = df[df["generated"] == 0][["id", "text"]]
    generated = df[df["generated"] == 1][["id", "text"]]

    human.to_csv(OUTPUT_HUMAN, index=False)
    generated.to_csv(OUTPUT_GENERATED, index=False)


if __name__ == "__main__":
    main()
