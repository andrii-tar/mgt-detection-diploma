from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/Training_Essay_Data.csv")
OUTPUT_HUMAN = Path("data/raw/kaggle_train_hg.csv")
OUTPUT_GENERATED = Path("data/raw/kaggle_train_mg.csv")


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
