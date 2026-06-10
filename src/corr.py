import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import VIZUALISATION_PATH


def corr(df: pd.DataFrame):
    """
    Correlation plot instead of scatter plot because scatter doesnt bring much compared the the pairplot.
    """
    features = df.select_dtypes(include="number").columns.difference(["Index"]).tolist()
    df = df[features]  # type: ignore

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax)
    fig.savefig(f"{VIZUALISATION_PATH}/corr.png", bbox_inches="tight")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python describe.py <dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    corr(df)


if __name__ == "__main__":
    main()
