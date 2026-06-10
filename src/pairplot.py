import sys

import pandas as pd
import seaborn as sns


def pairplot(df: pd.DataFrame):
    target = "Hogwarts House"
    features = df.select_dtypes(include="number").columns.difference(["Index"]).tolist()

    # for i, f in enumerate(features):
    #     for j, g in enumerate(features):
    #         if j == i or f == target or g == target:
    #             continue
    #         ax = axes[i][j]
    #         sns.scatterplot(data=df, x=df[f], y=df[g], ax=ax, hue=target)

    plot = sns.pairplot(df, hue=target, vars=features)
    plot.savefig("pairplot.png")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python describe.py <dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    pairplot(df)

if __name__ == "__main__":
    main()

