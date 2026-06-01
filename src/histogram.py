import pandas as pd
import sys
import math
import seaborn as sns
import matplotlib.pyplot as plt


def histogram(df: pd.DataFrame):
    courses = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))

    for ax, col in zip(axes.flat, courses):
        sns.histplot(data=df, x=col, ax=ax, linewidth=0, hue="Hogwarts House", alpha=0.5)
        ax.set_title(col)
        ax.set_xlabel("")
        plt.tight_layout()

    plt.savefig("histogram.png")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python describe.py <dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    histogram(df)


main()
