import math
import sys

import pandas as pd


def mean(n: pd.Series) -> float:
    n = n.dropna()
    try:
        mean = float(sum(n) / len(n))
        return mean
    except ZeroDivisionError:
        return 0


def std(n: pd.Series, mean):
    n = n.dropna()
    try:
        var = sum((x - mean) ** 2 for i, x in enumerate(n)) / len(n)
        return math.sqrt(var)
    except ZeroDivisionError:
        return 0


def quartile(data: pd.Series, q: float):
    data = data.dropna()
    s = sorted(data)
    n = len(data)
    idx = q * n - 1
    lower = int(idx)
    upper = int(idx + 1)
    frac = idx - lower
    try:
        if upper >= n:
            return float(s[lower])
        return float(s[lower] + frac * (s[upper] - s[lower]))
    except IndexError:
        return 0


def describe(path: str):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit("File not found")

    descr = []
    col: str

    indices = ["count", "mean", "std", "min", "q1", "q2", "q3", "max"]
    descr = {stat: [] for stat in indices}

    numeric = df.select_dtypes(include="number").drop(columns="Index", errors="ignore")
    for col in numeric:
        c = pd.Series(df[col])
        m = mean(c)

        descr["count"].append(float(c.notnull().sum()))
        descr["mean"].append(mean(c))
        descr["std"].append(std(c, m))
        descr["min"].append(min(c))
        descr["q1"].append(quartile(c, 0.25))
        descr["q2"].append(quartile(c, 0.50))
        descr["q3"].append(quartile(c, 0.75))
        descr["max"].append(max(c))

    res = pd.DataFrame(descr).T
    res.columns = [f"V{i}" for i in range(len(res.columns))]

    print(res)
    # print(numeric.describe())


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python describe.py <dataset path>")
    path = sys.argv[1]

    describe(path)


main()
