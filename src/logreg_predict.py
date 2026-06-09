import sys
import joblib
import pandas as pd
import seaborn as sns
from config import MODEL_PATH
from preprocess import preprocess, split_num_cat_feat


def predict(df: pd.DataFrame):
    pass


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_predict.py <test dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    model = joblib.load(MODEL_PATH)
    clf = model["model"]
    scaler = model["scaler"]

    X_test, _ = preprocess(df)

    num, _ = split_num_cat_feat(X_test)
    X_test[num] = scaler.transform(X_test[num])

    target = "Hogwarts House"

    y = clf.predict(X_test)

    df = X_test.copy()

    df["Hogwarts House"] = y
    print(df)

    df.to_csv("prediction.csv")

    ax = sns.countplot(x=target, data=df)
    ax.figure.savefig("test.png")  # type: ignore


if __name__ == "__main__":
    main()
