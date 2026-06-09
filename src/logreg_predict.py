import sys
import joblib
import pandas as pd
import seaborn as sns
import numpy as np
from config import MODEL_PATH
import sys
import joblib
from config import MODEL_PATH, PREPROCESSOR_PATH


def preprocess(X: pd.DataFrame) -> np.ndarray:
    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday"]
    X = X[X.columns.difference(cols_to_drop).tolist()]

    preprocessor = joblib.load("preproc.pkl")
    X_proc = preprocessor.transform(X)

    return X_proc


def predict(df: pd.DataFrame):
    pass


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_predict.py <test dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")

    X_test = preprocess(df)

    y = model.predict(X_test)
    y_decoded = le.classes_

    y = [y_decoded[i] for i in y]

    df = pd.DataFrame()
    df["Hogwarts House"] = y
    df.index.name = "Index"

    df.to_csv("houses.csv", index=True)

    # ax = sns.countplot(x=target, data=df)
    # ax.figure.savefig("test.png")  # type: ignore


if __name__ == "__main__":
    main()
