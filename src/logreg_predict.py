import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from ft_logistic_regression import FtLogisticRegression as LogisticRegression


def preprocess(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocess test data.
    - drop unused columns
    - apply preprocessor fitted on training data


    """
    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday"]
    X = X[X.columns.difference(cols_to_drop).tolist()]

    preprocessor: ColumnTransformer = joblib.load("preproc.pkl")
    X_proc = preprocessor.transform(X)

    return np.asarray(X_proc)


def save_predictions(y, output="houses.csv"):
    """
    Save predictions to csv.
    Format is:

    Index, Hogwarts House
    ---
    0, Ravenclaw
    """
    le: LabelEncoder = joblib.load("label_encoder.pkl")
    y_decoded = le.classes_

    y = [y_decoded[i] for i in y]

    df = pd.DataFrame()
    df["Hogwarts House"] = pd.Series(y)
    df.index.name = "Index"

    df.to_csv(output, index=True)
    print(f"saved prediction to {output}")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_predict.py <test dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    try:
        model: LogisticRegression = joblib.load("model.pkl")
    except FileNotFoundError:
        sys.exit("No model found. Make sur to run logreg_train.py first")

    X_test = preprocess(df)

    y = model.predict(X_test)

    save_predictions(y)

    # ax = sns.countplot(x=target, data=df)
    # ax.figure.savefig("test.png")  # type: ignore


if __name__ == "__main__":
    main()
