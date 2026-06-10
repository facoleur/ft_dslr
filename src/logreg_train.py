import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from config import MODEL_PATH
from ft_logistic_regression import FtLogisticRegression as LogisticRegression


def preprocess(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Returns cleaned X, cleaned Y, and the preprocessor pipeline to transform() Test and Valid data.
    """
    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday"]
    X = X[X.columns.difference(cols_to_drop).tolist()]

    categorical_features = ["Best Hand"]
    num_features = X.columns.difference(categorical_features).tolist()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    num_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="passthrough",
    )

    X_proc = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "preproc.pkl")

    return np.asarray(X_proc), np.asarray(y_enc), preprocessor


def train(df: pd.DataFrame):
    target = "Hogwarts House"

    y = df[target]
    X = df.drop(columns=target)

    X_train, y_train, _ = preprocess(X, y)

    clf = LogisticRegression().fit(X_train, y_train)
    joblib.dump(clf, "model.pkl")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_train.py <train dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    train(df)

    print(f"model saved in {MODEL_PATH}")


if __name__ == "__main__":
    main()
