import sys
import joblib
import numpy as np
import pandas as pd

# from sklearn.linear_model import LogisticRegression
from ft_logistic_regression import FtLogisticRegression as LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from config import MODEL_PATH, RANDOM_STATE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from config import PREPROCESSOR_PATH
from typing import Literal


def preprocess(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = "Hogwarts House"

    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday"]
    X = X[X.columns.difference(cols_to_drop).tolist()]

    y = X[target]
    X = X.drop(columns=target)

    categorical_features = ["Best Hand"]
    num_features = X.columns.difference(categorical_features).tolist()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    num_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder())]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, categorical_features)],
        remainder="passthrough",
    )

    X_proc = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "preproc.pkl")

    return X_proc, y_enc


def train(df: pd.DataFrame):
    X_train, y_train = preprocess(df)

    clf = LogisticRegression().fit(X_train, y_train)
    joblib.dump(clf, "model.pkl")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_train.py <train dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    train(df)

    print(f"model saved in {MODEL_PATH}")


main()
