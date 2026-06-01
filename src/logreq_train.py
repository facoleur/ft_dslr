import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import MODEL_PATH, RANDOM_STATE


# TODO: split so i can reuse preprocess on test data.
# -> scaling fit_transform must not be used on test data
# ALSO: save everything into model.pkl with a dict
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = "Hogwarts House"

    X = df.drop(columns=target)
    y = pd.Series(df[target])

    # encode categorical
    X["Best Hand"] = X["Best Hand"].map({"Left": 0, "Right": 1})  # type: ignore
    # extract month
    X["Birth Month"] = pd.to_datetime(X["Birthday"]).dt.month

    # drop noise + target
    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday"]
    X = X[X.columns.difference(cols_to_drop).tolist()]  # type: ignore

    categorical_features = ["Best Hand"]
    num_features = X.columns.difference(categorical_features).tolist()

    # handle na
    X[num_features] = X[num_features].fillna(X[num_features].median())  # type: ignore
    X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])  # type: ignore

    # scaler
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")

    return pd.DataFrame(X, columns=X.columns), y


def train(df: pd.DataFrame):
    X_train, y_train = preprocess(df)
    clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train, y_train)
    joblib.dump(clf, f"{MODEL_PATH}/model.pkl")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_train.py <train dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    train(df)


main()
