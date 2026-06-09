import pandas as pd


# TODO: split so i can reuse preprocess on test data.
# -> scaling fit_transform must not be used on test data
# ALSO: save everything into model.pkl with a dict
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # target = "Hogwarts House"

    X = df

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

    return pd.DataFrame(X, columns=X.columns)


def split_num_cat_feat(X: pd.DataFrame):
    categorical_features = ["Best Hand"]
    num_features = X.columns.difference(categorical_features).tolist()

    return num_features, categorical_features
