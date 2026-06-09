import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from preprocess import preprocess, split_num_cat_feat
from config import MODEL_PATH, RANDOM_STATE


def train(df: pd.DataFrame):

    X, y = preprocess(df)

    # categorical_features = ["Best Hand"]
    # num_features = X.columns.difference(categorical_features).tolist()
    num, _ = split_num_cat_feat(X)

    scaler = StandardScaler()
    X[num] = scaler.fit_transform(X[num])

    print(X)

    clf = LogisticRegression(random_state=RANDOM_STATE).fit(X, y)
    joblib.dump({"scaler": scaler, "model": clf}, MODEL_PATH)

    print(f"model saved in {MODEL_PATH}")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python logreq_train.py <train dataset path>")
    path = sys.argv[1]
    df = pd.read_csv(path)

    train(df)


main()
