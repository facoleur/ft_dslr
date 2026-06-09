from math import exp
import numpy as np


class NotFittedError(ValueError):
    def __init__(self, msg="This instance is not fitted yet Call .fit first") -> None:
        super().__init__(msg)


class FtLogisticRegression:
    """Multiclass logistic regression classifier trained with softmax and gradient descent."""

    def __init__(self) -> None:
        """Initialize hyperparameters and empty model parameters."""
        self.lr = 0.1
        self.max_iter = 2000

        self.W_ = np.array([])
        self.b_ = np.array([])
        self.loss_ = np.array([])

    def fit(self, X, y):
        """Fit the model to the data via gradient descent."""
        self.classes_ = np.unique(y)

        self.W_ = np.zeros((len(X[0]), len(self.classes_)))
        self.b_ = np.zeros(len(self.classes_))

        for _ in range(self.max_iter):
            P = self._softmax(X)
            self.loss_ = np.append(self.loss_, self._loss(P, y))

            dw, db = self._gradient(P, y, X, len(self.classes_))

            self.W_ = self.W_ - self.lr * dw
            self.b_ = self.b_ - self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for each sample."""
        return self._softmax(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample."""
        if self.W_ is None or self.b_ is None:
            raise NotFittedError

        proba = self.predict_proba(X)

        return np.argmax(proba, axis=1)

    def _logit_score(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the linear logit scores for each class."""
        return np.dot(X, W) + b

    def _softmax(self, X) -> np.ndarray:
        """Return softmax class probabilities for each sample."""
        scores = self._logit_score(X, self.W_, self.b_)
        exp_score = np.exp(scores)
        sum_exp_score = exp_score.sum(axis=1).reshape(-1, 1)

        return np.array(np.exp(scores) / sum_exp_score)

    def _gradient(self, P, y, X, classes_len) -> tuple[np.ndarray, np.ndarray]:
        """Compute the weight and bias gradients of the loss."""
        # y_arr = np.zeros((len(y), classes_len))
        # for i in range(len(y)):
        #     idx = y[i]
        #     y_arr[i][idx] = 1
        y_one_hot = np.eye(classes_len)[y]

        error = P - y_one_hot
        dw = (1 / len(y)) * np.dot(X.T, error)
        db = (1 / len(y)) * error.sum(axis=0)

        return dw, db

    def _loss(self, P: np.ndarray, y) -> float:
        """Compute the cross-entropy loss for the predictions."""
        P_win = np.array([p.max() for p in P])
        L = -sum(y * np.log(P_win))
        return L
