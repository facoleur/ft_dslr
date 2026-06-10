import numpy as np


class NotFittedError(ValueError):
    def __init__(self, msg="This instance is not fitted yet Call .fit first") -> None:
        super().__init__(msg)


class FtLogisticRegression:
    """Multiclass logistic regression classifier trained with softmax and gradient descent."""

    def __init__(self, *, solver="default", max_iter=2000, batch_size=32) -> None:
        """Initialize hyperparameters and empty model parameters."""
        self.lr = 0.1
        self.max_iter = max_iter
        self.solver = solver
        self.batch_size = batch_size

        self.weights_ = np.array([])
        self.bias_ = np.array([])
        self.loss_ = np.array([])

    def fit_bgd(self, X, y):
        for _ in range(self.max_iter):
            P = self._softmax(X)
            self.loss_ = np.append(self.loss_, self._loss(P, y))

            dw, db = self._gradient(P, y, X, len(self.classes_))

            self.weights_ = self.weights_ - self.lr * dw
            self.bias_ = self.bias_ - self.lr * db

        return self

    def fit_mbgd(self, X, y):
        for _ in range(self.max_iter):
            start = 0
            end = self.batch_size - 1
            for xi, yi in zip(X[start:end], y[start:end]):
                P = self._softmax(xi)

                dw, db = self._gradient(P, yi, xi, len(self.classes_))

                self.weights_ = self.weights_ - self.lr * dw
                self.bias_ = self.bias_ - self.lr * db

                start = end + 1
                end += self.batch_size
            self.loss_ = np.append(self.loss_, self._loss(P, yi))

        return self

    def fit_sgd(self, X, y):
        for _ in range(self.max_iter):
            for xi, yi in zip(X, y):
                P = self._softmax(xi)

                dw, db = self._gradient(P, yi, xi, len(self.classes_))

                self.weights_ = self.weights_ - self.lr * dw
                self.bias_ = self.bias_ - self.lr * db
            self.loss_ = np.append(self.loss_, self._loss(P, y))

        return self

    def fit(self, X, y):
        """Fit the model to the data via gradient descent."""
        self.classes_ = np.unique(y)
        self.weights_ = np.zeros((len(X[0]), len(self.classes_)))
        self.bias_ = np.zeros(len(self.classes_))

        if self.solver == "default":
            return self.fit_bgd(X, y)

        if self.solver == "sgd":
            return self.fit_sgd(X, y)

        if self.solver == "mbgd":
            return self.fit_mbgd(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for each sample."""
        return self._softmax(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample."""
        if self.weights_ is None or self.bias_ is None:
            raise NotFittedError

        proba = self.predict_proba(X)

        return np.argmax(proba, axis=1)

    def _logit_score(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the linear logit scores for each class."""
        return np.dot(X, W) + b

    def _softmax(self, X) -> np.ndarray:
        """Return softmax class probabilities for each sample."""
        X = np.atleast_2d(X)
        scores = self._logit_score(X, self.weights_, self.bias_)
        exp_score = np.exp(scores)
        sum_exp_score = exp_score.sum(axis=1).reshape(-1, 1)

        return np.array(np.exp(scores) / sum_exp_score)

    def _gradient(self, P, y, X, classes_len) -> tuple[np.ndarray, np.ndarray]:
        """Compute the weight and bias gradients of the loss."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
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
