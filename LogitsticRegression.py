import numpy as np


class LogisticRegression:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.w = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _normalize(self, X):
        """
        Normalization de X pour stabilizer l'entrainement
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X, y, n_iter=50):
        X = self._normalize(X)

        n, d = X.shape
        y = y.reshape(-1, 1)

        self.w = np.random.randn(d, 1)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        lambda_max = S[0]
        alpha = 4 / (lambda_max ** 2)

        if self.optimizer == "gd":
            for i in range(n_iter):
                z = X @ self.w
                y_hat = self._sigmoid(z)

                loss = -(y.T @ np.log(y_hat) + (1 - y.T) @ np.log(1 - y_hat))
                self.loss_history.append(loss[0, 0])

                grad_w = X.T @ (y_hat - y)
                self.w -= alpha * grad_w

        elif self.optimizer == "newton":
            for i in range(n_iter):
                z = X @ self.w
                y_hat = self._sigmoid(z)

                # eps = 1e-15
                # y_hat = np.clip(y_hat, eps, 1 - eps)
                loss = -(y.T @ np.log(y_hat) + (1 - y.T) @ np.log(1 - y_hat))
                self.loss_history.append(loss[0, 0])

                grad_w = X.T @ (y_hat - y)
                D = np.diag((y_hat * (1 - y_hat)).reshape(-1))
                H = X.T @ D @ X

                self.w -= np.linalg.inv(H) @ grad_w

    def predict(self, X):
        X = self._normalize(X)
        proba = self._sigmoid(X @ self.w)
        return (proba >= 0.5).astype(int)