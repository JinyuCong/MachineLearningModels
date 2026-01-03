import numpy as np
from sklearn.datasets import load_iris


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


class MulticlassLogisticRegression:
    def __init__(self):
        self.w = None
        self.num_classes = None
        self.loss_history = []

    def _normalize(self, X):
        """
        Normalize X to have mean 0 and standard deviation 1
        """
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / (sigma + 1e-12)

    def _softmax(self, z):
        """
        :param z: logit of the X @ w, shape: (n, k)
        :return:
        """
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def fit(self, X, y, n_iter=100, lr=0.1):
        """
        Train the model on training set
        :param X: train set, shape: (n, d)
        :param y: target, shape: (n, 1), num classes = k
        :param n_iter: int, number of iterations
        """
        X = self._normalize(X)
        n, d = X.shape

        # 类别编码转换为0开始到k-1
        classes, y_enc = np.unique(y, return_inverse=True)
        self.num_classes = len(classes)
        onehot_y = np.eye(self.num_classes)[y_enc]

        # 加bias
        X = np.hstack([X, np.ones((n, 1))])
        d = d + 1

        self.w = np.random.randn(d, self.num_classes) * 0.01

        eps = 1e-12
        for epoch in range(n_iter):
            z = X @ self.w  # (n, k)
            pi = self._softmax(z)

            loss = -np.sum(onehot_y * np.log(np.clip(pi, eps, 1.0))) / n
            self.loss_history.append(loss)

            grad_w = X.T @ (pi - onehot_y) / n

            self.w -= lr * grad_w
            print(loss)


if __name__ == "__main__":
    ds = load_iris()
    X, y = ds["data"], ds["target"].reshape(-1, 1)

    model = MulticlassLogisticRegression()
    model.fit(X, y, n_iter=500)
