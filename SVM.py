import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, kernel, C, lr=1e-4, epochs=1000):
        # alpha : (200, 1)
        # y : (200, 1)
        # X : (200, 2)
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        self.alpha = None
        self.kernel = kernel
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def _rbf_kernel(self, X1: np.array, X2: np.array):
        """
        X1: (n1, d)
        X2: (n2, d)
        return: (n1, n2), matrix K[i,j] = exp(-gamma ||X1[i]-X2[j]||^2)
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        gamma = 1 / (2 * np.var(self.X))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X1[i] - X2[j]) ** 2)
        return K

    def _linear_kernel(self, X1: np.array, X2: np.array):
        return X1 @ X2.T

    def fit(self, X, y):
        self.X = X
        self.y = y
        y = y.reshape(-1, 1)
        alpha = np.random.randn(X.shape[0], 1)

        if self.kernel == "linear":
            # linear kernel
            K = self._linear_kernel(X, X)
            Q = (y @ y.T) * K

            for _ in range(self.epochs):
                lag = -0.5 * alpha.T @ Q @ alpha + np.sum(alpha)
                grad_alpha = 1 - (alpha.T @ Q)
                alpha += self.lr * grad_alpha.T
                alpha = np.clip(alpha, 0, self.C)
                alpha -= (alpha.T @ y) / np.sum(y ** 2) * y

            self.alpha = alpha
            self.w = (alpha * y).T @ X
            # choisir un vector de support aléatoire pour calculer b
            sv = (alpha > 1e-5).ravel()
            k = np.where(sv)[0][0]
            self.b = y[k] - X[k] @ self.w.T

        elif self.kernel == "rbf":
            # rbf kernel
            K = self._rbf_kernel(X, X)
            Q = (y @ y.T) * K

            for _ in range(self.epochs):
                lag = -0.5 * alpha.T @ Q @ alpha + np.sum(alpha)
                grad_alpha = 1 - (alpha.T @ Q)
                alpha += self.lr * grad_alpha.T
                alpha = np.clip(alpha, 0, self.C)
                alpha -= (alpha.T @ y) / np.sum(y ** 2) * y

            self.alpha = alpha
            sv = (alpha > 1e-5).ravel()
            k = np.where(sv)[0][0]  # choisir un vector de support aléatoire pour calculer b
            coeff = alpha * y  # (n, 1)
            K_sv = self._rbf_kernel(X[k:k + 1], X)  # (1, n)
            # b = y_k - Σ α_i y_i K(x_i, x_k)
            self.b = float((y[k] - K_sv @ coeff)[0][0])

    def predict(self, X_test):
        if self.kernel == "linear":
            scores = X_test @ self.w.T + self.b
        elif self.kernel == "rbf":
            K = self._rbf_kernel(X_test, self.X)
            coeff = self.alpha * self.y.reshape(-1, 1)
            scores = K @ coeff + self.b
        return np.sign(scores).ravel()

    def visualize(self):
        assert self.X.shape[1] == 2

        X = self.X
        y = self.y.reshape(-1)

        # plot X points
        plt.figure(figsize=(6, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y, cmap='bwr', s=40, edgecolors='k')

        # grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]  # (N_grid, 2)

        # calculate decision for each grid point
        if self.kernel == "linear":
            Z = (grid @ self.w.T + self.b).reshape(xx.shape)
        elif self.kernel == "rbf":
            K_grid = self._rbf_kernel(grid, self.X)  # (N_grid, n_train)
            coeff = self.alpha * self.y.reshape(-1, 1)  # (n_train, 1)
            Z = (K_grid @ coeff + self.b).reshape(xx.shape)  # (N_grid, 1)

        # border and margin
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='-')  # border
        plt.contour(xx, yy, Z, levels=[-1, 1], linestyles='--')  # margin

        # hightlight support vectors
        sv = (self.alpha > 1e-5).ravel()
        plt.scatter(X[sv, 0], X[sv, 1], s=120, facecolors='none',
                    edgecolors='yellow', linewidths=2, label='SV')

        plt.title(f"SVM with {self.kernel} kernel")
        plt.legend()
        plt.show()