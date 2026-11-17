import numpy as np
import matplotlib.pyplot as plt

from dataset_generator import cluster_generator, CENTERS, plot_points


class KMeans:
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
        self.intra_inertia_ = []
    
    def assign(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """给每个中心分配点

        Args:
            X (np.ndarray): 数据点
            centers (np.ndarray): 中心

        Returns:
            np.ndarray: 重新分配后的划分
        """
        n = X.shape[0]
        D = ((np.linalg.norm(X, axis=1, keepdims=True) ** 2).repeat(self.k, axis=1) - 
             2 * X @ centers.T +
             (np.linalg.norm(centers, axis=1) ** 2).reshape(1, -1).repeat(n, axis=0))
        P = np.argmin(D, axis=1)
        return P
    
    def update(self, X: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, float]:
        """更新中心

        Args:
            X (np.ndarray): 数据点
            P (np.ndarray): 给数据点分配的标签

        Returns:
            tuple[np.ndarray, float]: 一步更新后得到的新的中心和新的组内惯性
        """
        centers = []
        I_w = 0
        for label in range(self.k):
            mask = (P == label)
            C_i = X[mask].mean(axis=0)
            I_w += ((X[mask] - C_i) ** 2).sum()
            centers.append(C_i)
        return np.array(centers), I_w
    
    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """在数据点上训练

        Args:
            X (np.ndarray): 数据点

        Returns:
            tuple[np.ndarray, np.ndarray]: 训练后得到的中心和划分
        """
        # 初始化中心
        C = X[np.random.choice(X.shape[0], self.k)]
        last_P = None
        for _ in range(self.max_iter):
            P = self.assign(X, C)
            if last_P is not None and np.all(last_P == P):
                break
            C, I_w = self.update(X, P)
            self.intra_inertia_.append(I_w)
            last_P = P
                    
        return C, P


class OnlineKMeans:
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
        self.intra_inertia_ = []
    
    def fit(self, X: np.ndarray):
        np.random.shuffle(X)
        C = X[np.random.choice(X.shape[0], self.k)]
        counts = np.zeros((self.k,), dtype=int)
        last_P = None
        for ep in range(self.max_iter):
            P = []
            I_w = 0
            for x in X:
                D = ((x - C) ** 2).sum(axis=1)
                c_ids: int = np.argmin(D, axis=0)
                counts[c_ids] += 1
                C[c_ids] += (x - C[c_ids]) / counts[c_ids]
                P.append(c_ids)
            
            P = np.array(P)
            for label in range(self.k):
                mask = (P == label)
                I_w += ((X[mask] - C[label]) ** 2).sum()
            print(f"epoch {ep}  intra inertia: {I_w:.3f}")
            
            if last_P is not None and np.all(last_P == P):
                break
            last_P = P
        return C, P, counts
            

class FuzzyKMeans:
    def __init__(self, k: int, alpha: float = 2.0, max_iter: int = 100, epsilon: float = 1e-5, seed: int = 0):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
    
    def fit(self, X: np.ndarray):
        n, d = X.shape
        C = self.rng.random((n, self.k))
        C /= C.sum(axis=1, keepdims=True)  # 隶属度矩阵 (n, k)
        
        for ep in range(self.max_iter):
            W = C ** self.alpha  # (n, k)
            mu = W.T @ X / W.T.sum(axis=1, keepdims=True)  # 更新中心 (k, d) 
            D = ((np.linalg.norm(X, axis=1, keepdims=True) ** 2).repeat(self.k, axis=1) -   # (n, k)
                2 * X @ mu.T +
                (np.linalg.norm(mu, axis=1) ** 2).reshape(1, -1).repeat(n, axis=0))
            inv = D ** (-1 / (self.alpha - 1))
            C_new = inv / inv.sum(axis=1, keepdims=True)
            if np.max(np.abs(C_new - C)) < self.epsilon:
                break
            C = C_new
            
        W = C ** self.alpha
        mu = W.T @ X / W.T.sum(axis=1, keepdims=True)
            
        return C, mu

        
class DynamicClustering:
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, X: np.ndarray):
        kmeans = KMeans(self.k, self.max_iter)
        centers, partition = kmeans.fit(X)
        for label in range(self.k):
            mask = (partition == label)
            X_k = X[mask]
            mu_k = X_k.mean(axis=0)
            S_k = (X_k - mu_k).T @ (X_k - mu_k)  # (d, d) 类协方差矩阵
            M_k = np.linalg.inv(S_k)  # (d, d)
            D = (X_k - mu_k) @ M_k @ (X_k - mu_k).T - np.log(np.linalg.det(M_k))
            print(D.shape)
            
            
        

if __name__ == "__main__":
    points = cluster_generator(CENTERS)
    dyn = DynamicClustering(4, 100)
    dyn.fit(points)
    # plot_points(points, mu)
