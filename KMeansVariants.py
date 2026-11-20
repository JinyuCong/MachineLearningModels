import numpy as np
import matplotlib.pyplot as plt

from dataset.dataset_generator import round_cluster, ROUND_CENTERS, shear_cluster, SHEAR_CENTERS, plot_points


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
        C = X[np.random.choice(X.shape[0], self.k, replace=False)]
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


class KMedoids:
    """输入：相异度矩阵 $D\in\mathbb{R}^{n\times n}$（不需要坐标），$K$
    输出：medoid 的下标，标签

    步骤：

    1. 预先算好 $D$（可以是 $L_1$、欧氏距离或任意相异度）
    2. 随机选 $K$ 个下标作为 medoid
    3. 分配步：$z_i=\arg\min_kD[i,m_k]$，直接查表
    4. 更新步，两种写法任选：
        - 精确版：对每类，在类内找使 $\sum_{j\in P_k}D[i,j]$ 最小的 $i$（取 $D$ 的类内子矩阵按行求和再取 argmin）
        - 课件的随机版：随机抽一个非 medoid 点，若替换某个 medoid 后总准则下降就接受
        准则 $C=\sum_iD[i,m_{z_i}]$，不再下降就停止

    实现要点：代表元是下标而不是向量，这是和 k-means 的根本区别。
    精确版每类代价 $O(n_k^2)$，体会一下为什么慢。

    检查：在数据里加一个很远的离群点，对比 k-means 和 k-medoids 的中心偏移，
    体会稳健性。与 R 的 cluster::pam 或 sklearn_extra 的 KMedoids 对照
    （注意它们用 BUILD+SWAP，结果可能不完全一样，比较准则值即可）。
    """
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, X: np.ndarray):
        n, d = X.shape
        square_X = (X ** 2).sum(axis=-1, keepdims=True).repeat(n, axis=1)
        D = np.sqrt((square_X - 2 * X @ X.T + square_X.T) + 1e-12)
        medoids = np.random.choice(n, self.k, replace=False)
        all_ids = np.arange(n)
        C = None
        
        for ep in range(self.max_iter):
            z = medoids[np.argmin(D[:, medoids], axis=1)]  # 所有点对应的最近的medoids的下标
            new_C = 0
            for i, med_idx in enumerate(medoids):
                med_mask = (z == med_idx)  # 属于这个medoid的点的掩码
                med_ids = all_ids[med_mask]   # 属于这个medoid的点的index
                sub_D = D[med_mask, :][:, med_mask]  # 从D中找出属于这个medoid的点之间的距离
                new_C += sub_D.sum() / 2
                new_med = med_ids[np.argmin(sub_D.sum(axis=-1))]  # 这类中所有点到其它点之间的距离的最小的点的index
                medoids[i] = new_med
                
            if C is not None and new_C >= C:
                break
            C = new_C
        
        return medoids


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
        n, d = X.shape
        kmeans = KMeans(self.k, self.max_iter)
        _, partition = kmeans.fit(X)
        D = np.zeros((n, self.k))
        
        for ep in range(self.max_iter):
            for label in range(self.k):
                mask = (partition == label)
                X_k = X[mask]
                mu_k = X_k.mean(axis=0)
                S_k = (X_k - mu_k).T @ (X_k - mu_k)  # (d, d) 类协方差矩阵
                M_k = np.linalg.inv(S_k)  # (d, d)
                
                D_k = ((X - mu_k) @ M_k * (X - mu_k)).sum(axis=-1) - np.log(np.linalg.det(M_k))
                D[:, label] = D_k
            
            new_partition = np.argmin(D, axis=-1)
            if np.array_equal(new_partition, partition):
                break
            partition = new_partition
        
        centers = np.array([X[partition == l].mean(axis=0) for l in range(self.k)])
        
        return centers, partition


class SphericalKMeans:
    """输入：$X\in\mathbb{R}^{n\times p}$，要聚的是列
    输出：$p$ 个变量的标签，$L$ 个单位中心

    步骤：

    1. 预处理：每列中心化，再除以列的范数，使每列单位长度。
    之后操作的对象是 $V=X^\top$（$p\times n$），每行是一个变量
    2. 随机划分变量，计算各类均值并归一化为单位向量 $c_\ell$
    3. 分配：计算 $V c_\ell$（即余弦相似度），取 argmax
    4. 更新：类内均值再除以其范数
    5. 准则 $\sum_\ell\sum_{j\in Q_\ell}v_j^\top c_\ell$ 应单调不减

    实现要点：注意是 max 不是 min；均值向量为零向量时无法归一化（极少见，空类同样处理）。
    课件的一般情形要用相关系数的绝对值，这时可以先把每个变量乘以 $\pm1$ 使它与所在类中心正相关，
    或者直接把相似度改成 $|v_j^\top c_\ell|$。

    检查：构造几组相关变量（例如 $x_2=x_1+$ 噪声，$x_3=-x_1+$ 噪声），看它们是否被分到一类。
    用 abs(np.corrcoef(X.T)) 的热图按聚类结果重排，应出现块状结构。
    """
    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.max_iter = max_iter
        
    def fit(X: np.ndarray):
        pass

if __name__ == "__main__":
    points = round_cluster(ROUND_CENTERS)
    k_medoids = KMedoids(4, 100)
    medoids = k_medoids.fit(points)
    
    plot_points(points, medoids)  
            
        


