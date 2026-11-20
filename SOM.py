import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class SOM:
    def __init__(self, K1: int, K2: int, max_iter: int = 2000, epsilon: float = 0.5, sigma: float = None):
        self.K1 = K1
        self.K2 = K2
        self.K = K1 * K2
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.sigma = sigma if sigma is not None else max(K1, K2) / 2
    
    def fit(self, X: np.ndarray):
        n, d = X.shape
        mu = X[np.random.choice(n, self.K, replace=False)]  # 从数据中随机抽 K 个点
        grid = mu.reshape(self.K1, self.K2, d)  # 给每个原型点分配一个网格点
        
        for t in range(self.max_iter):
            # 步长衰减
            epsilon_t = self.epsilon * (0.005 / self.epsilon) ** (t / self.max_iter)
            sigma_t = self.sigma * (0.5 / self.sigma) ** (t / self.max_iter)
            x = X[np.random.choice(n, 1)]
            
            # 在输入空间中找离 x 最近的原型，称为获胜节点（BMU，Best Matching Unit）
            dist = ((grid - x) ** 2).sum(axis=-1)
            min_index = np.argmin(dist)
            c = np.array(np.unravel_index(min_index, dist.shape))
            
            # 找出bmu对应的网格坐标
            for i in range(self.K1):
                for j in range(self.K2):
                    c_k = np.array([i, j])
                    h = np.exp(-((c_k - c) ** 2).sum() / (2 * sigma_t ** 2))  # 在网格上计算每个节点与获胜节点的距离，通过邻域函数转成权重
                    grid[i, j] += epsilon_t * h * (x.reshape(-1) - grid[i, j])

        return grid


def u_matrix(grid: np.ndarray) -> np.ndarray:
    """U-matrix：每个节点的原型与其网格上（上下左右）相邻节点原型在输入空间中的平均距离

    Args:
        grid (np.ndarray): 训练好的原型 (K1, K2, d)

    Returns:
        np.ndarray: (K1, K2)，值越大说明该节点附近原型越分散（类间边界），越小说明越密集（类内部）
    """
    K1, K2, _ = grid.shape
    U = np.zeros((K1, K2))
    for i in range(K1):
        for j in range(K2):
            dists = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < K1 and 0 <= nj < K2:
                    dists.append(np.linalg.norm(grid[i, j] - grid[ni, nj]))
            U[i, j] = np.mean(dists)
    return U


def plot_u_matrix(U: np.ndarray, title: str = "Neighbour distance plot"):
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    K1, K2 = U.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    # 每个节点画一个圆，第 i 行第 j 列画在 (j, i) 处
    circles = [Circle((j, i), 0.45) for i in range(K1) for j in range(K2)]
    pc = PatchCollection(circles, cmap="jet", edgecolor="black", linewidth=0.8)
    pc.set_array(U.ravel())
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, location="left", fraction=0.05, pad=0.02)

    ax.set_xlim(-0.6, K2 - 0.4)
    ax.set_ylim(-0.6, K1 - 0.4)
    ax.invert_yaxis()  # 第 0 行在最上面
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    iris = datasets.load_iris()
    iris_data, iris_target = iris["data"], iris["target"]
    normalized_iris = (iris_data - iris_data.mean(axis=0)) / iris_data.std(axis=0)
    som = SOM(10, 10, max_iter=5000)
    grid = som.fit(normalized_iris)
    plot_u_matrix(u_matrix(grid))

    