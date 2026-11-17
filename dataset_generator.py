from ctypes import pointer

import numpy as np
import matplotlib.pyplot as plt


CENTERS = [(0, 0), (0.5, 5), (5.5, 0), (5.5, 5.5)]

def cluster_generator(centers: list[tuple], per_cluster_p: int = 50, seed: int = 0):
    def shear(X: np.ndarray, k: float):
        shear_M = np.array([
            [1, k],
            [0, 1],
        ])
        return X @ shear_M
    
    rng = np.random.default_rng(seed)
    centers = np.array(centers, dtype=float)
    all_p = []
    for i, c in enumerate(centers):
        c = c.reshape(1, -1)
        points = np.repeat(c, per_cluster_p, axis=0)
        points += rng.normal(scale=0.5, size=(points.shape[0], points.shape[1]))
        if i in [0, 2]:
            points = shear(points, 2)
        
        all_p.append(points)
        
    all_p = np.array(all_p).reshape(-1, 2)
        
    ext_points = np.array([
        [9.5, 9.5],
        [-5.0, -5.0]
    ])
    all_p = np.vstack([all_p, ext_points])
    np.random.shuffle(all_p)
    return all_p


def plot_points(points: np.ndarray, centers: np.ndarray = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(x=points[:, 0], y=points[:, 1], c="blue", label="point")
    if centers is not None:
        plt.scatter(x=centers[:, 0], y=centers[:, 1], c="red", label="center")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.legend()
    plt.show()
    plt.close()