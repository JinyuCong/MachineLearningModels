import numpy as np
import matplotlib.pyplot as plt

def make_sigmoid_dataset(left: float, right: float, step: float):
    def sigmoid(xs: np.ndarray):
        return 1 / (np.exp(-xs) + 1)
    
    xs = np.arange(left, right, step).reshape(-1, 1)
    ys = sigmoid(xs)
    points = np.concatenate([xs, ys], axis=1)
    points += np.random.normal(-0.05, 0.05, size=points.shape)
    
    return points


def plot_points(X: np.ndarray, y_pred: np.ndarray):
    plt.scatter(X[:, 0], X[:, 1], c="blue")
    plt.scatter(X[:, 0], y_pred, c="red")
    plt.show()
    plt.close()


class LinearRegression:
    def __init__(self, lr: float):
        self.lr = lr
        self.a = 0.0
        self.b = 0.0
        
    def __call__(self, x: np.ndarray):
        return self.forward(x)    
        
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray):
        return ((y_pred - y_true) ** 2).mean()
    
    def forward(self, x: np.ndarray):
        return self.a * x + self.b
    
    def backward(self, grad_a: float, grad_b: float):
        
        self.a -= self.lr * grad_a
        self.b -= self.lr * grad_b


class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)
    
    def forward(self, x: np.ndarray):
        mask = (x >= 0).astype(int)
        return np.abs(x * mask)
    
    def backward(self, x: np.ndarray):
        return (x >= 0).astype(int)
    
    
class Softmax:
    def __init__(self):
        pass
    
    def __call__(self, x: np.ndarray, dim: int):
        return self.forward(x, dim=dim)
    
    def forward(self, x: np.ndarray, dim: int):
        return np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
    
    @staticmethod
    def backward(p: np.ndarray):
        return p * (1 - p)

def one_hot(x: np.ndarray, num_classes: int):
    one_hot_m = np.zeros((x.shape[0], num_classes))
    one_hot_m[np.arange(x.shape[0]), x] = 1
    return one_hot_m

class MoE:
    def __init__(self, num_experts: int = 3, lr: float = 0.01,
                 hidden: int = 32):
        self.num_experts = num_experts
        self.lr = lr
        self.experts = [LinearRegression(lr=lr) for _ in range(num_experts)]

        # gate
        self.gate_w1 = np.random.normal(size=(1, hidden))
        self.gate_b1 = np.random.normal(size=(1, hidden))
        self.relu = ReLU()
        self.gate_w2 = np.random.normal(size=(hidden, num_experts))
        self.gate_b2 = np.random.normal(size=(1, num_experts))
        self.softmax = Softmax()

        self.x = None
        self.h_pre = None
        self.h = None
        self.p = None
        self.k = None
        self.g = None
        self.f = None
        
    def __call__(self, x: np.ndarray):
        return self.forward(x)
        
    def forward(self, x: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): (B, 1)
        """
        self.x = x
        self.h_pre = x @ self.gate_w1 + self.gate_b1
        self.h = self.relu(self.h_pre)
        z = self.h @ self.gate_w2 + self.gate_b2  # (B, 3)
        self.p = self.softmax(z, dim=1)  # (B, 3)
        
        self.k = np.argmax(self.p, axis=1, keepdims=True)  # (B, 1) 每个数据点被选中的专家的下标, 反向传播到专家网络
        self.g = np.take_along_axis(self.p, self.k, axis=1)  # (B, 1) 门控概率最高的概率，反向传播到门控
        
        self.f = np.zeros_like(x)  # (B, 1) 每个数据点对应的专家的输出
        for i, expert_ids in enumerate(self.k):
            expert = self.experts[expert_ids.item()]
            f_e = expert(x[i])
            self.f[i] = f_e

        return self.g * self.f  # (B, 1) y_hat

    def backward(self, y_hat: np.ndarray, y: np.ndarray):
        B = y.shape[0]
        grad_y_hat = 2 * (y_hat - y) / B  # (B, 1)
        grad_f = grad_y_hat * self.g  # (B, 1)
        grad_g = grad_y_hat * self.f  # (B, 1)

        # 专家参数梯度：只用路由到自己的那部分样本
        for e in range(self.num_experts):
            expert_mask = self.k == e
            expert_grad_f = grad_f[expert_mask]
            expert_x = self.x[expert_mask]
            expert_grad_a = expert_grad_f @ expert_x
            expert_grad_b = expert_grad_f.sum().item()

            self.experts[e].backward(expert_grad_a, expert_grad_b)

        # 门控参数梯度：g -> p -> z，按 Top-1 化简后的完整 softmax 雅可比
        # dL/dz = grad_g * g * (onehot(k) - p)
        onehot = one_hot(self.k[:, 0], self.num_experts)  # (B, E)
        grad_z = grad_g * self.g * (onehot - self.p)  # (B, E)

        grad_w2 = self.h.T @ grad_z  # (H, E)
        grad_b2 = grad_z.sum(axis=0, keepdims=True)  # (1, E)
        grad_h = grad_z @ self.gate_w2.T  # (B, H)

        grad_h_pre = grad_h * self.relu.backward(self.h_pre)  # (B, H)
        grad_w1 = self.x.T @ grad_h_pre  # (1, H)
        grad_b1 = grad_h_pre.sum(axis=0, keepdims=True)  # (1, H)

        self.gate_w1 -= self.lr * grad_w1
        self.gate_b1 -= self.lr * grad_b1
        self.gate_w2 -= self.lr * grad_w2
        self.gate_b2 -= self.lr * grad_b2

        return ((y_hat - y) ** 2).mean()


def main():
    sigmoid_d = make_sigmoid_dataset(-10, 10, 0.01)
    
    x, y = sigmoid_d[:, 0].reshape(-1, 1), sigmoid_d[:, 1].reshape(-1, 1)  # (B, 1) (B, 1)
    
    moe = MoE(num_experts=3, lr=0.01)
    batch_size = 20
    
    for ep in range(100):
        all_loss = 0.0
        for i in range(0, x.shape[0], batch_size):
            batch_x = x[i : i+batch_size]
            batch_y = y[i : i+batch_size]
        
            y_hat = moe(batch_x)
            loss = moe.backward(y_hat, batch_y)
            all_loss += loss

        print(all_loss / batch_size)
    
    preds = moe.forward(x)
    plot_points(sigmoid_d, preds)
    
    

if __name__ == "__main__":
    main()
    
