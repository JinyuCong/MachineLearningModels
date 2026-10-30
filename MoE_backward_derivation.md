# MoE 反向传播推导（Top-1 路由）

## 0. 符号与前向定义

- 输入 $x \in \mathbb{R}^{B \times 1}$，标签 $y \in \mathbb{R}^{B \times 1}$
- 门控第一层：$W_1$ (`gate_l1`), $b_1$，隐藏维度 $H$

$$
h_{\text{pre}} = x W_1 + b_1 \in \mathbb{R}^{B \times H}, \qquad h = \mathrm{ReLU}(h_{\text{pre}})
$$

- 门控第二层：$W_2$ (`gate_l2`), $b_2$，专家数 $E$

$$
z = h W_2 + b_2 \in \mathbb{R}^{B \times E}
$$

- Softmax 概率：

$$
p_j = \frac{e^{z_j}}{\sum_{l=1}^{E} e^{z_l}}, \qquad p \in \mathbb{R}^{B \times E}
$$

- 路由选择（Top-1）：

$$
k = \arg\max_j p_j, \qquad g = p_k
$$

- 第 $e$ 个专家（线性回归）：

$$
f_e(x) = a_e x + b_e
$$

- 被选中专家的输出与最终预测：

$$
f = f_k(x), \qquad \hat{y} = g \cdot f
$$

- 损失（MSE）：

$$
\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}(\hat{y}_i - y_i)^2
$$

反向传播就是从 $\mathcal{L}$ 开始，沿着上面定义反方向把梯度链回去。

---

## 1. Loss → 预测值

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{2}{B}(\hat{y} - y)
$$

## 2. 预测值 → 门控输出 $g$ 和专家输出 $f$

因为 $\hat{y} = g \cdot f$（逐元素乘）：

$$
\frac{\partial \mathcal{L}}{\partial f} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot g,
\qquad
\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot f
$$

## 3. 专家参数的梯度（只用路由到该专家的样本）

设 $S_e = \{ i : k_i = e \}$ 为被路由到专家 $e$ 的样本集合。因为 $f_e(x) = a_e x + b_e$：

$$
\frac{\partial \mathcal{L}}{\partial a_e} = \sum_{i \in S_e} \frac{\partial \mathcal{L}}{\partial f_i}\, x_i,
\qquad
\frac{\partial \mathcal{L}}{\partial b_e} = \sum_{i \in S_e} \frac{\partial \mathcal{L}}{\partial f_i}
$$

参数更新：

$$
a_e \leftarrow a_e - \eta \frac{\partial \mathcal{L}}{\partial a_e}, \qquad
b_e \leftarrow b_e - \eta \frac{\partial \mathcal{L}}{\partial b_e}
$$

## 4. $g$ → softmax 概率 $p$

$g$ 只是对 $p$ 的一次“取值”操作：$g = p_k = \sum_j \mathbb{1}[j=k]\, p_j$，所以

$$
\frac{\partial \mathcal{L}}{\partial p_j} = \frac{\partial \mathcal{L}}{\partial g} \cdot \mathbb{1}[j = k]
$$

即梯度只流向被选中的那一列，其余列为 0。

## 5. Softmax 的完整雅可比（这是原代码写错的地方）

对任意 softmax，完整的雅可比是：

$$
\frac{\partial p_i}{\partial z_m} = p_i \left( \delta_{im} - p_m \right),
\qquad
\delta_{im} = \begin{cases} 1 & i = m \\ 0 & i \ne m \end{cases}
$$

**注意：$\dfrac{\partial p}{\partial z} \ne p(1-p)$，`p * (1 - p)` 只是 $i=m$ 时的对角项，漏掉了 $i \ne m$ 时的 $-p_i p_m$ 项。**

利用链式法则，并结合第 4 步（只有 $j=k$ 处梯度非零）：

$$
\frac{\partial \mathcal{L}}{\partial z_m}
= \sum_{j=1}^{E} \frac{\partial \mathcal{L}}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_m}
= \frac{\partial \mathcal{L}}{\partial g} \cdot p_k \left( \mathbb{1}[k = m] - p_m \right)
$$

代入 $g = p_k$，写成向量形式（$e_k$ 是第 $k$ 个位置为 1 的 one-hot 向量）：

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial g} \cdot g \cdot \left( e_k - p \right) \in \mathbb{R}^{B \times E}
$$

## 6. $z$ → 门控第二层参数 $W_2, b_2$，以及 $h$

$z = h W_2 + b_2$：

$$
\frac{\partial \mathcal{L}}{\partial W_2} = h^{\top} \frac{\partial \mathcal{L}}{\partial z} \in \mathbb{R}^{H \times E},
\qquad
\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^{B} \frac{\partial \mathcal{L}}{\partial z_i} \in \mathbb{R}^{1 \times E}
$$

$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial z} W_2^{\top} \in \mathbb{R}^{B \times H}
$$

## 7. $h$ → $h_{\text{pre}}$（ReLU 反传）

$$
\frac{\partial \mathcal{L}}{\partial h_{\text{pre}}} = \frac{\partial \mathcal{L}}{\partial h} \odot \mathbb{1}[h_{\text{pre}} > 0]
$$

（$\odot$ 表示逐元素乘）

## 8. $h_{\text{pre}}$ → 门控第一层参数 $W_1, b_1$

$h_{\text{pre}} = x W_1 + b_1$：

$$
\frac{\partial \mathcal{L}}{\partial W_1} = x^{\top} \frac{\partial \mathcal{L}}{\partial h_{\text{pre}}} \in \mathbb{R}^{1 \times H},
\qquad
\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^{B} \frac{\partial \mathcal{L}}{\partial h_{\text{pre},i}} \in \mathbb{R}^{1 \times H}
$$

参数更新（$W_1, b_1, W_2, b_2$ 同理）：

$$
W_1 \leftarrow W_1 - \eta \frac{\partial \mathcal{L}}{\partial W_1}, \qquad
b_1 \leftarrow b_1 - \eta \frac{\partial \mathcal{L}}{\partial b_1}
$$

---

## 附：各基础模块的求导公式（通用形式，不限于 Top-1）

### MSE

$$
\mathcal{L} = \frac{1}{B}\sum_i (\hat{y}_i - y_i)^2
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{B}(\hat{y}_i - y_i)
$$

### 线性层 $y = ax + b$

$$
\frac{\partial \mathcal{L}}{\partial a} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i} x_i,
\qquad
\frac{\partial \mathcal{L}}{\partial b} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i}
$$

### ReLU

$$
\mathrm{ReLU}(x) = \max(0, x)
\quad\Longrightarrow\quad
\frac{\partial\, \mathrm{ReLU}(x)}{\partial x} = \mathbb{1}[x > 0]
$$

### Softmax（完整雅可比，适用于 soft-MoE 或任意下游组合）

$$
p_i = \frac{e^{z_i}}{\sum_l e^{z_l}},
\qquad
\frac{\partial p_i}{\partial z_m} = p_i(\delta_{im} - p_m)
$$

若下游用交叉熵损失（分类任务），两者结合会出现经典化简：

$$
\mathcal{L} = -\sum_i y_i \log p_i
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}}{\partial z} = p - y
$$

但本项目是 MoE 路由 + MSE，不是这种化简场景，所以必须按第 5 步的完整雅可比来算，不能抄这个化简结果。
