import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class Node:
    def __init__(self, value_counts=None):
        self.value_counts = value_counts or {}

        self.is_leaf = False
        self.prediction = None

        # split
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.sub_nodes = None


class CART:
    def __init__(self, criterion):
        self.criterion = criterion

    def _impurity(self, y: pd.DataFrame):
        n = len(y)
        counts = y.value_counts().values
        if self.criterion == "gini":
            G = 1 - np.sum((counts / n) ** 2)
        elif self.criterion == "entropy":
            G = -np.sum(((counts / n) * np.log2(counts / n)))

        return G

    def _best_split(self, X: pd.DataFrame, y: pd.DataFrame):
        data = []

        # trouver chaque valeur (value) dans chaque variable (feature) de X
        for feature in X.columns:
            col = X[feature]
            values = col.unique()

            # pour les variables quantitatives
            if values.dtype == float:
                values = np.sort(values)
                if len(values) == 1:
                    continue

                # le milieu entre les valeurs adjacentes est un seuil
                thresholds = (values[:-1] + values[1:]) / 2.0
                for t in thresholds:
                    mask = (col < t)  # masquer toutes les valeurs qui sont inférieur à ce seuil
                    y_left, y_right = y[mask], y[~mask]
                    X_left, X_right = X[mask], X[~mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    impurity_all = self._impurity(y)
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)

                    gain = impurity_all - len(y_left) / len(y) * impurity_left - len(y_right) / len(y) * impurity_right
                    data.append((feature, t, gain, (X_left, y_left, X_right, y_right)))

            else:
                # pour les variables qualitatives
                for value in values:
                    mask = (col == value)  # masquer tous les éléments de ce feature qui est égale à value
                    y_left, y_right = y[mask], y[~mask]
                    X_left, X_right = X[mask], X[~mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    impurity_all = self._impurity(y)
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)

                    # calcul gain, choisir le feature et le seuil qui fait le gain le plus haut
                    gain = impurity_all - len(y_left) / len(y) * impurity_left - len(y_right) / len(y) * impurity_right
                    data.append((feature, value, gain, (X_left, y_left, X_right, y_right)))

        if not data:
            return None, None, (None, None, None, None)

        best_set = max(data, key=lambda x: x[2])  # le meilleur split est quand le gain le plus haut
        best_feature, best_threshold, best_data = best_set[0], best_set[1], best_set[3]

        return best_feature, best_threshold, best_data

    def _tree_growing(self, X: pd.DataFrame, y: pd.DataFrame, depth=0):
        node = Node(y.value_counts().to_dict())

        if len(y.value_counts().values) == 1:
            node.is_leaf = True
            node.prediction = y.iloc[0]
            return node

        best_feature, best_threshold, (X_left, y_left, X_right, y_right) = self._best_split(X, y)

        if best_feature is None:
            node.is_leaf = True
            node.prediction = y.value_counts().idxmax()
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._tree_growing(X_left, y_left, depth + 1)
        node.right = self._tree_growing(X_right, y_right, depth + 1)

        return node

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self._tree_growing(X, y)

    def predict(self, X: pd.DataFrame, tree):
        if tree.is_leaf:
            return tree.prediction
        for feature in X.columns:
            if feature == tree.feature_index:  # si le feature_index d'un neud a trouvé
                val = X[feature].item()
                if isinstance(val, float):  # est variable quantitative
                    if val < tree.threshold:
                        return self.predict(X, tree.left)
                    else:
                        return self.predict(X, tree.right)

                else:  # est variable qualitative
                    if val == tree.threshold:
                        return self.predict(X, tree.left)
                    else:
                        return self.predict(X, tree.right)


class CHAID:
    def __init__(self):
        pass

    def _tschuprowT(self, contingence_table: np.array) -> float:
        h, w = contingence_table.shape
        total = np.sum(contingence_table)
        chi2 = 0

        for k in range(h):
            for l in range(w):
                line_total = np.sum(contingence_table[k, :])
                col_total = np.sum(contingence_table[:, l])

                obs = contingence_table[k, l]
                exp = line_total * (col_total / total)
                if exp > 0:
                    chi2 += (obs - exp) ** 2 / exp

        if h < 2 or w < 2 or total == 0:
            return 0.0
        den = np.sqrt((h - 1) * (w - 1))
        if den == 0:
            return 0.0

        T = np.sqrt((chi2 / total) / den)

        return T

    def _contingency_table(self, feature, target):
        return pd.crosstab(feature, target)

    def _best_split(self, X: pd.DataFrame, y: pd.DataFrame):
        y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        data = []
        for feature in X.columns:
            col = X[feature]

            # 这一列是quantitative的
            if pd.api.types.is_numeric_dtype(col):
                vals = np.sort(col.unique())
                if len(vals) < 2:
                    continue
                thresholds = (vals[:-1] + vals[1:]) / 2.0

                for t in thresholds:
                    mask = (col < t)
                    # 防止子集为空集
                    if mask.sum() == 0 or mask.sum() == len(mask):
                        continue

                    X_left, X_right = X[mask], X[~mask]
                    y_left, y_right = y[mask], y[~mask]

                    # 对每一个阈值计算一个contingence table
                    ct = self._contingency_table(mask.values, y.values)
                    ct_np = ct.to_numpy()

                    T = self._tschuprowT(ct_np)  # 计算对于这个阈值t的Tschuprow's T

                    sub_data = [(X_left, y_left), (X_right, y_right)]

                    data.append((feature, t, T, sub_data))

            # 这一列是qualitative的
            else:
                ct = self._contingency_table(col.values, y.values)
                ct_np = ct.to_numpy()

                T = self._tschuprowT(ct_np)  # 计算这个feature的Tschuprow's T

                sub_data = []
                for val in col.unique():
                    mask = (col == val)
                    if mask.sum() == 0:
                        continue
                    sub_X, sub_y = X[mask], y[mask]

                    sub_data.append((sub_X, sub_y))

                data.append((feature, None, T, sub_data))

        if not data:
            return None, None, None, 0.0

        best_split = max(data, key=lambda x: x[2])
        return best_split[0], best_split[1], best_split[3], best_split[2]

    def _tree_growing(self, X: pd.DataFrame, y: pd.DataFrame, depth=0,
                      max_depth=10, min_samples_split=2, min_T=1e-6):
        y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        node = Node(y.value_counts().to_dict())

        if len(y.value_counts().values) == 1:
            node.is_leaf = True
            node.prediction = y.iloc[0]
            return node

        # 停止条件
        if depth >= max_depth or len(y) < min_samples_split:
            node.is_leaf = True
            node.prediction = y.value_counts().idxmax()
            return node

        best_feature, best_threshold, best_subdata, best_T = self._best_split(X, y)

        if best_feature is None or best_subdata is None or not np.isfinite(best_T) or best_T < min_T:
            node.is_leaf = True
            node.prediction = y.value_counts().idxmax()
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.sub_nodes = [self._tree_growing(sub_X, sub_y, depth+1, max_depth, min_samples_split, min_T) for sub_X, sub_y in best_subdata]

        return node

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self._tree_growing(X, y)

    def predict(self, X: pd.DataFrame, tree):
        pass


if __name__ == '__main__':
    iris_dataset = load_iris()

    df = pd.read_excel("./weather_play_dataset.xlsx")

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    chaid = CHAID()
    node = chaid.fit(X, y)

