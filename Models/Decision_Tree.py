from collections import Counter
import numpy as np

seed_value = 2023
np.random.seed(seed_value)

# 生成树
def _grow_tree(self, X, y, depth=0):
    n_samples, n_features = X.shape
    n_labels = len(set(y))

    if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    feature_indices = range(n_features)
    best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
    left_indices, right_indices = self._split(X[:, best_feature], best_threshold)

    left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
    right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
    return Node(best_feature, best_threshold, left, right)

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# 模型预测
def predict(self, X):
    return np.array([self._predict(inputs) for inputs in X])

# 预测节点类型
def _predict(self, inputs):
    node = self.tree_
    while node.value is None:
        if inputs[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def _most_common_label(self, y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

# 信息增益最大的特征和阈值
def _best_criteria(self, X, y, feature_indices):
    best_gain = -1
    split_index, split_threshold = None, None
    for i in feature_indices:
        column = X[:, i]
        thresholds = set(column)
        for threshold in thresholds:
            gain = self._information_gain(y, column, threshold)
            if gain > best_gain:
                best_gain = gain
                split_index = i
                split_threshold = threshold
    return split_index, split_threshold


# 返回标签
def _most_common_label(self, y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


# 精度
def score(self, y_pred, y):
    accuracy = (y_pred == y).sum() / len(y)
    return accuracy