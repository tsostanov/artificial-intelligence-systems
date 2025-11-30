import math
import numpy as np
import pandas as pd

df = pd.read_csv("DATA (1).csv")

df["successful"] = (df["GRADE"] >= 4).astype(int)
target_col = "successful"

y = df[target_col].values
X = df.drop(columns=["STUDENT ID", target_col])

for col in X.columns:
    X[col] = X[col].astype(str)

all_features = list(X.columns)
n_features = len(all_features)
k = int(math.sqrt(n_features))

np.random.seed(42)
selected_features = list(np.random.choice(all_features, size=k, replace=False))

print("Всего признаков:", n_features)
print("Выбрано признаков:", k)
print("Список выбранных признаков:", selected_features)

X = X[selected_features]


class TreeNode:
    def __init__(self,
                 feature_name=None,
                 children=None,
                 is_leaf=False,
                 prediction=None,
                 proba=None):
        self.feature_name = feature_name
        self.children = children or {}
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.proba = proba


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    _values, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def split_info(feature_values):
    n = len(feature_values)
    _, counts = np.unique(feature_values, return_counts=True)
    probs = counts / n
    return -np.sum(probs * np.log2(probs))


def gain_ratio(y, feature_values):
    H_before = entropy(y)
    n = len(y)
    unique_vals, counts = np.unique(feature_values, return_counts=True)

    cond_entropy = 0.0
    for v, count in zip(unique_vals, counts):
        mask = (feature_values == v)
        y_sub = y[mask]
        cond_entropy += (count / n) * entropy(y_sub)

    info_gain = H_before - cond_entropy
    si = split_info(feature_values)

    if si == 0:
        return 0.0
    return info_gain / si


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class DecisionTreeC45:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, list(X.columns), depth=0)

    def _build_tree(self, X, y, features, depth):
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            cls = unique_classes[0]
            proba = 1.0 if cls == 1 else 0.0
            return TreeNode(is_leaf=True, prediction=cls, proba=proba)

        if (len(features) == 0 or
            len(y) < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)):
            cls = majority_class(y)
            proba = np.mean(y)
            return TreeNode(is_leaf=True, prediction=cls, proba=proba)

        best_feature = None
        best_gr = -1.0

        for feature in features:
            gr = gain_ratio(y, X[feature].values)
            if gr > best_gr:
                best_gr = gr
                best_feature = feature

        if best_feature is None:
            cls = majority_class(y)
            proba = np.mean(y)
            return TreeNode(is_leaf=True, prediction=cls, proba=proba)

        node = TreeNode(feature_name=best_feature)

        feature_values = X[best_feature].values
        unique_vals = np.unique(feature_values)

        remaining_features = [f for f in features if f != best_feature]

        for v in unique_vals:
            mask = (feature_values == v)
            X_sub = X[mask]
            y_sub = y[mask]

            if len(y_sub) == 0:
                cls = majority_class(y)
                proba = np.mean(y)
                child = TreeNode(is_leaf=True, prediction=cls, proba=proba)
            else:
                child = self._build_tree(X_sub, y_sub,
                                         remaining_features, depth + 1)

            node.children[str(v)] = child

        node.prediction = majority_class(y)
        node.proba = np.mean(y)

        return node

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction

        val = str(x.get(node.feature_name))
        child = node.children.get(val)

        if child is None:
            return node.prediction

        return self._predict_one(x, child)

    def _predict_proba_one(self, x, node):
        if node.is_leaf:
            return node.proba

        val = str(x.get(node.feature_name))
        child = node.children.get(val)

        if child is None:
            return node.proba

        return self._predict_proba_one(x, child)

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            preds.append(self._predict_one(row.to_dict(), self.root))
        return np.array(preds)

    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            probs.append(self._predict_proba_one(row.to_dict(), self.root))
        return np.array(probs)


def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    test_len = int(len(y) * test_size)
    test_idx = idx[:test_len]
    train_idx = idx[test_len:]
    return (X.iloc[train_idx], X.iloc[test_idx],
            y[train_idx], y[test_idx])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeC45(max_depth=None, min_samples_split=2)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
y_score = tree.predict_proba(X_test)


def confusion_counts(y_true, y_pred):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")


def roc_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    TPR = [0.0]
    FPR = [0.0]
    TP = 0
    FP = 0
    prev_score = None

    for i in range(len(y_true)):
        if prev_score is None or y_score[i] != prev_score:
            TPR.append(TP / P if P > 0 else 0.0)
            FPR.append(FP / N if N > 0 else 0.0)
            prev_score = y_score[i]

        if y_true[i] == 1:
            TP += 1
        else:
            FP += 1

    TPR.append(1.0)
    FPR.append(1.0)

    return np.array(FPR), np.array(TPR)


def precision_recall_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    TP = 0
    FP = 0

    precisions = []
    recalls = []
    prev_score = None

    for i in range(len(y_true)):
        if y_true[i] == 1:
            TP += 1
        else:
            FP += 1

        if prev_score is None or y_score[i] != prev_score:
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / P if P > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            prev_score = y_score[i]

    return np.array(recalls), np.array(precisions)


def auc_trapezoid(x, y):
    area = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        height = (y[i] + y[i - 1]) / 2
        area += width * height
    return area


fpr, tpr = roc_curve_manual(y_test, y_score)
auc_roc = auc_trapezoid(fpr, tpr)

recalls, precisions = precision_recall_curve_manual(y_test, y_score)
order_pr = np.argsort(recalls)
recalls_sorted = recalls[order_pr]
precisions_sorted = precisions[order_pr]
auc_pr = auc_trapezoid(recalls_sorted, precisions_sorted)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")
