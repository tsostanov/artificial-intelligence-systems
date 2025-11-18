import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("WineDataset.csv")
print("Размер датасета:", df.shape)
print(df.head())

print("\nОписательная статистика:")
print(df.describe())

df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 4:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {1: "red", 2: "green", 3: "blue"}
    point_colors = df["Wine"].map(color_map)

    ax.scatter(
        df[num_cols[0]],
        df[num_cols[1]],
        df[num_cols[2]],
        c=point_colors,
        s=25
    )

    ax.set_xlabel(num_cols[0])
    ax.set_ylabel(num_cols[1])
    ax.set_zlabel(num_cols[2])
    plt.title("3D визуализация признаков")
    plt.show()


# обработка пропусков
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# целевая колонка и признаки
target_col = "Wine"
y = df[target_col].values.astype(int)
X_df = df.drop(columns=[target_col])

# нормализация
X = X_df.values.astype(float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# train / test
np.random.seed(69)
idx = np.random.permutation(len(X_norm))
train_size = int(0.8 * len(X_norm))
train_idx = idx[:train_size]
test_idx = idx[train_size:]

X_train = X_norm[train_idx]
y_train = y[train_idx]
X_test = X_norm[test_idx]
y_test = y[test_idx]

def euclidean_distance(x, X):
    diff = X - x
    return np.sqrt(np.sum(diff ** 2, axis=1))

def knn_predict_one(x, X_train, y_train, k):
    dists = euclidean_distance(x, X_train)
    nn_idx = np.argsort(dists)[:k]
    nn_labels = y_train[nn_idx]
    vals, counts = np.unique(nn_labels, return_counts=True)
    return vals[np.argmax(counts)]

def knn_predict(X, X_train, y_train, k):
    preds = []
    for i in range(X.shape[0]):
        preds.append(knn_predict_one(X[i], X_train, y_train, k))
    return np.array(preds)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        i = np.where(classes == yt)[0][0]
        j = np.where(classes == yp)[0][0]
        cm[i, j] += 1
    return cm, classes

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

results = {}

# модель 1: случайные признаки
np.random.seed(69)
all_feat_idx = list(range(X_train.shape[1]))
rand_feat_idx = np.random.choice(all_feat_idx, size=4, replace=False)

X_train_rand = X_train[:, rand_feat_idx]
X_test_rand = X_test[:, rand_feat_idx]

# модель 2: фиксированные признаки
fixed_cols = ["Alcohol", "Malic Acid", "Color intensity", "Proline"]
fixed_idx = [X_df.columns.get_loc(c) for c in fixed_cols]

X_train_fix = X_train[:, fixed_idx]
X_test_fix = X_test[:, fixed_idx]

ks = [3, 5, 20]

def inspect_neighbors(X_test_subset, X_train_subset, y_train_subset, sample_indices, ks_values, max_neighbors=20, label=""):
    title = "Анализ ближайших соседей"
    if label:
        title += f" ({label})"
    print(f"\n{title}:")
    for idx in sample_indices:
        if idx >= len(X_test_subset):
            print(f"  Индекс {idx} вне диапазона тестовой выборки.")
            continue
        sample = X_test_subset[idx]
        dists = euclidean_distance(sample, X_train_subset)
        order = np.argsort(dists)
        neighbor_labels = y_train_subset[order[:max_neighbors]]
        uniq, cnt = np.unique(neighbor_labels, return_counts=True)
        print(f"\nТестовый объект #{idx}")
        print("Первые метки соседей:", neighbor_labels.tolist())
        print("Распределение для k=20:", dict(zip(uniq.astype(int), cnt)))
        for k in ks_values:
            top_k = neighbor_labels[:k]
            uniq_k, cnt_k = np.unique(top_k, return_counts=True)
            print(f"  k={k}: {dict(zip(uniq_k.astype(int), cnt_k))}")

# sample_indices = range(20)
# inspect_neighbors(X_test_rand, X_train_rand, y_train, sample_indices, ks, label="модель 1 (случайные признаки)")
# inspect_neighbors(X_test_fix, X_train_fix, y_train, sample_indices, ks, label="модель 2 (фиксированные признаки)")

print("\nМодель 1 (случайные признаки):")
for k in ks:
    y_pred = knn_predict(X_test_rand, X_train_rand, y_train, k)
    acc = accuracy(y_test, y_pred)
    cm, _classes = confusion_matrix(y_test, y_pred)
    print(f"\nk = {k}, accuracy = {acc:.4f}")
    print("Матрица ошибок (истинный класс / предсказанный):")
    print(cm)
    results[f"model1_k{k}"] = acc

print("\nМодель 2 (фиксированные признаки):")
for k in ks:
    y_pred = knn_predict(X_test_fix, X_train_fix, y_train, k)
    acc = accuracy(y_test, y_pred)
    cm, _classes = confusion_matrix(y_test, y_pred)
    print(f"\nk = {k}, accuracy = {acc:.4f}")
    print("Матрица ошибок (истинный класс / предсказанный):")
    print(cm)
    results[f"model2_k{k}"] = acc

print("\nИтоговые точности:")
for name, acc in results.items():
    print(name, "-", f"{acc:.4f}")
