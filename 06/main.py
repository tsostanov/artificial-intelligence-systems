import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

y = df["Survived"].values.astype(int)

use_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X0 = df[use_cols].copy()

def train_test_split_manual(X, y, test_size=0.3, random_state=0):
    np.random.seed(random_state)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    test_len = int(len(y) * test_size)
    test_idx = idx[:test_len]
    train_idx = idx[test_len:]
    return X.iloc[train_idx].copy(), X.iloc[test_idx].copy(), y[train_idx].copy(), y[test_idx].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split_manual(X0, y, test_size=0.3, random_state=0)

def make_features_fit_transform(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    age_med = pd.to_numeric(X_train["Age"], errors="coerce").median()
    fare_med = pd.to_numeric(X_train["Fare"], errors="coerce").median()
    emb_mode = X_train["Embarked"].mode(dropna=True)
    emb_mode = emb_mode.iloc[0] if len(emb_mode) else "S"

    for X in (X_train, X_test):
        X["Age"] = pd.to_numeric(X["Age"], errors="coerce").fillna(age_med)
        X["Fare"] = pd.to_numeric(X["Fare"], errors="coerce").fillna(fare_med)
        X["Embarked"] = X["Embarked"].fillna(emb_mode)
        X["Sex"] = X["Sex"].fillna("missing")

    Xtr = pd.get_dummies(X_train, columns=["Sex", "Embarked"], drop_first=True)
    Xte = pd.get_dummies(X_test, columns=["Sex", "Embarked"], drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    mu = Xtr[num_cols].mean()
    sigma = Xtr[num_cols].std(ddof=0).replace(0, 1)

    Xtr[num_cols] = (Xtr[num_cols] - mu) / sigma
    Xte[num_cols] = (Xte[num_cols] - mu) / sigma

    return Xtr, Xte, num_cols

X_train_df, X_test_df, numeric_cols = make_features_fit_transform(X_train_raw, X_test_raw)

print("Размер train:", X_train_df.shape, "Размер test:", X_test_df.shape)
print("\nСтатистика по числовым признакам:")
stats = X_train_raw[["Pclass", "Age", "SibSp", "Parch", "Fare"]].describe().T
print(stats)

plt.figure()
cols = ["mean", "std", "25%", "50%", "75%", "max"]
M = stats[cols].values
M_scaled = (M - M.min(axis=0, keepdims=True)) / (M.max(axis=0, keepdims=True) - M.min(axis=0, keepdims=True) + 1e-12)
plt.imshow(M_scaled, aspect="auto", cmap="magma", vmin=0, vmax=1)
plt.yticks(np.arange(len(stats.index)), stats.index)
plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
plt.title("describe() (train, нормировка по столбцам)")
plt.colorbar(label="scaled per statistic")
plt.tight_layout()
plt.show()

df_num = X_train_raw[["Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
df_num["Age"] = pd.to_numeric(df_num["Age"], errors="coerce")
df_num["Fare"] = pd.to_numeric(df_num["Fare"], errors="coerce")
feat_list = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
fig, axes = plt.subplots(len(feat_list), 1, figsize=(8, 10), sharex=False)
for ax, col in zip(axes, feat_list):
    ax.boxplot(
        df_num[col].dropna(),
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#ffcc80", color="#e65100"),
        medianprops=dict(color="#bf360c"),
        whiskerprops=dict(color="#e65100"),
        capprops=dict(color="#e65100")
    )
    ax.set_title(f"{col}")
    ax.grid(True, axis="x", alpha=0.3)
plt.suptitle("Boxplot числовых признаков (train)")
plt.tight_layout()
plt.show()

plt.figure()
counts = pd.Series(y_train).value_counts().sort_index()
plt.bar(counts.index.astype(str), counts.values)
plt.title("Распределение классов Survived в train части")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True, axis="y")
plt.show()

X_train = X_train_df.values.astype(float)
X_test = X_test_df.values.astype(float)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

def confusion_counts(y_true, y_pred):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    d = TP + TN + FP + FN
    return (TP + TN) / d if d > 0 else 0.0

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

class LogisticRegressionScratch:
    def __init__(self, method="gd", learning_rate=0.1, n_iter=1000, l2=0.0, tol=1e-6):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol
        self.w = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def log_loss(self, y, p):
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        base = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if self.l2 > 0 and self.w is not None:
            base += (self.l2 / (2 * len(y))) * np.sum(self.w[1:] ** 2)
        return base

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.loss_history = []
        prev_loss = None

        if self.method == "gd":
            for _ in range(self.n_iter):
                p = self.sigmoid(X @ self.w)
                grad = (X.T @ (p - y)) / n
                if self.l2 > 0:
                    reg = (self.l2 / n) * self.w
                    reg[0] = 0.0
                    grad += reg
                self.w -= self.learning_rate * grad
                p = self.sigmoid(X @ self.w)
                loss = self.log_loss(y, p)
                self.loss_history.append(loss)
                if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        elif self.method == "newton":
            for _ in range(self.n_iter):
                p = self.sigmoid(X @ self.w)
                grad = (X.T @ (p - y)) / n
                if self.l2 > 0:
                    reg = (self.l2 / n) * self.w
                    reg[0] = 0.0
                    grad += reg
                R = p * (1 - p)
                XR = X * R[:, None]
                H = (X.T @ XR) / n
                if self.l2 > 0:
                    I = np.eye(d)
                    I[0, 0] = 0.0
                    H += (self.l2 / n) * I
                H = H + 1e-8 * np.eye(d)
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(H) @ grad
                self.w -= step
                p = self.sigmoid(X @ self.w)
                loss = self.log_loss(y, p)
                self.loss_history.append(loss)
                if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        else:
            raise ValueError("method must be 'gd' or 'newton'")
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def evaluate(method, lr, n_iter, l2=1e-2):
    m = LogisticRegressionScratch(method=method, learning_rate=lr, n_iter=n_iter, l2=l2)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return {
        "method": method,
        "lr": lr,
        "n_iter": n_iter,
        "accuracy": accuracy(y_test, pred),
        "precision": precision(y_test, pred),
        "recall": recall(y_test, pred),
        "f1": f1_score(y_test, pred),
        "final_loss": m.loss_history[-1] if len(m.loss_history) else np.nan,
        "loss_history": m.loss_history
    }

baseline = evaluate("gd", 0.3, 2000, l2=1e-2)
print("\nBaseline GD:")
print("Accuracy:", round(baseline["accuracy"], 4))
print("Precision:", round(baseline["precision"], 4))
print("Recall:", round(baseline["recall"], 4))
print("F1:", round(baseline["f1"], 4))

plt.figure()
plt.plot(baseline["loss_history"], marker=".")
plt.title("GD: log loss по итерациям")
plt.xlabel("iteration")
plt.ylabel("log loss")
plt.grid(True)
plt.show()

baseline_nt = evaluate("newton", 1.0, 10, l2=1e-2)
print("\nBaseline Newton:")
print("Accuracy:", round(baseline_nt["accuracy"], 4))
print("Precision:", round(baseline_nt["precision"], 4))
print("Recall:", round(baseline_nt["recall"], 4))
print("F1:", round(baseline_nt["f1"], 4))

plt.figure()
plt.plot(baseline_nt["loss_history"], marker=".")
plt.title("Newton: log loss по итерациям")
plt.xlabel("iteration")
plt.ylabel("log loss")
plt.grid(True)
plt.show()

learning_rates = [0.001, 0.01, 0.1, 0.3, 0.5]
iters_grid = [200, 1000, 3000]
newton_iters = [3, 5, 10]

results = []
for lr in learning_rates:
    for n_iter in iters_grid:
        results.append(evaluate("gd", lr, n_iter, l2=1e-2))
for n_iter in newton_iters:
    results.append(evaluate("newton", 1.0, n_iter, l2=1e-2))

rows = []
for r in results:
    rows.append({
        "method": r["method"],
        "lr": r["lr"],
        "n_iter": r["n_iter"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "final_loss": r["final_loss"]
    })

res_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)

print("\nТаблица результатов (по F1):")
print(res_df)

best = res_df.iloc[0]
print("\nЛучшее по F1:")
print(best)

plt.figure()
for lr in learning_rates:
    sub = res_df[(res_df["method"] == "gd") & (res_df["lr"] == lr)].sort_values("n_iter")
    plt.plot(sub["n_iter"].values, sub["f1"].values, marker=".", label=f"lr={lr}")
plt.title("GD: F1 при разных lr и n_iter")
plt.xlabel("n_iter")
plt.ylabel("F1")
plt.grid(True)
plt.legend()
plt.show()

def find_loss(method, lr, n_iter):
    for r in results:
        if r["method"] == method and float(r["lr"]) == float(lr) and int(r["n_iter"]) == int(n_iter):
            return r["loss_history"]
    return None

best_gd = res_df[res_df["method"] == "gd"].iloc[0]
best_nt = res_df[res_df["method"] == "newton"].iloc[0]

loss_gd = find_loss("gd", best_gd["lr"], best_gd["n_iter"])
loss_nt = find_loss("newton", 1.0, best_nt["n_iter"])

plt.figure()
if loss_gd is not None:
    plt.plot(loss_gd, marker=".", label=f"Best GD lr={best_gd['lr']} it={best_gd['n_iter']}")
if loss_nt is not None:
    plt.plot(loss_nt, marker=".", label=f"Best Newton it={best_nt['n_iter']}")
plt.title("Сходимость для лучших настроек")
plt.xlabel("iteration")
plt.ylabel("log loss")
plt.grid(True)
plt.legend()
plt.show()
