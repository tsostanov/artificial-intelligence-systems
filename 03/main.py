import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. читаем данные
df = pd.read_csv("Student_Performance.csv")
print("Размер:", df.shape)
print(df.head())

# 2. базовая статистика и графики
print("\nСтатистика (числа):")
print(df.describe())

print("\nСтатистика (все):")
print(df.describe(include='all'))

df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# 3. подготовка данных
# заполняем пропуски
for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# категориальное в числа
df["Extracurricular Activities"] = (
    df["Extracurricular Activities"]
    .map({"Yes": 1, "No": 0})
    .astype(int)
)

# целевая колонка
target_col = "Performance Index"
y = df[target_col].values

# признаки без таргета
X_df = df.drop(columns=[target_col])

# нормализация
X = X_df.values.astype(float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# делим на train/test
np.random.seed(42)
idx = np.random.permutation(len(X_norm))
train_size = int(0.8 * len(X_norm))
train_idx = idx[:train_size]
test_idx = idx[train_size:]

X_train = X_norm[train_idx]
y_train = y[train_idx]
X_test = X_norm[test_idx]
y_test = y[test_idx]

# 5. своя линейная регрессия
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def linreg_fit(X, y):
    Xb = add_bias(X)
    # формула МНК: (X^T X)^(-1) X^T y
    XtX = Xb.T @ Xb
    XtY = Xb.T @ y
    w = np.linalg.inv(XtX) @ XtY
    return w

def linreg_predict(X, w):
    Xb = add_bias(X)
    return Xb @ w

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

results = {}

# 2 простых признака
m1_cols = ["Hours Studied", "Previous Scores"]
X_train_m1 = X_df.loc[train_idx, m1_cols].values.astype(float)
X_test_m1 = X_df.loc[test_idx, m1_cols].values.astype(float)

# отдельно нормируем
m1_mean = X_train_m1.mean(axis=0)
m1_std = X_train_m1.std(axis=0)
m1_std[m1_std == 0] = 1.0
X_train_m1 = (X_train_m1 - m1_mean) / m1_std
X_test_m1 = (X_test_m1 - m1_mean) / m1_std

w1 = linreg_fit(X_train_m1, y_train)
y_pred1 = linreg_predict(X_test_m1, w1)
r2_1 = r2_score(y_test, y_pred1)
results["Модель 1"] = r2_1
print("\nR2 модель 1:", r2_1)
print("Коэффициенты модель 1:")
print("  bias:", w1[0])
for name, coef in zip(m1_cols, w1[1:]):
    print(f"  {name}: {coef}")

# Все признаки
w2 = linreg_fit(X_train, y_train)
y_pred2 = linreg_predict(X_test, w2)
r2_2 = r2_score(y_test, y_pred2)
results["Модель 2"] = r2_2
print("\nR2 модель 2:", r2_2)
print("Коэффициенты модель 2:")
print("  bias:", w2[0])
for name, coef in zip(X_df.columns, w2[1:]):
    print(f"  {name}: {coef}")

# Все + синтетический признак
combo = df["Hours Studied"] * df["Sample Question Papers Practiced"]

X_df_syn = X_df.copy()
X_df_syn["Study_Papers_Combo"] = combo

X_syn = X_df_syn.values.astype(float)
Xs_mean = X_syn.mean(axis=0)
Xs_std = X_syn.std(axis=0)
Xs_std[Xs_std == 0] = 1.0
X_syn_norm = (X_syn - Xs_mean) / Xs_std

X_train_syn = X_syn_norm[train_idx]
X_test_syn = X_syn_norm[test_idx]

w3 = linreg_fit(X_train_syn, y_train)
y_pred3 = linreg_predict(X_test_syn, w3)
r2_3 = r2_score(y_test, y_pred3)
results["Модель 3"] = r2_3
print("\nR2 модель 3:", r2_3)
print("Коэффициенты модель 3:")
print("  bias:", w3[0])
for name, coef in zip(X_df_syn.columns, w3[1:]):
    print(f"  {name}: {coef}")

# вывод
print("\nИтог:")
for name, val in results.items():
    print(f"{name}: {val:.4f}")
