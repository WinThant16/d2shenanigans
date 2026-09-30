import numpy as np
from sklearn.linear_model import LogisticRegression


# ---- Step 1: Load the original hero-only features ----
data = np.load("data/processed/features.npz")

X = data["X"]
y = data["y"]

print("X.shape:", X.shape)
print("y.shape:", y.shape)

# ---- Step 2: Same 70/15/15 temporal split ----
n = len(y)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

# ---- Step 3: Tune hero-only regularization ----
c_values = [
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
]

best_model = None
best_c = None
best_val_acc = 0

for c in c_values:
    model = LogisticRegression(
        C=c,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)

    print(f"C={c}: validation acc = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c = c
        best_model = model

# ---- Step 4: Evaluate selected hero-only model ----
test_accuracy = best_model.score(X_test, y_test)

print(f"\nBest C: {best_c}")
print(f"Best validation acc: {best_val_acc:.4f}")
print(f"Hero-only Logistic test acc: {test_accuracy:.4f}")
print(f"Baseline always Radiant: {y_test.mean():.4f}")