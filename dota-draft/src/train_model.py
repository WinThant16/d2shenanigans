import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import xgboost as xgb

# ---- Step 1: Load the saved features ----
# np.load returns a dict-like; pull X and y back out.
data = np.load("data/processed/features.npz")
X, y = data["X"], data["y"]
print(f"X {X.shape}, y {y.shape}")


# ---- Step 2: Three-way TEMPORAL split ----
# X and y are already sorted oldest->newest (in fixed build_features).
# So slicing by position IS slicing by time. No shuffling.
#   train = first 70%   (model fits on this)
#   val   = next 10%    (XGBoost early stopping uses this)
#   test  = last 20%    (touched once, for the final number)
# Compute two split points and make three slices of each array.
n = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

# three way slice, slicing numpy array with:    array[start:stop]
X_train = X[:train_end]                 #start of array up to train_end
X_val = X[train_end:val_end]            #middle
X_test = X[val_end:]                    #val_end to the end

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]
# print all six shapes to confirm they partition n with no overlap
print(f"X_train_shape {X_train.shape}, X_val_shape {X_val.shape}, X_test_shape {X_test.shape}")
print(f"y_train_shape {y_train.shape}, y_val_shape {y_val.shape}, y_test_shape {y_test.shape}")

# ---- Step 3: Retrain the logistic baseline on the SAME train set ----
# So the comparison is fair (same 70% train, same 20% test as XGBoost).
# This gives baseline number to beat.
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print(f"LogReg test acc: {logreg.score(X_test, y_test):.3f}")


# ---- Step 4: XGBoost with DEFAULTS first (no tuning, no early stopping) ----
# Set random_state for reproducibility.
# eval_metric='logloss' is a sensible classification metric.
# Fit on train only. This is the plain reference.
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
	max_depth=3,
	learning_rate=0.05,
	n_estimators=2000,
	early_stopping_rounds=100,
	colsample_bytree=0.67,
	subsample=0.9,
	min_child_weight=5
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print(f"XGB test acc: {xgb_model.score(X_test, y_test):.3f}")

# ---- Step 5: Compare accuracy, then calibration ----
# Print both test accuracies side by side, plus the always-radiant baseline
# (y_test.mean()) so all three sit together.
# Then rerun calibration table/curve for BOTH models on X_test.
# IDEA: XGB may edge accuracy up a point or two, but its probabilities
# may be LESS calibrated than logistic (boosting pushes toward 0/1).
# probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
# ... calibration_curve(y_test, probs_xgb, n_bins=10, strategy="quantile")
probs_logreg = logreg.predict_proba(X_test)[:, 1]
probs_xgb    = xgb_model.predict_proba(X_test)[:, 1]

true_freq, pred_freq = calibration_curve(y_test, probs_xgb, n_bins=10, strategy="quantile")

print(f"Baseline (always radiant): {y_test.mean():.3f}")
print(f"LogReg:  {logreg.score(X_test, y_test):.3f}")
print(f"XGBoost: {xgb_model.score(X_test, y_test):.3f}")
print(f"Best iteration: {xgb_model.best_iteration}")



# ---- Step 6 (LATER, second pass): add early stopping ----
# Once the plain fit works, add early stopping so we stop adding trees
# when val performance plateaus. Per current XGBoost, these go in the
# CONSTRUCTOR, not fit():
#   xgb.XGBClassifier(..., early_stopping_rounds=20)
#   .fit(X_train, y_train, eval_set=[(X_val, y_val)])


# plots
import matplotlib.pyplot as plt

# probs_logreg and probs_xgb
lr_true, lr_pred = calibration_curve(y_test, probs_logreg, n_bins=10, strategy="quantile")
xgb_true, xgb_pred = calibration_curve(y_test, probs_xgb, n_bins=10, strategy="quantile")

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.plot(lr_pred, lr_true, marker="o", label="Logistic Regression")
plt.plot(xgb_pred, xgb_true, marker="s", label="XGBoost")
plt.xlabel("Predicted win probability")
plt.ylabel("Actual win rate")
plt.title("Calibration: predicted vs actual")
plt.legend()
plt.savefig("reports/calibration.png", dpi=120, bbox_inches="tight")
plt.close()


# heroes plots
import json
with open("constants/heroes.json") as f:
    hero_data = json.load(f)
id_to_name = {int(k): v["localized_name"] for k, v in hero_data.items()}

coefs = logreg.coef_[0]                      # one weight per hero slot
order = np.argsort(coefs)                     # indices sorted low -> high
top = order[-10:][::-1]                        # 10 highest (strongest)
bottom = order[:10]                            # 10 lowest (weakest)

print("Strongest heroes (by coefficient):", top.tolist())
print("Weakest heroes:", bottom.tolist())

idx = np.concatenate([bottom, top[::-1]])
plt.figure(figsize=(8, 6))
plt.barh(range(len(idx)), coefs[idx])
plt.yticks(range(len(idx)), [id_to_name.get(i, f"hero {i}") for i in idx])
plt.xlabel("Contribution to team's win probability.")
plt.title("Most impactful heroes")
plt.axvline(0, color="gray", linewidth=0.8)
plt.savefig("reports/hero_coefficients.png", dpi=120, bbox_inches="tight")
plt.close()
