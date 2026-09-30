import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz
import json

# ---- Step 1: Load interaction features + labels ----
X = load_npz(
    "data/processed/interaction_features.npz"
)

y = np.load(
    "data/processed/interaction_labels.npy"
)

print("X.shape:", X.shape)
print("y.shape:", y.shape)

# ---- Step 1b: Load feature metadata ----
# This tells us exactly which columns belong to which hero pairs.
# Use the mapping saved by the feature builder instead of rebuilding it here.
with open("data/processed/interaction_metadata.json") as f:
    metadata = json.load(f)

assert X.shape[1] == metadata["total_features"]

# JSON turns tuples into lists, so turn each pair back into a tuple.
hero_pairs = [
    tuple(pair)
    for pair in metadata["hero_pairs"]
]

pair_to_idx = {
    pair: i
    for i, pair in enumerate(hero_pairs)
}

synergy_offset = metadata["synergy_offset"]
matchup_offset = metadata["matchup_offset"]




# ---- Step 2: Keep the same temporal 70/15/15 split ----
# The feature builder already sorted matches from older -> newer,
# so slicing like this gives us a time-based split.
n = len(y)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# ---- Step 3: Tune regularization on validation data ----
# Smaller C = stronger regularization.
# With thousands of pair features, stronger regularization might help
# keep noisy matchup/synergy effects from getting too large.
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

# ---- Step 4: Evaluate the best model on the newest 15% ----
# We picked best_c using ONLY the validation set.
# Now test that selected model once on the held-out test set.
test_accuracy = best_model.score(X_test, y_test)

print(f"\nBest C: {best_c}")
print(f"Best validation acc: {best_val_acc:.4f}")
print(f"Interaction Logistic test acc: {test_accuracy:.4f}")
print(f"Baseline always Radiant: {y_test.mean():.4f}")

mean_abs_coef = np.mean(np.abs(best_model.coef_))
max_abs_coef = np.max(np.abs(best_model.coef_))

print(
    f"mean |coef| = {mean_abs_coef:.4f}, "
    f"max |coef| = {max_abs_coef:.4f}"
)


# ---- Step 5: Inspect Axe vs Shadow Fiend ----
axe_id = 2
shadow_fiend_id = 11

pair = tuple(sorted((axe_id, shadow_fiend_id)))

pair_idx = pair_to_idx[pair]

# Matchup columns start after the hero + synergy feature sections.
matchup_col = matchup_offset + pair_idx

coef = best_model.coef_[0][matchup_col]
odds_ratio = np.exp(coef)

print("\nAxe vs Shadow Fiend")
print("Matchup feature column:", matchup_col)
print(f"Adjusted coefficient: {coef:.4f}")
print(f"Odds ratio: {odds_ratio:.4f}")