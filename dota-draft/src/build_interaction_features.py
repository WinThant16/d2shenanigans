import pandas as pd
import numpy as np
import json

from itertools import combinations
from scipy.sparse import coo_matrix, save_npz


# ---- Step 1: Load + sort the matches ----
# Keep the same temporal ordering as the original feature builder.
# Smaller match IDs = older matches.
df = pd.read_parquet("data/raw/valid-matches.parquet")
print(f"Loaded {len(df)} matches")

df = df.sort_values("match_id").reset_index(drop=True)

# Keep each match as a normal dictionary so the encoder is easier to work with.
all_matches = df.to_dict("records")


# ---- Step 2: Get all hero IDs that actually show up ----
# Hero IDs aren't perfectly consecutive, so keep a list of the actual IDs.
hero_ids = sorted({
    int(hero_id)
    for match in all_matches
    for team_name in ("radiant_team", "dire_team")
    for hero_id in match[team_name]
})

# Still use max ID + 1 for the individual hero feature section.
# This means hero ID 2 can literally use column 2, hero ID 11 uses column 11, etc.
max_hero_id = max(hero_ids)
num_slots = max_hero_id + 1

print(f"Heroes in dataset: {len(hero_ids)}")
print(f"num_slots = {num_slots}")


# ---- Step 3: Build every possible hero pair ----
# combinations() gives each pair only once.
#
# Example:
# (2, 11) exists
# (11, 2) does NOT separately exist
#
# We use the same pair list for both synergy and matchup features.
hero_pairs = list(combinations(hero_ids, 2))

# Give every pair a compact index.
# Example:
# (1, 2) -> 0
# (1, 3) -> 1
# ...
pair_to_idx = {
    pair: i
    for i, pair in enumerate(hero_pairs)
}


# ---- Step 4: Decide where each feature group lives ----
#
# X looks roughly like:
#
# | individual heroes | synergy pairs | matchup pairs |
#
# The first 156 columns are the same hero features as the original model.
synergy_offset = num_slots

# Matchup features start after all 8,001 synergy features.
matchup_offset = synergy_offset + len(hero_pairs)

# 156 hero features
# + 8001 synergy features
# + 8001 matchup features
total_features = matchup_offset + len(hero_pairs)

print(f"Hero features: {num_slots}")
print(f"Synergy features: {len(hero_pairs)}")
print(f"Matchup features: {len(hero_pairs)}")
print(f"Total features: {total_features}")


# ---- Step 5: Encode ONE match ----
# Instead of making a dense 16k-wide row full of zeros,
# only return the features that are actually active.
#
# cols:
#   which feature columns are active
#
# values:
#   whether each feature is +1 or -1
#
# One normal 5v5 match should return exactly 55 features.
def encode_interaction_match(
    match,
    pair_to_idx,
    synergy_offset,
    matchup_offset,
):
    cols = []
    values = []

    radiant_team = match["radiant_team"]
    dire_team = match["dire_team"]


    # ---- Individual hero features ----
    # Same exact representation as the original model.
    #
    # Radiant hero = +1
    # Dire hero    = -1
    #
    # 5 + 5 = 10 features.
    for hero_id in radiant_team:
        cols.append(hero_id)
        values.append(1)

    for hero_id in dire_team:
        cols.append(hero_id)
        values.append(-1)


    # ---- Radiant synergy features ----
    # Five heroes have C(5, 2) = 10 unique teammate pairs.
    #
    # +1 means this pair is together on Radiant.
    for hero_a, hero_b in combinations(radiant_team, 2):
        pair = tuple(sorted((hero_a, hero_b)))
        pair_idx = pair_to_idx[pair]

        cols.append(synergy_offset + pair_idx)
        values.append(1)


    # ---- Dire synergy features ----
    # Same feature, but -1 because the pair is together on Dire.
    for hero_a, hero_b in combinations(dire_team, 2):
        pair = tuple(sorted((hero_a, hero_b)))
        pair_idx = pair_to_idx[pair]

        cols.append(synergy_offset + pair_idx)
        values.append(-1)


    # ---- Opponent matchup features ----
    # Every Radiant hero faces every Dire hero.
    #
    # 5 x 5 = 25 opponent pair features.
    for radiant_hero in radiant_team:
        for dire_hero in dire_team:
            # Always put the lower hero ID first so one matchup
            # only needs one column.
            pair = tuple(sorted((radiant_hero, dire_hero)))
            pair_idx = pair_to_idx[pair]

            # Use the lower-ID hero to define the sign.
            #
            # Example:
            # Axe = 2
            # Shadow Fiend = 11
            #
            # Axe Radiant / SF Dire -> +1
            # SF Radiant / Axe Dire -> -1
            #
            # That way a single learned coefficient can represent
            # the Axe vs SF relationship from either side.
            low_hero = pair[0]

            value = 1 if radiant_hero == low_hero else -1

            cols.append(matchup_offset + pair_idx)
            values.append(value)

    return cols, values


# ---- Step 6: Sanity check ONE match first ----
test_cols, test_values = encode_interaction_match(
    all_matches[0],
    pair_to_idx,
    synergy_offset,
    matchup_offset,
)

print("Total nonzero features:", len(test_values))

hero_count = sum(
    col < synergy_offset
    for col in test_cols
)

synergy_count = sum(
    synergy_offset <= col < matchup_offset
    for col in test_cols
)

matchup_count = sum(
    col >= matchup_offset
    for col in test_cols
)

print("Hero features:", hero_count)
print("Synergy features:", synergy_count)
print("Matchup features:", matchup_count)


# Make sure the Radiant/Dire signs are also behaving correctly.
hero_values = [
    value
    for col, value in zip(test_cols, test_values)
    if col < synergy_offset
]

synergy_values = [
    value
    for col, value in zip(test_cols, test_values)
    if synergy_offset <= col < matchup_offset
]

print("Hero +1:", hero_values.count(1))
print("Hero -1:", hero_values.count(-1))

print("Synergy +1:", synergy_values.count(1))
print("Synergy -1:", synergy_values.count(-1))


# ---- Step 7: Preallocate the full sparse matrix data ----
num_matches = len(all_matches)

# Every valid match has:
# 10 hero + 20 synergy + 25 matchup = 55 active features.
features_per_match = 55

num_nonzero = num_matches * features_per_match

# Row numbers are predictable because every match has exactly 55 entries.
#
# Basically:
# [0, 0, ... 55 times,
#  1, 1, ... 55 times,
#  2, 2, ...]
rows = np.repeat(
    np.arange(num_matches, dtype=np.int32),
    features_per_match
)

# We'll fill these while encoding every match.
cols = np.empty(num_nonzero, dtype=np.int32)
values = np.empty(num_nonzero, dtype=np.int8)


# ---- Step 8: Encode every match ----
for i, match in enumerate(all_matches):
    cols_i, values_i = encode_interaction_match(
        match,
        pair_to_idx,
        synergy_offset,
        matchup_offset,
    )

    # Dataset validation should guarantee this,
    # but keep the check here so bad data fails loudly.
    assert len(cols_i) == features_per_match
    assert len(values_i) == features_per_match

    # Each match owns exactly 55 spots in the preallocated arrays.
    start = i * features_per_match
    end = start + features_per_match

    cols[start:end] = cols_i
    values[start:end] = values_i


# ---- Step 9: Build the sparse feature matrix ----
# COO lets us build a matrix directly from:
# (value, row, column)
#
# Convert to CSR afterward since that's better for slicing + model training.
X = coo_matrix(
    (values, (rows, cols)),
    shape=(num_matches, total_features),
    dtype=np.int8,
).tocsr()


# ---- Step 10: Build the labels ----
# 1 = Radiant won
# 0 = Dire won
#
# Since df was sorted before all_matches was created,
# X[i] and y[i] still refer to the exact same match.
y = df["radiant_win"].astype(np.int8).values


# ---- Step 11: Final sanity checks ----
print("X.shape:", X.shape)
print("X.nnz:", X.nnz)
print("y.shape:", y.shape)


# ---- Step 12: Save the interaction dataset ----
save_npz(
    "data/processed/interaction_features.npz",
    X
)

np.save(
    "data/processed/interaction_labels.npy",
    y
)


# ---- Step 13: Save feature metadata ----
# We need this later so model coefficients can be translated
# back into actual hero pairs instead of meaningless column numbers.
metadata = {
    "hero_ids": hero_ids,
    "hero_pairs": hero_pairs,
    "num_slots": num_slots,
    "synergy_offset": synergy_offset,
    "matchup_offset": matchup_offset,
    "total_features": total_features,
}

with open(
    "data/processed/interaction_metadata.json",
    "w"
) as f:
    json.dump(metadata, f)

print("Saved interaction features, labels, and metadata")