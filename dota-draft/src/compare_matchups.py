import json

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from analyze_matchups import build_pair_stats, rank_matchups


# ---- Settings ----
# Start with Axe.
# Later we can change this one ID to analyze any hero.
TARGET_HERO_ID = 2

# Ignore matchup stats with very small samples.
MIN_GAMES = 100

# We already selected this using the validation set
# in train_interaction_model.py.
BEST_C = 0.01


# ---- Step 1: Load interaction features + labels ----
X = load_npz(
    "data/processed/interaction_features.npz"
)

y = np.load(
    "data/processed/interaction_labels.npy"
)

print("X.shape:", X.shape)
print("y.shape:", y.shape)


# ---- Step 2: Load interaction feature metadata ----
# The model only sees columns like:
#
# 8291
#
# Metadata tells us that a column represents something like:
#
# Axe vs Shadow Fiend
with open("data/processed/interaction_metadata.json") as f:
    metadata = json.load(f)

assert X.shape[1] == metadata["total_features"]

hero_ids = [
    int(hero_id)
    for hero_id in metadata["hero_ids"]
]

# JSON changes tuples into lists.
# Convert the hero pairs back into tuples so we can use them
# as dictionary keys again.
hero_pairs = [
    tuple(pair)
    for pair in metadata["hero_pairs"]
]

pair_to_idx = {
    pair: i
    for i, pair in enumerate(hero_pairs)
}

matchup_offset = metadata["matchup_offset"]


# ---- Step 3: Load hero names ----
# The model uses hero IDs.
# heroes.json lets us print readable names instead.
with open("constants/heroes.json") as f:
    hero_data = json.load(f)

id_to_name = {
    int(hero_id): info["localized_name"]
    for hero_id, info in hero_data.items()
}

target_name = id_to_name[TARGET_HERO_ID]

print(f"\nTarget hero: {target_name}")


# ---- Step 4: Load + sort the raw matches ----
# This needs the SAME ordering used by the feature builders.
#
# Smaller match IDs = older matches.
#
# That means:
#
# df[i]
# X[i]
# y[i]
#
# should all refer to the same match.
df = pd.read_parquet(
    "data/raw/valid-matches.parquet"
)

df = df.sort_values(
    "match_id"
).reset_index(drop=True)

assert len(df) == len(y)
assert X.shape[0] == len(y)


# ---- Step 5: Make the final 85/15 split ----
# Earlier we used:
#
# 70% train
# 15% validation
# 15% test
#
# Validation helped us select C=0.01.
#
# Now that C is already selected, combine:
#
# train + validation = first 85%
#
# The final 15% stays untouched during fitting.
n = len(y)
fit_end = int(n * 0.85)

X_fit = X[:fit_end]
X_test = X[fit_end:]

y_fit = y[:fit_end]
y_test = y[fit_end:]

# Use the SAME first 85% for the raw statistics.
df_fit = df.iloc[:fit_end]

print("X_fit:", X_fit.shape)
print("X_test:", X_test.shape)


# ---- Step 6: Train the final interaction model ----
# No C tuning here.
#
# We already did that experiment.
# C=0.01 is now treated as fixed.
final_model = LogisticRegression(
    C=BEST_C,
    max_iter=1000,
)

final_model.fit(
    X_fit,
    y_fit,
)

test_accuracy = final_model.score(
    X_test,
    y_test,
)

print(
    f"\nFinal interaction test accuracy: "
    f"{test_accuracy:.4f}"
)


# ---- Step 7: Calculate raw matchup stats ----
# build_pair_stats() scans the first 85% of matches once.
#
# We mainly need:
#
# matchup_games
# matchup_wins
#
# but the existing helper also calculates synergy stats.
num_slots = metadata["num_slots"]

(
    matchup_games,
    matchup_wins,
    synergy_games,
    synergy_wins,
) = build_pair_stats(
    df_fit,
    num_slots,
)


# ---- Step 8: Rank raw matchups for our target hero ----
# Everything returned here is from Axe's perspective:
#
# games:
#   how many times Axe faced this opponent
#
# wins:
#   how many of those games Axe's team won
#
# win_rate:
#   wins / games
raw_matchups = rank_matchups(
    hero_id=TARGET_HERO_ID,
    hero_ids=hero_ids,
    matchup_games=matchup_games,
    matchup_wins=matchup_wins,
    min_games=MIN_GAMES,
)


# ---- Step 9: Get the adjusted coefficient for one matchup ----
# Important sign issue:
#
# The interaction feature builder stores ONE column per pair.
#
# Example:
#
# Axe = 2
# Shadow Fiend = 11
#
# stored pair:
#
# (2, 11)
#
# Positive direction always belongs to the LOWER hero ID.
#
# That's useful internally, but this script wants every coefficient
# from TARGET_HERO_ID's perspective.
#
# So if our target hero is the higher-ID hero,
# flip the coefficient.
def get_adjusted_matchup_coef(
    hero_id,
    opponent_id,
    model,
):
    pair = tuple(
        sorted((hero_id, opponent_id))
    )

    pair_idx = pair_to_idx[pair]

    matchup_col = (
        matchup_offset
        + pair_idx
    )

    coef = float(
        model.coef_[0][matchup_col]
    )

    # Stored coefficient points toward the lower-ID hero.
    #
    # If our target hero is NOT the lower-ID hero,
    # reverse the sign so the result is from our hero's perspective.
    if hero_id != pair[0]:
        coef = -coef

    return coef


# ---- Step 10: Join raw + adjusted results ----
results = []

for matchup in raw_matchups:
    opponent_id = matchup["opponent_id"]

    adjusted_coef = get_adjusted_matchup_coef(
        hero_id=TARGET_HERO_ID,
        opponent_id=opponent_id,
        model=final_model,
    )

    results.append({
        "opponent_id": opponent_id,
        "opponent": id_to_name.get(
            opponent_id,
            f"Hero {opponent_id}",
        ),
        "games": matchup["games"],
        "wins": matchup["wins"],
        "raw_win_rate": matchup["win_rate"],
        "adjusted_coef": adjusted_coef,

        # Logistic regression coefficients are log-odds.
        # exp(coef) converts that into an odds ratio.
        "odds_ratio": np.exp(adjusted_coef),
    })


results_df = pd.DataFrame(results)


# ---- Step 11: Rank by raw WR and adjusted effect ----
# Rank 1 = most favorable matchup for Axe.
results_df["raw_rank"] = (
    results_df["raw_win_rate"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

results_df["adjusted_rank"] = (
    results_df["adjusted_coef"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)


# Example:
#
# raw_rank = 50
# adjusted_rank = 10
#
# rank_shift = +40
#
# That means the matchup looks MUCH BETTER after adjustment.
#
# Negative values mean the raw matchup looked better
# than the adjusted model thinks it is.
results_df["rank_shift"] = (
    results_df["raw_rank"]
    - results_df["adjusted_rank"]
)


# ---- Step 12: Measure how similar the rankings are ----
# Spearman correlation compares ordering/ranking.
#
# +1:
# raw and adjusted rankings are basically the same
#
# 0:
# very little relationship
#
# -1:
# rankings are basically reversed
rank_corr, rank_p = spearmanr(
    results_df["raw_win_rate"],
    results_df["adjusted_coef"],
)

print(
    f"\nRaw vs adjusted rank correlation: "
    f"{rank_corr:.3f}"
)


# ---- Step 13: Helper for readable printing ----
# Keep the real DataFrame numeric.
# Only format a COPY when displaying it.
def print_matchup_table(
    title,
    table,
):
    display = table.copy()

    display["raw_win_rate"] = (
        display["raw_win_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display["adjusted_coef"] = (
        display["adjusted_coef"]
        .map(lambda x: f"{x:+.4f}")
    )

    display["odds_ratio"] = (
        display["odds_ratio"]
        .map(lambda x: f"{x:.3f}")
    )

    print(f"\n{title}")

    print(
        display[
            [
                "opponent",
                "games",
                "raw_win_rate",
                "adjusted_coef",
                "odds_ratio",
                "raw_rank",
                "adjusted_rank",
                "rank_shift",
            ]
        ].to_string(index=False)
    )


# ---- Step 14: Raw top 10 ----
# This is basically what a normal raw-stat matchup table would show.
raw_top = (
    results_df
    .sort_values(
        "raw_win_rate",
        ascending=False,
    )
    .head(10)
)

print_matchup_table(
    f"Top {target_name} matchups by RAW win rate",
    raw_top,
)


# ---- Step 15: Adjusted top 10 ----
# These are the strongest matchup-specific coefficients
# after the model accounts for the rest of the draft.
adjusted_top = (
    results_df
    .sort_values(
        "adjusted_coef",
        ascending=False,
    )
    .head(10)
)

print_matchup_table(
    f"Top {target_name} matchups by ADJUSTED effect",
    adjusted_top,
)


# ---- Step 16: Adjusted bottom 10 ----
adjusted_bottom = (
    results_df
    .sort_values(
        "adjusted_coef",
        ascending=True,
    )
    .head(10)
)

print_matchup_table(
    f"Worst {target_name} matchups by ADJUSTED effect",
    adjusted_bottom,
)


# ---- Step 17: Biggest ranking disagreements ----
# These are probably the most interesting cases.
#
# Example:
#
# raw rank = 3
# adjusted rank = 90
#
# That's exactly the kind of matchup where raw stats may
# be giving a misleading story.
results_df["abs_rank_shift"] = (
    results_df["rank_shift"].abs()
)

biggest_changes = (
    results_df
    .sort_values(
        "abs_rank_shift",
        ascending=False,
    )
    .head(10)
)

print_matchup_table(
    f"Biggest raw vs adjusted changes for {target_name}",
    biggest_changes,
)


# ---- Step 18: Save the full table ----
# Save the numeric version, not the formatted percentages.
#
# That lets us easily graph/analyze this later.
output_path = (
    f"reports/matchup_comparison_"
    f"{TARGET_HERO_ID}.csv"
)

results_df.sort_values(
    "adjusted_coef",
    ascending=False,
).to_csv(
    output_path,
    index=False,
)

print(f"\nSaved: {output_path}")