import json

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from analyze_matchups import build_pair_stats, rank_matchups


# ============================================================
# SETTINGS
# ============================================================

# Same minimum sample size we used for Axe.
MIN_GAMES = 100

# Already selected using the validation set.
BEST_C = 0.01

# restrict summary to heroes with enough qualifying opponents:
MIN_MATCHUPS_PER_HERO = 100
# ============================================================
# STEP 1: Load interaction features + labels
# ============================================================
#
# X:
#   one row per match
#   hero + synergy + matchup interaction features
#
# y:
#   1 = Radiant win
#   0 = Dire win

X = load_npz(
    "data/processed/interaction_features.npz"
)

y = np.load(
    "data/processed/interaction_labels.npy"
)

print("X.shape:", X.shape)
print("y.shape:", y.shape)


# ============================================================
# STEP 2: Load feature metadata
# ============================================================
#
# We need this to translate a hero pair such as:
#
# (2, 11)
#
# into the correct matchup-feature column.

with open("data/processed/interaction_metadata.json") as f:
    metadata = json.load(f)

assert X.shape[1] == metadata["total_features"]

hero_ids = [
    int(hero_id)
    for hero_id in metadata["hero_ids"]
]

hero_pairs = [
    tuple(pair)
    for pair in metadata["hero_pairs"]
]

pair_to_idx = {
    pair: i
    for i, pair in enumerate(hero_pairs)
}

num_slots = metadata["num_slots"]
matchup_offset = metadata["matchup_offset"]


# ============================================================
# STEP 3: Load hero names
# ============================================================

with open("constants/heroes.json") as f:
    hero_data = json.load(f)

id_to_name = {
    int(hero_id): info["localized_name"]
    for hero_id, info in hero_data.items()
}

print(f"Heroes in dataset: {len(hero_ids)}")


# ============================================================
# STEP 4: Load raw matches in the SAME temporal order
# ============================================================
#
# build_interaction_features.py sorted by match_id.
#
# Do exactly the same thing here so:
#
# df[i]
# X[i]
# y[i]
#
# all correspond to the same historical match.

df = pd.read_parquet(
    "data/raw/valid-matches.parquet"
)

df = df.sort_values(
    "match_id"
).reset_index(drop=True)

assert len(df) == len(y)
assert X.shape[0] == len(y)


# ============================================================
# STEP 5: Final 85 / 15 split
# ============================================================
#
# Earlier:
#
# 70% train
# 15% validation
# 15% test
#
# Validation selected:
#
# C = 0.01
#
# So now:
#
# first 85% -> fit final model + calculate raw stats
# last 15%  -> evaluate model

n = len(y)
fit_end = int(n * 0.85)

X_fit = X[:fit_end]
X_test = X[fit_end:]

y_fit = y[:fit_end]
y_test = y[fit_end:]

df_fit = df.iloc[:fit_end]

print("X_fit:", X_fit.shape)
print("X_test:", X_test.shape)


# ============================================================
# STEP 6: Train ONE final interaction model
# ============================================================
#
# Important:
#
# We do NOT train a separate model for every hero.
#
# One model contains coefficients for every hero,
# synergy pair, and matchup pair.

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


# ============================================================
# STEP 7: Build raw pair statistics ONCE
# ============================================================
#
# Again, we do NOT run this separately for every hero.
#
# This one pass gives us:
#
# matchup_games[a][b]
# matchup_wins[a][b]
#
# for every hero pair.

print("\nBuilding raw pair statistics...")

(
    matchup_games,
    matchup_wins,
    synergy_games,
    synergy_wins,
) = build_pair_stats(
    df_fit,
    num_slots,
)

print("Finished raw pair statistics.")


# ============================================================
# STEP 8: Helper for adjusted matchup coefficient
# ============================================================
#
# Same logic used in compare_matchups.py.
#
# After this function:
#
# positive coefficient
#     -> favors hero_id
#
# negative coefficient
#     -> disfavors hero_id
#
# The sign flip is necessary because the feature builder stores
# each pair from the lower-ID hero's direction.

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

    # The stored coefficient points toward the lower-ID hero.
    # Flip it when hero_id is the higher-ID hero.
    if hero_id != pair[0]:
        coef = -coef

    return coef


# ============================================================
# STEP 9: Compare raw vs adjusted rankings for ONE hero
# ============================================================
#
# This is basically the core logic from compare_matchups.py,
# turned into a reusable function.
#
# It returns ONE summary row for a hero.

def analyze_hero(hero_id):
    raw_matchups = rank_matchups(
        hero_id=hero_id,
        hero_ids=hero_ids,
        matchup_games=matchup_games,
        matchup_wins=matchup_wins,
        min_games=MIN_GAMES,
    )

    rows = []

    for matchup in raw_matchups:
        opponent_id = matchup["opponent_id"]

        adjusted_coef = get_adjusted_matchup_coef(
            hero_id=hero_id,
            opponent_id=opponent_id,
            model=final_model,
        )

        rows.append({
            "opponent_id": opponent_id,
            "raw_win_rate": matchup["win_rate"],
            "adjusted_coef": adjusted_coef,
        })


    matchup_df = pd.DataFrame(rows)

    # We need enough opponents to meaningfully compare rankings.
    if len(matchup_df) < 2:
        return None


    # ---- Raw ranking ----
    matchup_df["raw_rank"] = (
        matchup_df["raw_win_rate"]
        .rank(
            ascending=False,
            method="min",
        )
    )


    # ---- Adjusted ranking ----
    matchup_df["adjusted_rank"] = (
        matchup_df["adjusted_coef"]
        .rank(
            ascending=False,
            method="min",
        )
    )


    # ---- How far did each opponent move? ----
    matchup_df["rank_shift"] = (
        matchup_df["raw_rank"]
        - matchup_df["adjusted_rank"]
    )

    matchup_df["abs_rank_shift"] = (
        matchup_df["rank_shift"].abs()
    )


    # ---- Overall similarity between the two rankings ----
    rank_corr, _ = spearmanr(
        matchup_df["raw_win_rate"],
        matchup_df["adjusted_coef"],
    )


    # Find the single matchup with the biggest disagreement.
    biggest_idx = (
        matchup_df["abs_rank_shift"]
        .idxmax()
    )

    biggest_row = matchup_df.loc[
        biggest_idx
    ]

    biggest_opponent_id = int(
        biggest_row["opponent_id"]
    )


    return {
        "hero_id": hero_id,
        "hero": id_to_name.get(
            hero_id,
            f"Hero {hero_id}",
        ),

        # Number of opponent matchups that survived
        # the MIN_GAMES filter.
        "num_matchups": len(matchup_df),

        # Main measurement:
        # how similar are raw and adjusted rankings?
        "rank_correlation": rank_corr,

        # Typical amount of ranking movement.
        "mean_abs_rank_shift":
            matchup_df["abs_rank_shift"].mean(),

        "median_abs_rank_shift":
            matchup_df["abs_rank_shift"].median(),

        # Largest disagreement for this hero.
        "max_abs_rank_shift":
            matchup_df["abs_rank_shift"].max(),

        "biggest_shift_opponent":
            id_to_name.get(
                biggest_opponent_id,
                f"Hero {biggest_opponent_id}",
            ),

        # Signed value:
        #
        # positive -> opponent moved UP after adjustment
        # negative -> opponent moved DOWN after adjustment
        "biggest_rank_shift":
            biggest_row["rank_shift"],
    }


# ============================================================
# STEP 10: Run the comparison for EVERY hero
# ============================================================

summary_rows = []

for i, hero_id in enumerate(hero_ids, start=1):
    hero_name = id_to_name.get(
        hero_id,
        f"Hero {hero_id}",
    )

    print(
        f"[{i}/{len(hero_ids)}] "
        f"Analyzing {hero_name}"
    )

    result = analyze_hero(hero_id)

    if result is not None:
        summary_rows.append(result)


summary_df = pd.DataFrame(
    summary_rows
)

# ---- Keep heroes with enough qualifying opponents ----
# Correlations based on only a handful of opponents are not very
# comparable to heroes with 100+ matchup observations.
reliable_summary_df = summary_df[
    summary_df["num_matchups"] >= 100
].copy()

# ============================================================
# STEP 11: Overall dataset-level summary
# ============================================================

print("\n===================================")
print("ALL-HERO SUMMARY")
print("===================================")

print(
    "Mean raw-vs-adjusted correlation:",
    f"{reliable_summary_df['rank_correlation'].mean():.3f}",
)

print(
    "Median raw-vs-adjusted correlation:",
    f"{reliable_summary_df['rank_correlation'].median():.3f}",
)

print(
    "Mean absolute rank shift:",
    f"{reliable_summary_df['mean_abs_rank_shift'].mean():.2f}",
)

print(
    "Median absolute rank shift:",
    f"{reliable_summary_df['median_abs_rank_shift'].median():.2f}",
)


# ============================================================
# STEP 12: Heroes whose rankings changed the MOST
# ============================================================
#
# Lower Spearman correlation means:
#
# raw matchup ordering
# and
# adjusted matchup ordering
#
# disagree more strongly.

most_changed = (
    reliable_summary_df
    .sort_values(
        "rank_correlation",
        ascending=True,
    )
    .head(10)
)

print(
    "\nHeroes with LOWEST "
    "raw-vs-adjusted correlation:"
)

print(
    most_changed[
        [
            "hero",
            "num_matchups",
            "rank_correlation",
            "mean_abs_rank_shift",
            "median_abs_rank_shift",
            "biggest_shift_opponent",
            "biggest_rank_shift",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 13: Heroes whose rankings changed the LEAST
# ============================================================

least_changed = (
    reliable_summary_df
    .sort_values(
        "rank_correlation",
        ascending=False,
    )
    .head(10)
)

print(
    "\nHeroes with HIGHEST "
    "raw-vs-adjusted correlation:"
)

print(
    least_changed[
        [
            "hero",
            "num_matchups",
            "rank_correlation",
            "mean_abs_rank_shift",
            "median_abs_rank_shift",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 14: Find Axe as a sanity check
# ============================================================
#
# We already know roughly what Axe should look like.
#
# If this differs wildly from compare_matchups.py,
# something is probably wrong.

axe_row = reliable_summary_df[
    reliable_summary_df["hero_id"] == 2
]

print("\nAxe sanity check:")

print(
    axe_row.to_string(
        index=False
    )
)


# ============================================================
# STEP 15: Save the summary
# ============================================================

output_path = (
    "reports/all_hero_matchup_summary.csv"
)

reliable_summary_df.sort_values(
    "rank_correlation"
).to_csv(
    output_path,
    index=False,
)

print(
    f"\nSaved: {output_path}"
)