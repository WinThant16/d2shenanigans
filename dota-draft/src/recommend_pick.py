import json

import numpy as np

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

# We already selected this using the validation set.
BEST_C = 0.01


# ============================================================
# STEP 1: Load interaction features + labels
# ============================================================
#
# For this recommender prototype, we'll fit the deployment model
# using all available matches.
#
# We already evaluated C=0.01 using the earlier temporal
# train / validation / test experiment.
#
# This model is NOT being used to report another test accuracy.
# It is the model we want to use for recommendations.

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
# The metadata tells us:
#
# - which heroes exist
# - which hero pairs exist
# - where synergy features begin
# - where matchup features begin

with open(
    "data/processed/interaction_metadata.json"
) as f:
    metadata = json.load(f)


hero_ids = [
    int(hero_id)
    for hero_id in metadata["hero_ids"]
]


# JSON stores tuples as lists, so convert them back.
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


assert X.shape[1] == metadata["total_features"]


# ============================================================
# STEP 3: Load hero names
# ============================================================

with open("constants/heroes.json") as f:
    hero_data = json.load(f)


id_to_name = {
    int(hero_id): info["localized_name"]
    for hero_id, info in hero_data.items()
}


# Also build the opposite mapping so we can type:
#
# "Axe"
#
# instead of remembering:
#
# 2

name_to_id = {
    info["localized_name"].lower(): int(hero_id)
    for hero_id, info in hero_data.items()
}


def get_hero_id(hero_name):
    """
    Convert a hero name such as 'Axe' into its numeric hero ID.
    Matching is case-insensitive.
    """

    key = hero_name.strip().lower()

    if key not in name_to_id:
        raise ValueError(
            f"Unknown hero name: {hero_name}"
        )

    hero_id = name_to_id[key]

    if hero_id not in hero_ids:
        raise ValueError(
            f"{hero_name} exists in heroes.json "
            f"but not in this dataset."
        )

    return hero_id


# ============================================================
# STEP 4: Fit the deployment interaction model
# ============================================================
#
# Earlier we used:
#
# 70% train
# 15% validation
# 15% test
#
# to select C and evaluate the model.
#
# That experiment is finished.
#
# Now that C=0.01 is fixed, we can train the version used by
# the recommender on all available historical matches.

print("\nTraining deployment interaction model...")

model = LogisticRegression(
    C=BEST_C,
    max_iter=1000,
)

model.fit(
    X,
    y,
)

print("Finished training model.")


# ============================================================
# STEP 5: Get one individual hero coefficient
# ============================================================
#
# Individual hero features use the hero ID directly as the
# feature column:
#
# hero ID 2  -> column 2
# hero ID 11 -> column 11
#
# Because the original feature representation was:
#
# Radiant hero = +1
# Dire hero    = -1
#
# a positive coefficient represents a favorable association
# for whichever team the hero is playing on.

def get_hero_coef(hero_id):
    return float(
        model.coef_[0][hero_id]
    )


# ============================================================
# STEP 6: Get a teammate synergy coefficient
# ============================================================
#
# Example:
#
# Axe + Dazzle
#
# Synergy features were encoded:
#
# +1 if the pair is together on Radiant
# -1 if the pair is together on Dire
#
# Therefore a positive coefficient means the pair is favorable
# for the team they're on.
#
# Unlike matchup features, we do NOT need an ID-based sign flip.

def get_synergy_coef(hero_a, hero_b):
    pair = tuple(
        sorted((hero_a, hero_b))
    )

    pair_idx = pair_to_idx[pair]

    synergy_col = (
        synergy_offset
        + pair_idx
    )

    return float(
        model.coef_[0][synergy_col]
    )


# ============================================================
# STEP 7: Get matchup coefficient from ONE hero's perspective
# ============================================================
#
# Matchup features use the lower-ID hero as the stored
# reference direction.
#
# After this function:
#
# positive -> favorable for hero_id
# negative -> unfavorable for hero_id

def get_matchup_coef(
    hero_id,
    opponent_id,
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

    # Stored coefficient favors/disfavors the lower-ID hero.
    #
    # If our candidate is the higher-ID hero, reverse the
    # direction so the number is always from the candidate's
    # perspective.
    if hero_id != pair[0]:
        coef = -coef

    return coef


# ============================================================
# STEP 8: Score ONE candidate
# ============================================================
#
# Candidate score =
#
# individual hero effect
# + synergy with each known ally
# + matchup with each known enemy
#
# Existing ally-ally, ally-enemy, and enemy-enemy effects
# don't need to be included here.
#
# Why?
#
# They are already part of the draft and do not change when
# comparing one candidate against another candidate.

def score_candidate(
    candidate_id,
    ally_ids,
    enemy_ids,
):
    picked_ids = set(
        ally_ids + enemy_ids
    )

    if candidate_id in picked_ids:
        raise ValueError(
            "Candidate hero has already been picked."
        )


    # ---- Individual contribution ----

    individual = get_hero_coef(
        candidate_id
    )


    # ---- New teammate synergies ----

    synergy_details = []

    for ally_id in ally_ids:
        coef = get_synergy_coef(
            candidate_id,
            ally_id,
        )

        synergy_details.append({
            "hero_id": ally_id,
            "hero": id_to_name[ally_id],
            "coef": coef,
        })


    # ---- New enemy matchup effects ----

    matchup_details = []

    for enemy_id in enemy_ids:
        coef = get_matchup_coef(
            candidate_id,
            enemy_id,
        )

        matchup_details.append({
            "hero_id": enemy_id,
            "hero": id_to_name[enemy_id],
            "coef": coef,
        })


    synergy_total = sum(
        item["coef"]
        for item in synergy_details
    )

    matchup_total = sum(
        item["coef"]
        for item in matchup_details
    )


    total_score = (
        individual
        + synergy_total
        + matchup_total
    )


    return {
        "candidate_id": candidate_id,
        "candidate":
            id_to_name[candidate_id],

        "individual_coef":
            individual,

        "synergy_total":
            synergy_total,

        "matchup_total":
            matchup_total,

        "total_score":
            total_score,

        "synergies":
            synergy_details,

        "matchups":
            matchup_details,
    }


# ============================================================
# STEP 9: Helper for printing signed numbers
# ============================================================
#
# This makes:
#
# 0.123 -> +0.1230
#
# easier to read in the output.

def signed(value):
    return f"{value:+.4f}"


# ============================================================
# STEP 10: Rank all available candidate heroes
# ============================================================
#
# Run score_candidate() for every hero that has not already
# been picked.
#
# Since every candidate is evaluated against the SAME partial
# draft, the scores are directly comparable.
#
# Higher total_score = more favorable according to the model.

def recommend_candidates(
    ally_ids,
    enemy_ids,
):
    picked_ids = set(
        ally_ids + enemy_ids
    )

    results = []

    for candidate_id in hero_ids:

        # A hero cannot be recommended if it has already
        # been picked by either team.
        if candidate_id in picked_ids:
            continue

        result = score_candidate(
            candidate_id=candidate_id,
            ally_ids=ally_ids,
            enemy_ids=enemy_ids,
        )

        results.append(result)


    # Highest candidate contribution first.
    results.sort(
        key=lambda result:
            result["total_score"],
        reverse=True,
    )

    return results


# ============================================================
# STEP 11: Example partial draft
# ============================================================

ALLY_NAMES = [
    "Axe",
    "Dazzle",
]

ENEMY_NAMES = [
    "Shadow Fiend",
    "Juggernaut",
]


ally_ids = [
    get_hero_id(name)
    for name in ALLY_NAMES
]

enemy_ids = [
    get_hero_id(name)
    for name in ENEMY_NAMES
]


recommendations = recommend_candidates(
    ally_ids=ally_ids,
    enemy_ids=enemy_ids,
)


# ============================================================
# STEP 12: Print top recommendations
# ============================================================

print("\n===================================")
print("PARTIAL DRAFT")
print("===================================")

print(
    "Allies:",
    ", ".join(ALLY_NAMES),
)

print(
    "Enemies:",
    ", ".join(ENEMY_NAMES),
)


print("\n===================================")
print("TOP 10 RECOMMENDATIONS")
print("===================================")

print(
    f"{'Rank':<6}"
    f"{'Hero':<22}"
    f"{'Individual':>12}"
    f"{'Synergy':>12}"
    f"{'Matchup':>12}"
    f"{'Total':>12}"
)

print("-" * 76)


for rank, result in enumerate(
    recommendations[:10],
    start=1,
):
    print(
        f"{rank:<6}"
        f"{result['candidate']:<22}"
        f"{signed(result['individual_coef']):>12}"
        f"{signed(result['synergy_total']):>12}"
        f"{signed(result['matchup_total']):>12}"
        f"{signed(result['total_score']):>12}"
    )


# ============================================================
# STEP 13: Detailed breakdown for top 3
# ============================================================
#
# A ranked list is useful, but we also want the recommender
# to explain WHY each hero scored highly.

print("\n===================================")
print("TOP 3 BREAKDOWNS")
print("===================================")


for rank, result in enumerate(
    recommendations[:3],
    start=1,
):
    print(
        f"\n#{rank} {result['candidate']}"
    )

    print(
        f"  Individual hero effect: "
        f"{signed(result['individual_coef'])}"
    )


    print("  Synergies:")

    for item in result["synergies"]:
        print(
            f"    with {item['hero']:<20} "
            f"{signed(item['coef'])}"
        )


    print(
        f"    Synergy total: "
        f"{signed(result['synergy_total'])}"
    )


    print("  Matchups:")

    for item in result["matchups"]:
        print(
            f"    vs {item['hero']:<22} "
            f"{signed(item['coef'])}"
        )


    print(
        f"    Matchup total: "
        f"{signed(result['matchup_total'])}"
    )


    print(
        f"  TOTAL: "
        f"{signed(result['total_score'])}"
    )