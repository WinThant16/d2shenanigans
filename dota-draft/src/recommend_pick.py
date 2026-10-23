import json

import numpy as np

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

# Best regularization value selected earlier using validation data.
BEST_C = 0.01


# ============================================================
# STEP 1: Load interaction features + labels
# ============================================================
#
# X contains:
#
# individual hero features
# + teammate synergy features
# + opponent matchup features
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
# STEP 1b: Count how many matches support each feature
# ============================================================
#
# X is sparse.
#
# A nonzero entry means that feature was active in that match.
#
# Examples:
#
# individual hero column:
#     number of matches where that hero was picked
#
# synergy pair column:
#     number of matches where those heroes were teammates
#
# matchup pair column:
#     number of matches where those heroes were opponents
#
# This lets us show how much historical support is behind
# each coefficient used by the recommender.

feature_support = np.asarray(
    X.getnnz(axis=0)
).ravel()

assert len(feature_support) == X.shape[1]

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


# Make sure our feature matrix matches the metadata.
assert X.shape[1] == metadata["total_features"]


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
# instead of remembering hero ID 2.

name_to_id = {
    info["localized_name"].lower(): int(hero_id)
    for hero_id, info in hero_data.items()
}


# ============================================================
# ROLE HELPERS
# ============================================================
#
# heroes.json contains broad Valve-style roles such as:
#
# Carry
# Support
# Initiator
# Disabler
# Durable
# Nuker
# Escape
# Pusher
#
# These are NOT exact Dota positions 1-5.

available_roles = sorted({
    role
    for info in hero_data.values()
    for role in info.get("roles", [])
})


def normalize_role(role):
    """
    Convert something like 'carry' or 'CARRY'
    into the canonical role name 'Carry'.

    None means no role filtering.
    """

    if role is None:
        return None

    role_lookup = {
        available_role.lower(): available_role
        for available_role in available_roles
    }

    key = role.strip().lower()

    if key not in role_lookup:
        raise ValueError(
            f"Unknown role: {role}\n"
            f"Available roles: {', '.join(available_roles)}"
        )

    return role_lookup[key]


def hero_has_role(
    hero_id,
    required_role,
):
    """
    Return True if this hero has the requested role.

    If required_role is None, every hero is eligible.
    """

    if required_role is None:
        return True

    roles = hero_data[
        str(hero_id)
    ].get(
        "roles",
        [],
    )

    return required_role in roles


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
# STEP 4: Train the interaction model used for recommendations
# ============================================================
#
# Model selection has already been done.
#
# C = 0.01 was selected using the earlier validation experiment.
#
# For this recommender prototype, fit that model on all
# available historical matches.
#
# We are NOT using this training run to report a new accuracy.
# We are using the learned coefficients to score candidate picks.

print("\nTraining interaction model for recommendations...")

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
# STEP 5: Individual hero coefficient + support
# ============================================================

def get_hero_coef(hero_id):
    return float(
        model.coef_[0][hero_id]
    )


def get_hero_games(hero_id):
    """
    Number of training matches where this hero appeared.
    """

    return int(
        feature_support[hero_id]
    )


# ============================================================
# STEP 6: Teammate synergy coefficient + support
# ============================================================

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


def get_synergy_games(hero_a, hero_b):
    """
    Number of training matches where these two heroes
    appeared on the same team.
    """

    pair = tuple(
        sorted((hero_a, hero_b))
    )

    pair_idx = pair_to_idx[pair]

    synergy_col = (
        synergy_offset
        + pair_idx
    )

    return int(
        feature_support[synergy_col]
    )


# ============================================================
# STEP 7: Opponent matchup coefficient + support
# ============================================================

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

    # Matchup features are stored from the lower-ID hero's
    # direction. Flip the coefficient when hero_id is the
    # higher-ID hero so the result is always from hero_id's
    # perspective.
    if hero_id != pair[0]:
        coef = -coef

    return coef


def get_matchup_games(
    hero_id,
    opponent_id,
):
    """
    Number of training matches where these heroes
    appeared on opposing teams.

    Direction does not matter for the game count:
    PL vs Axe and Axe vs PL have the same support.
    """

    pair = tuple(
        sorted((hero_id, opponent_id))
    )

    pair_idx = pair_to_idx[pair]

    matchup_col = (
        matchup_offset
        + pair_idx
    )

    return int(
        feature_support[matchup_col]
    )


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
# The score itself does NOT depend on sample counts.
#
# Game counts are included only so we can inspect how much
# historical support exists behind each learned coefficient.

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

    individual_games = get_hero_games(
        candidate_id
    )


    # ---- New teammate synergies ----

    synergy_details = []

    for ally_id in ally_ids:
        coef = get_synergy_coef(
            candidate_id,
            ally_id,
        )

        games = get_synergy_games(
            candidate_id,
            ally_id,
        )

        synergy_details.append({
            "hero_id": ally_id,
            "hero": id_to_name[ally_id],
            "coef": coef,
            "games": games,
        })


    # ---- New enemy matchup effects ----

    matchup_details = []

    for enemy_id in enemy_ids:
        coef = get_matchup_coef(
            candidate_id,
            enemy_id,
        )

        games = get_matchup_games(
            candidate_id,
            enemy_id,
        )

        matchup_details.append({
            "hero_id": enemy_id,
            "hero": id_to_name[enemy_id],
            "coef": coef,
            "games": games,
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
        "candidate_id":
            candidate_id,

        "candidate":
            id_to_name[candidate_id],

        "individual_coef":
            individual,

        "individual_games":
            individual_games,

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
# SANITY CHECK: synergy vs matchup direction
# ============================================================
#
# Before trusting recommendations, verify two things:
#
# 1. Synergy is symmetric:
#
#       PL + Axe == Axe + PL
#
# 2. Matchup direction reverses:
#
#       PL vs Axe == -(Axe vs PL)
#
# This checks that our coefficient helpers match the way
# interaction features were encoded during training.

axe_id = get_hero_id("Axe")
pl_id = get_hero_id("Phantom Lancer")


pl_axe_synergy = get_synergy_coef(
    pl_id,
    axe_id,
)

axe_pl_synergy = get_synergy_coef(
    axe_id,
    pl_id,
)


pl_vs_axe = get_matchup_coef(
    pl_id,
    axe_id,
)

axe_vs_pl = get_matchup_coef(
    axe_id,
    pl_id,
)


print("\n===================================")
print("PAIR DIRECTION SANITY CHECK")
print("===================================")

print(
    "PL + Axe synergy:",
    signed(pl_axe_synergy),
)

print(
    "Axe + PL synergy:",
    signed(axe_pl_synergy),
)

print(
    "PL vs Axe:",
    signed(pl_vs_axe),
)

print(
    "Axe vs PL:",
    signed(axe_vs_pl),
)


# Synergy pairs should be identical regardless of argument order.
assert np.isclose(
    pl_axe_synergy,
    axe_pl_synergy,
)


# Matchup direction should be exactly opposite.
assert np.isclose(
    pl_vs_axe,
    -axe_vs_pl,
)


print("Sanity checks passed.")


# ============================================================
# STEP 10: Rank eligible candidate heroes
# ============================================================
#
# Every unpicked hero is normally eligible.
#
# If required_role is supplied, only heroes containing that
# role in heroes.json are considered.
#
# IMPORTANT:
# The scoring formula itself does not change.
#
# We are only filtering candidates BEFORE scoring them.

def recommend_candidates(
    ally_ids,
    enemy_ids,
    required_role=None,
):
    required_role = normalize_role(
        required_role
    )

    picked_ids = set(
        ally_ids + enemy_ids
    )

    results = []

    for candidate_id in hero_ids:

        # Already picked by either team.
        if candidate_id in picked_ids:
            continue


        # Skip heroes that do not satisfy the requested role.
        if not hero_has_role(
            candidate_id,
            required_role,
        ):
            continue


        result = score_candidate(
            candidate_id=candidate_id,
            ally_ids=ally_ids,
            enemy_ids=enemy_ids,
        )

        results.append(result)


    # Highest incremental model contribution first.
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


# Change this to:
#
# "Carry"
# "Support"
# "Initiator"
# etc.
#
# Use None to disable role filtering completely.

REQUIRED_ROLE = "Carry"


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
    required_role=REQUIRED_ROLE,
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

print(
    "Required role:",
    REQUIRED_ROLE
    if REQUIRED_ROLE is not None
    else "Any",
)

print(
    "Eligible candidates:",
    len(recommendations),
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
        f"{signed(result['individual_coef'])} "
        f"({result['individual_games']:,} games)"
    )


    print("  Synergies:")

    for item in result["synergies"]:
        print(
            f"    with {item['hero']:<20} "
            f"{signed(item['coef']):>8} "
            f"({item['games']:,} games)"
        )


    print(
        f"    Synergy total: "
        f"{signed(result['synergy_total'])}"
    )


    print("  Matchups:")

    for item in result["matchups"]:
        print(
            f"    vs {item['hero']:<22} "
            f"{signed(item['coef']):>8} "
            f"({item['games']:,} games)"
        )


    print(
        f"    Matchup total: "
        f"{signed(result['matchup_total'])}"
    )


    print(
        f"  TOTAL: "
        f"{signed(result['total_score'])}"
    )