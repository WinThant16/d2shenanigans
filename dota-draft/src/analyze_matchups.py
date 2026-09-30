import pandas as pd
import json
from itertools import combinations

df = pd.read_parquet("data/raw/valid-matches.parquet")

with open("constants/heroes.json") as f:
    hero_data = json.load(f)
id_to_name = {int(k): v["localized_name"] for k, v in hero_data.items()}
hero_ids = list(id_to_name.keys())

def build_pair_stats(df, num_slots):
    matchup_games = [[0] * num_slots for _ in range(num_slots)]
    matchup_wins = [[0] * num_slots for _ in range(num_slots)]

    synergy_games = [[0] * num_slots for _ in range(num_slots)]
    synergy_wins = [[0] * num_slots for _ in range(num_slots)]

    for match in df.itertuples(index=False):
        radiant_team = match.radiant_team
        dire_team = match.dire_team
        radiant_win = match.radiant_win

        # Opposing hero matchups
        for radiant_hero in radiant_team:
            for dire_hero in dire_team:

                # Radiant hero's perspective
                matchup_games[radiant_hero][dire_hero] += 1

                if radiant_win:
                    matchup_wins[radiant_hero][dire_hero] += 1

                # Dire hero's perspective
                matchup_games[dire_hero][radiant_hero] += 1

                if not radiant_win:
                    matchup_wins[dire_hero][radiant_hero] += 1

        # Radiant teammate synergies
        for hero_a, hero_b in combinations(radiant_team, 2):
            synergy_games[hero_a][hero_b] += 1
            synergy_games[hero_b][hero_a] += 1

            if radiant_win:
                synergy_wins[hero_a][hero_b] += 1
                synergy_wins[hero_b][hero_a] += 1

        # Dire teammate synergies
        for hero_a, hero_b in combinations(dire_team, 2):
            synergy_games[hero_a][hero_b] += 1
            synergy_games[hero_b][hero_a] += 1

            if not radiant_win:
                synergy_wins[hero_a][hero_b] += 1
                synergy_wins[hero_b][hero_a] += 1

    return (
        matchup_games,
        matchup_wins,
        synergy_games,
        synergy_wins,
    )


def rank_matchups(
    hero_id,
    hero_ids,
    matchup_games,
    matchup_wins,
    min_games=100
):
    results = []

    for opponent_id in hero_ids:
        if opponent_id == hero_id:
            continue

        games = matchup_games[hero_id][opponent_id]

        if games < min_games:
            continue

        wins = matchup_wins[hero_id][opponent_id]
        win_rate = wins / games

        results.append({
            "hero_id": hero_id,
            "opponent_id": opponent_id,
            "games": games,
            "wins": wins,
            "win_rate": win_rate,
        })

    return sorted(
        results,
        key=lambda matchup: matchup["win_rate"],
        reverse=True
    )

num_slots = max(id_to_name.keys()) + 1

(
    matchup_games,
    matchup_wins,
    synergy_games,
    synergy_wins,
) = build_pair_stats(df, num_slots)

axe_matchups = rank_matchups(
    hero_id=2,
    hero_ids=hero_ids,
    matchup_games=matchup_games,
    matchup_wins=matchup_wins,
    min_games=100
)

print("\nBest Axe matchups:")

for matchup in axe_matchups[:5]:
    print(
        id_to_name[matchup["opponent_id"]],
        matchup["games"],
        f"{matchup['win_rate']:.2%}"
    )

print("\nWorst Axe matchups:")

for matchup in reversed(axe_matchups[-5:]):
    print(
        id_to_name[matchup["opponent_id"]],
        matchup["games"],
        f"{matchup['win_rate']:.2%}"
    )