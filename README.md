# Dota 2 Hero Pick Helper

A draft-only Dota 2 model for predicting match outcomes and analyzing hero matchup effects from high-rank public matches.

The project started with a simple question:

> How much can we predict about a Dota 2 match from the draft alone?

It has since expanded into a second question:

> How different are raw hero matchup win rates from matchup effects after accounting for the rest of the draft?

## Overview

The dataset contains **300,005 Divine 1+ Ranked All Pick matches** collected from the OpenDota API.

Each match contains only draft information and the final winner. The model does not use player skill, lanes, items, gold, kills, or any in-game information.

The project currently compares two feature representations:

1. **Hero-only features**

   - Radiant hero: `+1`
   - Dire hero: `-1`
   - Hero absent: `0`
2. **Interaction features**

   - Individual hero picks
   - Teammate hero pairs for synergy
   - Opposing hero pairs for matchups

The main goal is not just to improve prediction accuracy, but also to estimate matchup associations after accounting for other modeled draft relationships.

## Prediction Results

Models are evaluated using a temporal split, with older matches used for training and newer matches reserved for validation and testing.

| Model                           |    Test Accuracy |
| ------------------------------- | ---------------: |
| Always predict Radiant          |           52.96% |
| Hero-only Logistic Regression   |           56.36% |
| Interaction Logistic Regression | **57.08%** |

The Logistic Regression regularization parameter `C` was selected using the validation set:

- Hero-only model: `C = 0.003`
- Interaction model: `C = 0.01`

Adding explicit teammate and opponent pair features improved test accuracy by about **0.72 percentage points** over the tuned hero-only model.

This suggests that hero interactions contain predictive information beyond individual hero picks alone.

Because this is a draft-only model, prediction accuracy is naturally limited. Player skill, roles, lanes, itemization, execution, and other in-game factors are not included.

## Interaction Features

There are 127 heroes represented in the dataset.

The interaction feature matrix contains:

- 156 hero-ID slots
- 8,001 possible teammate pairs
- 8,001 possible opponent pairs
- **16,158 total features**

Each match activates:

- 10 individual hero features
- 20 teammate synergy features
- 25 opponent matchup features

for **55 active features per match**.

A sparse matrix is used so that the full dataset can be stored and trained efficiently.

## Raw vs. Adjusted Matchups

A raw matchup win rate answers a question like:

> When Axe played against Shadow Fiend, how often did Axe's team win?

That number also includes everything else happening in those drafts.

The interaction model instead learns a coefficient for the Axe-Shadow Fiend pair while simultaneously modeling:

- individual hero effects
- teammate synergies
- all other opponent pair interactions in the draft

The resulting coefficient is treated as an **adjusted matchup association**.

A positive adjusted coefficient favors the target hero, while a negative coefficient favors the opponent. These coefficients are associations learned from observational match data and should not be interpreted as causal or as "true counters."

### Example: Axe vs. Shadow Fiend

In the analysis subset:

- Axe's raw win rate against Shadow Fiend was about **57.35%**
- Shadow Fiend ranked **#2** among Axe's opponents by raw win rate
- After adjustment, the pair coefficient was slightly negative
- Shadow Fiend fell to approximately **#85** in Axe's adjusted matchup ranking

This suggests that much of Axe's strong historical win rate in those games was associated with other modeled parts of the draft rather than a large Axe-specific advantage over Shadow Fiend.

Other matchups remain similar before and after adjustment, so adjustment does not simply reverse raw matchup statistics.

## All-Hero Matchup Analysis

The same raw-vs-adjusted comparison was repeated across the hero pool.

To make hero-level correlations more comparable, the summary includes only heroes with at least **100 qualifying opponent matchups**, where each individual matchup also appeared in at least 100 games.

This leaves **113 heroes**.

Across those heroes:

| Metric                                  |                    Result |
| --------------------------------------- | ------------------------: |
| Mean raw-vs-adjusted rank correlation   |           **0.697** |
| Median raw-vs-adjusted rank correlation |           **0.708** |
| Mean absolute opponent rank shift       | **20.96 positions** |
| Median absolute opponent rank shift     |    **17 positions** |

The results show that raw matchup statistics contain substantial signal, but adjustment can meaningfully change the ordering of individual matchups.

For example, Axe's raw and adjusted rankings have a Spearman correlation of about **0.692**, close to the overall result.

## Approach

### Data Collection

`src/fetch_matches.py`

- Fetches matches using OpenDota `/publicMatches`
- Uses cursor-based pagination
- Filters for high-rank matches
- Requires Ranked All Pick
- Requires complete 5v5 drafts
- Saves valid matches locally as Parquet
- Supports resuming data collection

### Hero-Only Features

`src/build_features.py`

Encodes each draft using:

```text
Radiant hero = +1
Dire hero    = -1
Not picked   = 0
```

### Interaction Features

`src/build_interaction_features.py`

Extends the representation with:

```text
individual heroes
+
teammate pairs
+
opponent pairs
```

### Model Training

`src/train_hero_logistic.py`

Tunes the hero-only Logistic Regression baseline.

`src/train_interaction_model.py`

Tunes Logistic Regression using the full interaction representation.

### Matchup Analysis

`src/analyze_matchups.py`

Calculates historical raw matchup and synergy statistics.

`src/compare_matchups.py`

Compares raw matchup win rates against model-adjusted matchup coefficients for one hero.

`src/analyze_all_matchups.py`

Repeats the comparison across the hero pool and measures how much raw and adjusted matchup rankings differ.

## Project Structure

```text
dota-draft/
├── constants/
│   └── heroes.json
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── explore.ipynb
├── reports/
│   ├── calibration.png
│   ├── hero_coefficients.png
│   └── all_hero_matchup_summary.csv
└── src/
    ├── fetch_matches.py
    ├── build_features.py
    ├── train_hero_logistic.py
    ├── build_interaction_features.py
    ├── train_interaction_model.py
    ├── analyze_matchups.py
    ├── compare_matchups.py
    └── analyze_all_matchups.py
```

The `data/` directory is ignored by Git because the raw and processed datasets are generated locally.

## How to Run

From the `dota-draft` directory:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Fetch matches:

```bash
python src/fetch_matches.py
```

Build the hero-only features and train the baseline:

```bash
python src/build_features.py
python src/train_hero_logistic.py
```

Build and train the interaction model:

```bash
python src/build_interaction_features.py
python src/train_interaction_model.py
```

Run matchup analysis:

```bash
python src/analyze_matchups.py
python src/compare_matchups.py
python src/analyze_all_matchups.py
```

## Current Status

- [X] OpenDota data collection and validation
- [X] 300k-match dataset
- [X] Hero-only Logistic Regression baseline
- [X] XGBoost baseline comparison
- [X] Teammate synergy features
- [X] Opponent matchup features
- [X] Tuned interaction model
- [X] Raw matchup analysis
- [X] Raw vs. adjusted matchup comparison
- [X] All-hero matchup analysis
- [ ] Draft recommendation system
- [ ] Interactive demo

## Limitations

The adjusted matchup coefficients should not be interpreted as causal hero-counter estimates.

The dataset is observational, and the model does not explicitly account for factors such as:

- player skill
- player role
- lane assignments
- hero position
- patch changes
- item builds
- team strategy
- player familiarity with a hero

The interaction model instead measures conditional associations within the draft features available in the dataset.

## Next Steps

The next stage is to use the learned hero, synergy, and matchup effects to build a draft recommendation system that can evaluate candidate hero picks given an existing partial draft.
