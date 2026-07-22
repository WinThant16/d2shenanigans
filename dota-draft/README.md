# Dota 2 Hero Pick Helper

Hello, this is a project I've been working on in my own free time. It is a Dota 2 draft win-probability model, predicting match outcome purely from hero drafts alone.

## Overview

The goal is to predict the outcomes of games based on heroes on either Dire or Radiant. This model was created with the goal to suggest what heroes would be great against certain matchups. It is trained on ~150k Divine-ranked All Pick matches using data from OpenDota API.

## Results
- Test Accuracy: 56.2% on a temporal holdout
- Baseline (always predict Radiant): 53.7%
- Draft contributes ~2.6 points of predictive lift over the side-advantage baseline.
- ![calibration](data/calibration.png)
- Model is well-calibrated in the 0.4-0.7 probability range (where ~ 95% of predictions fall)
- This is a draft-only prediction, and it is inherently capped in the mid-50s because skill and in-game execution are not captured.

## Approach
- Data: OpenDota `/publicMatches`, cursor-based pagination, filtered to ranked all-pick (lobby 7, game mode 22), validated for complete drafts
- Features: each match is encoded as a vector: radiant +1, dire -1, over hero slots
- Split: temporal (train on older match IDs, test on newer) to avoid leakage from meta shifts
- Model: logistic regression baseline

## Status
- [x] Data pipeline (OpenDota fetch + validation)
- [x] Baseline model (logistic regression, 56.2%)
- [ ] Improved model (gradient boosting, synergy features)
- [ ] Draft recommender
- [ ] Interactive demo

## How to run
- Firstly after cloning the repo, make sure you are in the dota-draft directory.
- Create and activate a virtual environment.
- Run `pip install -r requirements.txt`, to install necessary libraries.
- Then run `python src/fetch_matches.py` and `python src/build_features.py`

