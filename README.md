# Premier League Match Outcome Predictor

> Status: in progress. This README is filled in incrementally as each module is built and evaluated — see TODOs below.

## Problem Statement

TODO — one paragraph: predict home win / draw / away win for Premier League
matches using pre-match information only, and evaluate rigorously against a
bookmaker-odds baseline.

## Method

TODO — summarize once `src/features.py` and `src/train.py` are done:
- data source and seasons used
- feature engineering (rolling form, Elo, rest days) and leakage safeguards
- validation strategy (walk-forward, time-based split)
- models compared (majority baseline, bookmaker-implied odds, logistic
  regression, Dixon-Coles Poisson, XGBoost)

## Results

TODO — table of accuracy / log loss / Brier score / calibration for each
model on the held-out test seasons, once `src/evaluate.py` is done.

## Limitations

TODO — cold-start for promoted teams, market efficiency ceiling, draw
prediction difficulty, etc.

## What I'd Do Next

TODO

## Repo Structure

```
src/            data.py, features.py, train.py, evaluate.py
tests/          leakage-guard unit tests
notebooks/      exploration only — not imported by src/
app.py          Streamlit demo
```

## Running Locally

```bash
pip install -r requirements.txt
python -m src.data          # download + cache raw CSVs
python -m src.features      # build feature table
python -m src.train         # walk-forward training
python -m src.evaluate      # metrics, calibration, SHAP
streamlit run app.py
```
