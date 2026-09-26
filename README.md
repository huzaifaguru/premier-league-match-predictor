# Premier League Match Outcome Predictor

[![CI](https://github.com/huzaifaguru/premier-league-match-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/huzaifaguru/premier-league-match-predictor/actions/workflows/ci.yml)

> **TL;DR.** Six models (logistic regression, XGBoost, time-decayed Dixon-Coles, an LR + Dixon-Coles blend, and calibrated variants) were tuned by walk-forward CV and tested on 1,140 held-out matches (2023/24 to 2025/26).
> The benchmark is the bookmakers' own forecast (the probabilities implied by Bet365's prices, with the profit margin removed). The best model reaches log loss 0.981 against the bookmaker's 0.966, a statistically clear gap (95% CI of the difference: +0.005 to +0.024), so **no model is as accurate as the professional forecast**.
> Elo team strength carries almost all of the signal: an Elo-only model is within 0.003 log loss of the full 59-feature model, and the difference is not significant.

**[Live app](https://premier-league-match-predictor-b.streamlit.app/)** · **[Results](#results)** · **[Limitations](#limitations)** · **[What I'd do next](#what-id-do-next)**

## Problem Statement

Given two Premier League teams and a match date, predict the probability
of a home win, draw, or away win using only information available before
kickoff. The interesting part of this problem isn't fitting a classifier.
It's that bookmakers already publish a very strong forecast for every
match, built by professional analysts with access to team news. So the
real question is whether feature engineering + ML can match that forecast
from public match data alone, and if not, why not. This project treats
"no" as a fully acceptable, reportable answer. It is a forecasting and
evaluation study, not a betting tool.

## Method

**Data.** 6,080 Premier League matches, 16 seasons (2010/11 to 2025/26),
downloaded from [football-data.co.uk](https://www.football-data.co.uk/)
and cached locally (`src/data.py`). Includes goals, shots, shots on
target, corners, and Bet365 prices, which are used only to build the
benchmark. A bookmaker's price converts to a probability (1 / price);
dividing by the total across home/draw/away removes the bookmaker's profit
margin and leaves its forecast. Two versions of that forecast are used:

- **Days before kickoff** (`B365H/D/A`), present in every season. Per
  football-data.co.uk's own notes these prices are collected on Friday
  afternoon for weekend matches and Tuesday afternoon for midweek ones.
  This is the main benchmark.
- **At kickoff** (the closing prices, `B365CH/CD/CA`), published only from
  2019/20, so available for all three test seasons. A second, stronger
  benchmark, since it includes late team news.

**Features (`src/features.py`), 59 total, all leakage-safe.** A single
forward pass through matches in date order builds each match's features
only from information strictly before it:
- Rolling form over each team's last 5 and 10 matches (goals for/against,
  shots, shots on target, corners).
- Home-specific and away-specific form (a team's last 5 home matches vs.
  its last 5 away matches).
- Rest days since each team's previous match.
- An Elo rating per team, updated after every match, partly reverted to
  the mean at each season boundary to approximate summer squad turnover.
  K-factor, home advantage and carryover are tuned (see below).
- The bookmaker forecast is computed alongside but **excluded** from the
  model's inputs. If the model could see it, comparing the two would be
  meaningless.

Leakage safety is tested, not just asserted. `tests/test_features.py`
includes a test that changes a future match's result and checks that
features for every earlier match are byte-for-byte unchanged (and that
later matches *do* change, so the test can't pass vacuously), and a test
that appending a later season leaves earlier features unchanged. The
suite has 19 tests in all and runs in CI.

**Validation.** Never a random split. Two layers, both time-based:
1. *Tuning and model selection*: walk-forward (expanding-window,
   season-by-season) cross-validation on the 13 earliest seasons only,
   9 folds. Everything tunable goes through it: Elo parameters, logistic
   regression C, XGBoost depth/learning rate/trees, the Dixon-Coles
   time-decay rate, the blend weight, and the choice between calibration
   methods. The **primary model is the one with the lowest CV log loss,
   named before the test seasons are scored.**
2. *Final evaluation*: genuine walk-forward on the 3 held-out seasons
   (2023/24 to 2025/26). To predict 2023/24, train on everything before
   it; to predict 2024/25, retrain including 2023/24; and so on.

**Models.**
- Majority-class baseline (always predict a home win).
- Bookmaker forecast: Bet365's probabilities with the margin removed, days before kickoff and at kickoff.
- Multinomial logistic regression.
- XGBoost.
- [Dixon-Coles (1997)](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model)
  Poisson goals model, with the paper's exponential time-decay weighting
  (tuned; a half-life of about 15 months won).
- A linear blend of logistic regression and Dixon-Coles (weight tuned: 0.6
  on logistic regression).
- Temperature, multinomial (matrix/Platt-style) and isotonic calibration
  on top of logistic regression and XGBoost, each fitted on out-of-fold
  predictions from the training window only.

**Uncertainty.** Every model is scored on the same 1,140 matches, so
differences are tested with a *paired* bootstrap over matches (5,000
resamples). A block bootstrap over calendar weeks gives the same
conclusions (both are in `reports/evaluation_report.md`). Resampling whole
seasons isn't meaningful with only three test seasons.

## Results

Held out from every stage of tuning and model selection: 2023/24 to
2025/26, 1,140 matches. RPS is the ranked probability score, which treats
home/draw/away as ordered. Lower is better for log loss, Brier and RPS.

| Model | CV log loss | Accuracy | Log loss [95% CI] | Brier | RPS | Δ log loss vs bookmaker [95% CI] |
|---|---|---|---|---|---|---|
| Bookmaker forecast, at kickoff | n/a | 0.550 | 0.960 [0.933, 0.988] | 0.570 | 0.194 | -0.006 [-0.010, -0.002] |
| **Bookmaker forecast, days before** | 0.959 | 0.542 | **0.966** [0.939, 0.994] | 0.574 | 0.196 | (reference) |
| Logistic regression | 0.982 | 0.536 | 0.981 [0.953, 1.007] | 0.584 | 0.200 | +0.015 [+0.005, +0.024] |
| LR + Dixon-Coles blend (primary, chosen by CV) | **0.975** | 0.527 | 0.982 [0.956, 1.008] | 0.585 | 0.201 | +0.016 [+0.008, +0.024] |
| XGBoost | 0.979 | 0.532 | 0.991 [0.963, 1.020] | 0.590 | 0.202 | +0.025 [+0.016, +0.034] |
| Dixon-Coles, time-decayed | 0.986 | 0.505 | 1.002 [0.976, 1.029] | 0.599 | 0.208 | +0.036 [+0.024, +0.049] |
| Dixon-Coles, no decay (old version) | 0.993 | 0.487 | 1.033 [1.007, 1.061] | 0.619 | 0.218 | +0.067 [+0.050, +0.085] |
| Always predict home win | n/a | 0.432 | not meaningful* | 1.137 | 0.446 | n/a |

\* *A degenerate 100/0/0 prediction scores the numerical floor on every
non-home-win match; accuracy is the fair number for this baseline.*

**No model is as accurate as the bookmaker forecast.** Every model's log
loss, Brier and RPS difference from it has a 95% CI entirely above zero,
and the at-kickoff forecast is better still. Among the models:

- **The blend won CV but not the test.** It had the best tuning-season CV
  score, so it was named the primary model in advance (and it is what the
  app serves). On the test seasons it ties plain logistic regression: the
  difference is +0.002, 95% CI [-0.004, +0.007].
- **Logistic regression beats XGBoost**, by 0.011 log loss (CI [0.002,
  0.019]). The best XGBoost found by CV uses depth-1 trees, which can't
  model interactions at all, which is another sign there is little
  non-linear structure for it to exploit.
- **Time decay was the biggest single improvement:** Dixon-Coles went
  from 1.033 to 1.002. It's still the weakest real model on its own, but
  it adds something in the blend during CV.

Full tables (all calibration variants, Brier/RPS CIs, every pairwise
comparison, block bootstrap) are in
[`reports/evaluation_report.md`](reports/evaluation_report.md).

**Why the professionals win: team strength carries almost all the signal.**
A feature ablation, with each feature set's hyperparameters tuned
separately by CV:

| Features | Logistic regression log loss | XGBoost log loss |
|---|---|---|
| Elo only (3 features) | 0.984 | 0.996 |
| Elo + overall rolling form (39) | 0.981 | 0.994 |
| All 59 | 0.981 | 0.991 |

Going from Elo only to all 59 features improves logistic regression by
0.003 (CI [-0.007, +0.012]) and XGBoost by 0.005 (CI [-0.001, +0.011]),
and neither is significant. Everything beyond Elo is worth far less than
the 0.015 gap to the bookmaker. SHAP on XGBoost says the same thing
(`elo_diff` alone has about 4.7x the importance of the next feature), but
the ablation is the stronger evidence, because SHAP splits credit across
correlated Elo features in ways that are hard to interpret.

![SHAP feature importance](reports/shap_importance.png)

The bookmakers' forecast already accounts for team strength, plus
injuries, lineups and news that aren't in this dataset. A model built from
a strict subset of what they know has a low ceiling by construction.

**Draws are never the top pick.** The blend's confusion matrix on the test
seasons:

|  | Predicted H | Predicted D | Predicted A |
|---|---|---|---|
| **Actual H** | 400 | 0 | 92 |
| **Actual D** | 175 | 0 | 104 |
| **Actual A** | 168 | 0 | 201 |

It never predicts a draw as the most likely outcome, out of 279 actual
draws. This is expected: a draw is rarely the single most likely outcome
of a match (it usually sits around 25 to 30%), so an argmax classifier
almost never picks it. The draw probabilities themselves are reasonably
calibrated (middle panel below), which is why the app shows the full
home/draw/away split instead of only the pick.

**Calibration: no method helped.** Reliability curves on the test seasons:

![Calibration plot](reports/calibration_plot.png)

The uncalibrated models already track the diagonal about as well as the
bookmaker. Calibrators were fitted only on out-of-fold predictions from
each training window, and each is compared with its own base model
(log loss difference, paired 95% CI):

| Method | On logistic regression | On XGBoost |
|---|---|---|
| Temperature scaling | +0.001 [-0.000, +0.003] | +0.002 [+0.000, +0.003] |
| Multinomial (matrix) scaling | -0.001 [-0.005, +0.004] | -0.006 [-0.013, +0.000] |
| Isotonic | +0.089 [+0.001, +0.206] | +0.029 [-0.006, +0.093] |

CV on the tuning seasons had already ranked every calibrated variant
below its base model, so none were adopted. Multinomial scaling on
XGBoost looks slightly better on test, but its CI includes zero and CV
disagreed. Isotonic regression is actively harmful, and the mechanism is
visible in the predictions: its step function assigns *exactly zero*
probability to some outcomes (1 test match for XGBoost, 3 for logistic
regression), each of which then costs about 36 nats of log loss. That is
what drove the earlier isotonic XGBoost result from 0.994 to 1.049.

**A note on apparent "edges".** When the model and the bookmaker
disagree, it is tempting to read that as the model spotting value. It
isn't. The model's probabilities differ from the bookmaker's by random
noise, and whenever you pick the largest of three noisy differences, one
almost always looks favourable (the "winner's curse"). A simulation with
pure noise of the same size reproduces the effect, and a paper-money
stress test of acting on those differences lost steadily. The details,
kept as a technical diagnostic, are in
[`reports/evaluation_report.md`](reports/evaluation_report.md). This
project is not a betting tool, and nothing in it is betting advice.

## Limitations

- **No lineup, injury, or news information.** This is the single largest
  gap vs. the bookmaker, whose at-kickoff forecast includes exactly this.
- **Only three test seasons.** Confidence intervals are about ±0.03 on log
  loss for a single model. Paired differences are much tighter, but
  season-level effects (one unusual season) can't be separated from
  model quality.
- **Cold start for promoted teams.** Elo starts at a league-average 1500,
  and Dixon-Coles uses average attack/defense, for teams with no Premier
  League history in the training window.
- **Elo tuning barely matters.** The original defaults (K=20, home
  advantage 60, carryover 0.75) score only 0.0003 worse in CV than the
  tuned best, and for any K from 15 to 30 and any home advantage, the
  best carryover gets within 0.0005. The chosen home advantage (180) sits
  on the grid edge, which is noise on a flat surface, not a meaningful
  optimum. Rolling-window lengths (5/10 matches) are still untuned.
- **The blend weight and every hyperparameter were picked on the same CV
  folds that score them**, so CV numbers are slightly optimistic. This
  applies to all models alike, and the test seasons were never used.

## What I'd Do Next

Done since the first version: RPS, bootstrap confidence intervals, a
time-decayed Dixon-Coles tuned in the CV loop, a blend, tuned Elo, a
feature ablation, and alternative calibration methods.

1. **Add news/lineup features** (confirmed starting XI, key injuries). The
   most likely way to close some of the gap to the bookmaker.
2. **A proper stacking model**: learn the blend on out-of-fold
   predictions with a meta-model, rather than a single linear weight.
3. **A second bookmaker's forecast** (Pinnacle, available from
   2012/13), widely regarded as the most accurate bookmaker forecast.
4. **Bayesian team ratings** (e.g. a hierarchical Dixon-Coles), giving
   each prediction an honest uncertainty band.
5. **Tune the rolling-window lengths**, the last untuned feature choice.

## Repo Structure

```
.github/workflows/  CI: installs requirements and runs pytest on every push/PR to main
.streamlit/         Streamlit theme config
assets/crests/      team crest images used by the app (see its README)
src/
  config.py         paths, seasons, default Elo settings
  data.py           download + cache raw CSVs from football-data.co.uk
  features.py       leakage-safe rolling form, Elo, rest days (LeagueState)
  train.py          all models, calibration, walk-forward tuning and CV model selection
  evaluate.py       test-season evaluation: bootstrap CIs, ablation, calibration, SHAP, "edge" diagnostic
  app_model.py      builds/loads the small model artifact the app serves
  app_helpers.py    app data helpers (recent form, standings, head-to-head); not model features
  ui.py             app presentation: colour tokens, the one CSS function, HTML components
tests/              leakage guards and model/metric tests (pytest, run in CI)
app.py              Streamlit app
models/             best_params.json, tuning/CV tables, app_model.joblib (6 KB)
reports/            generated plots and tables, including evaluation_report.md
data/               raw/ and processed/ caches (gitignored, re-downloadable)
```

## Running Locally

Requires Python 3.12 (the pinned numpy/scipy/xgboost versions have no
3.11 wheels).

```bash
pip install -r requirements.txt
python -m src.data          # download + cache raw CSVs
python -m src.train         # walk-forward tuning + CV model selection (~40 min)
python -m src.evaluate      # held-out evaluation, writes reports/ (~3 min)
python -m src.app_model     # refit and save the app's model artifact
pytest tests/               # leakage guards + model tests
streamlit run app.py        # the app
```
