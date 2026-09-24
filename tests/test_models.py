"""Checks for the model and metric code that the evaluation relies on."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src.evaluate import bootstrap_ci, bootstrap_indices, per_match_scores, ranked_probability_score, select_bets
from src.train import (
    CalibratedModel,
    DixonColesModel,
    MultinomialScaler,
    TemperatureScaler,
    _dc_tau_and_grads,
)


def _toy_goals(n_teams=6, n_rounds=6, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = np.linspace(-0.4, 0.4, n_teams)
    rows, day = [], pd.Timestamp("2020-08-01")
    for _ in range(n_rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i != j:
                    hg = rng.poisson(np.exp(0.3 + strength[i] - strength[j]))
                    ag = rng.poisson(np.exp(0.0 + strength[j] - strength[i]))
                    res = "H" if hg > ag else ("A" if ag > hg else "D")
                    rows.append(dict(date=day, season="s1", home_team=teams[i], away_team=teams[j],
                                     home_goals=hg, away_goals=ag, result=res))
                    day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_dixon_coles_gradient_matches_finite_differences():
    df = _toy_goals()
    model = DixonColesModel(xi=0.002)
    teams = sorted(set(df["home_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    args = (df["home_team"].map(idx).to_numpy(), df["away_team"].map(idx).to_numpy(),
            df["home_goals"].to_numpy(float), df["away_goals"].to_numpy(float),
            np.exp(-0.002 * np.arange(len(df))[::-1]), len(teams))
    x = np.random.default_rng(1).normal(0, 0.1, 2 * (len(teams) - 1) + 2)
    x[-1] = -0.05
    grad_norm = np.linalg.norm(model.neg_log_likelihood(x, *args)[1])
    err = check_grad(lambda p: model.neg_log_likelihood(p, *args)[0],
                     lambda p: model.neg_log_likelihood(p, *args)[1], x)
    assert err < 1e-4 * grad_norm


def test_dixon_coles_probabilities_are_valid_and_rank_teams():
    model = DixonColesModel(xi=0.0).fit(_toy_goals())
    proba = model.predict_proba(pd.DataFrame({"home_team": ["T5", "T0"], "away_team": ["T0", "T5"]}))
    assert np.allclose(proba.sum(axis=1), 1)
    assert (proba > 0).all()
    # Strongest team at home beats the weakest more often than the reverse.
    assert proba[0, 0] > proba[1, 0]


def test_dixon_coles_tau_matches_paper_definition():
    lam, mu, rho = np.array([1.3]), np.array([0.9]), -0.1
    expected = {(0, 0): 1 - 1.3 * 0.9 * rho, (0, 1): 1 + 1.3 * rho, (1, 0): 1 + 0.9 * rho,
                (1, 1): 1 - rho, (2, 1): 1.0}
    for (x, y), tau in expected.items():
        got, *_ = _dc_tau_and_grads(np.array([x]), np.array([y]), lam, mu, rho)
        assert got[0] == pytest.approx(tau)


def test_rps_known_values():
    perfect = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    assert ranked_probability_score(np.array([0, 2]), perfect) == pytest.approx(0)
    # Home win predicted as a sure draw: cumulative diffs (-1, 0) -> 1/2.
    assert ranked_probability_score(np.array([0]), np.array([[0, 1, 0.]])) == pytest.approx(0.5)
    # Home win predicted as a sure away win: cumulative diffs (-1, -1) -> 2/2.
    assert ranked_probability_score(np.array([0]), np.array([[0, 0, 1.]])) == pytest.approx(1.0)


def test_per_match_scores_average_to_sklearn_log_loss():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(0)
    proba = rng.dirichlet([2, 1, 2], size=200)
    y = rng.integers(0, 3, 200)
    pred = pd.DataFrame(proba, columns=["proba_H", "proba_D", "proba_A"])
    pred["result"] = np.array(["H", "D", "A"])[y]
    scores = per_match_scores(pred)
    assert scores["log_loss"].mean() == pytest.approx(log_loss(y, proba, labels=[0, 1, 2]))
    assert scores["rps"].mean() == pytest.approx(ranked_probability_score(y, proba))


def test_paired_bootstrap_ci_covers_the_mean_and_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    diff = rng.normal(-0.05, 0.1, 1000)
    lo, hi = bootstrap_ci(diff, bootstrap_indices(len(diff), n_boot=2000))
    assert lo < diff.mean() < hi
    assert hi < 0
    groups = np.repeat(np.arange(100), 10)
    lo_b, hi_b = bootstrap_ci(diff, bootstrap_indices(len(diff), groups=groups, n_boot=500))
    assert lo_b < diff.mean() < hi_b


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 0.6, size=(5000, 3))
    true_p = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in true_p])
    overconfident = np.exp(true_logits * 2) / np.exp(true_logits * 2).sum(axis=1, keepdims=True)
    assert TemperatureScaler().fit(overconfident, y).temperature == pytest.approx(2.0, rel=0.15)
    assert np.allclose(MultinomialScaler().fit(overconfident, y).transform(overconfident).sum(axis=1), 1)


def test_calibrated_model_never_sees_rows_outside_its_training_window():
    seen = []

    class Recorder:
        def fit(self, df):
            seen.append(set(df["season"]))
            return self

        def predict_proba(self, df):
            return np.tile([0.5, 0.25, 0.25], (len(df), 1))

    df = pd.DataFrame({
        "season": np.repeat(["a", "b", "c", "d"], 30),
        "date": pd.date_range("2020-01-01", periods=120),
        "result": np.tile(["H", "D", "A"], 40),
    })
    CalibratedModel(lambda: Recorder(), "temperature", n_calib_seasons=3).fit(df)
    # Out-of-fold models for b, c, d train only on seasons strictly before each.
    assert seen[:3] == [{"a"}, {"a", "b"}, {"a", "b", "c"}]
    assert seen[3] == {"a", "b", "c", "d"}


def test_select_bets_picks_max_ev_outcome_with_correct_odds():
    pred = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]), "season": ["2324"], "home_team": ["A"], "away_team": ["B"],
        "result": ["A"], "proba_H": [0.5], "proba_D": [0.2], "proba_A": [0.3],
        "odds_home": [1.8], "odds_draw": [4.0], "odds_away": [4.5],
        "implied_prob_home": [0.52], "implied_prob_draw": [0.23], "implied_prob_away": [0.21],
        "implied_close_prob_home": [0.5], "implied_close_prob_draw": [0.25], "implied_close_prob_away": [0.25],
    })
    bet = select_bets(pred).iloc[0]
    # EVs: H 0.5*1.8-1 = -0.10, D 0.2*4.0-1 = -0.20, A 0.3*4.5-1 = +0.35.
    assert bet["pick"] == "A"
    assert bet["odds"] == 4.5
    assert bet["claimed_ev"] == pytest.approx(0.35)
    assert bool(bet["won"])
    assert bet["flat_return"] == pytest.approx(3.5)
