"""Regression test for a dtype bug in tune_via_walk_forward: pulling a
winning hyperparameter set out of a DataFrame row silently upcast ints
(e.g. XGBoost's max_depth) to float, since a pandas Series has one dtype
for the whole row. XGBClassifier then failed on max_depth=2.0.
"""
import pandas as pd

from src.train import tune_via_walk_forward


class _DummyModel:
    """Ignores the data entirely; score is just a function of `depth` so
    the walk-forward machinery has something deterministic to select."""

    def __init__(self, depth: int, rate: float):
        self.depth = depth
        self.rate = rate

    def fit(self, train_df):
        return self

    def predict_proba(self, eval_df):
        import numpy as np
        # Lower `depth` -> better (lower) log loss, so depth=2 always wins.
        base = 1 / 3 + 0.05 * self.depth
        p_home = min(max(base, 0.01), 0.98)
        rest = (1 - p_home) / 2
        return np.tile([p_home, rest, rest], (len(eval_df), 1))


def _toy_df():
    seasons = ["s1", "s2", "s3", "s4", "s5"]
    rows = []
    for s in seasons:
        for i in range(6):
            rows.append({
                "season": s,
                "result": ["H", "D", "A"][i % 3],
            })
    return pd.DataFrame(rows)


def test_best_params_preserve_original_types_not_upcast_to_float():
    df = _toy_df()
    grid = [{"depth": 2, "rate": 0.1}, {"depth": 4, "rate": 0.1}]

    best_params, results = tune_via_walk_forward(
        lambda **p: _DummyModel(**p), grid, df, seasons=["s1", "s2", "s3", "s4", "s5"], min_train_seasons=2,
    )

    assert best_params["depth"] == 2
    assert isinstance(best_params["depth"], int), (
        f"depth should stay int, got {type(best_params['depth'])} -- "
        "this is the DataFrame-row upcast bug regressing"
    )
