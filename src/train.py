"""Model training: baselines, logistic regression, Dixon-Coles, XGBoost.

Every model here shares one interface — `.fit(train_df)` then
`.predict_proba(eval_df) -> (n, 3)` array in CLASSES order (H, D, A) — so
evaluate.py can loop over them identically regardless of what's underneath.

Hyperparameters for XGBoost and logistic regression are chosen by
walk-forward (expanding-window, season-by-season) cross-validation on the
training seasons only — never a random split, and never touching
TEST_SEASONS. See `walk_forward_splits` and `tune_via_walk_forward`.
"""
import itertools
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SEASON_CODES, TEST_SEASONS
from src.features import FEATURE_COLUMNS, build_feature_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["H", "D", "A"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
TRAIN_SEASONS = [s for s in SEASON_CODES if s not in TEST_SEASONS]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_feature_table() -> pd.DataFrame:
    from src.config import PROCESSED_DATA_DIR
    from src.data import load_raw_matches

    cache = PROCESSED_DATA_DIR / "features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_feature_table(load_raw_matches())
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(LABEL_TO_IDX).to_numpy()


# ---------------------------------------------------------------------------
# Walk-forward (expanding window) splitting — the only validation strategy
# used anywhere in this project. Each fold trains on every season strictly
# before the validation season, mirroring how the model would actually be
# deployed (predict the next season using everything known so far).
# ---------------------------------------------------------------------------

def walk_forward_splits(df: pd.DataFrame, seasons: list[str], min_train_seasons: int = 4):
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        train_mask = df["season"].isin(train_seasons)
        val_mask = df["season"] == val_season
        yield df.index[train_mask], df.index[val_mask], val_season


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class HomeWinBaseline:
    """Always predicts a home win. No fitting, no probabilities beyond a
    degenerate [1, 0, 0] — accuracy is the fair metric for this baseline;
    its log loss/Brier score are reported too but are not a meaningful
    comparison against genuinely probabilistic models (see README).
    """

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(eval_df), 3))
        proba[:, LABEL_TO_IDX["H"]] = 1.0
        return proba


class BookmakerBaseline:
    """De-vigged Bet365 implied probabilities, computed in features.py.
    Not fit on training data at all — it's the market's own forecast."""

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return eval_df[["implied_prob_home", "implied_prob_draw", "implied_prob_away"]].to_numpy()


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def make_logistic_model(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


class LogisticRegressionModel:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.pipeline = make_logistic_model(C)

    def fit(self, train_df: pd.DataFrame):
        X = train_df[FEATURE_COLUMNS]
        y = encode_labels(train_df)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        # pipeline.classes_ is sorted [0, 1, 2] == CLASSES order already.
        return self.pipeline.predict_proba(eval_df[FEATURE_COLUMNS])


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
)


class XGBoostModel:
    def __init__(self, **params):
        self.params = {**DEFAULT_XGB_PARAMS, **params}
        self.model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, **self.params,
        )

    def fit(self, train_df: pd.DataFrame):
        # XGBoost handles NaN natively (learns a default split direction for
        # missing values) — no imputation, unlike the logistic pipeline.
        X = train_df[FEATURE_COLUMNS]
        y = encode_labels(train_df)
        self.model.fit(X, y)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(eval_df[FEATURE_COLUMNS])


class CalibratedXGBoostModel:
    """XGBoost with isotonic calibration fit on a held-out chronological
    slice of the training window (the most recent `calib_frac` of matches
    by date — never the eval/test season itself).

    Why this exists: raw softmax probabilities from a gradient-boosted
    tree ensemble are not automatically well-calibrated — a predicted 0.35
    doesn't necessarily mean the outcome happens 35% of the time. Left
    uncorrected, this shows up starkly in evaluate.py's Kelly-criterion
    backtest: the raw model finds a nominal "positive edge" on ~88% of
    matches (mean claimed edge +25%), which is not a real market
    inefficiency, it's calibration noise around odds that are already
    close to fair. Isotonic regression, fit on data the base model never
    trained on, corrects the probability-to-frequency mapping.
    """

    def __init__(self, calib_frac: float = 0.2, **xgb_params):
        self.calib_frac = calib_frac
        self.xgb_params = xgb_params
        self.base_model: XGBoostModel | None = None
        self.calibrated: CalibratedClassifierCV | None = None

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.sort_values("date")
        n_calib = max(int(len(train_df) * self.calib_frac), 1)
        fit_df, calib_df = train_df.iloc[:-n_calib], train_df.iloc[-n_calib:]

        self.base_model = XGBoostModel(**self.xgb_params)
        self.base_model.fit(fit_df)

        X_calib = calib_df[FEATURE_COLUMNS]
        y_calib = encode_labels(calib_df)
        self.calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(self.base_model.model), method="isotonic")
        self.calibrated.fit(X_calib, y_calib)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.calibrated.predict_proba(eval_df[FEATURE_COLUMNS])


# ---------------------------------------------------------------------------
# Dixon-Coles (Dixon & Coles, 1997): a domain-specific alternative to
# treating this as a plain classification problem. Each team gets an attack
# and defense strength; home/away goals are modeled as (near-)independent
# Poisson variables built from those strengths, with a small correction
# (tau/rho) for the historically-observed excess of low-scoring draws.
# Included as a second, non-ML baseline that a football-analytics team
# would actually recognize — not just "yet another classifier."
# ---------------------------------------------------------------------------

def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    MAX_GOALS = 10

    def __init__(self):
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, train_df: pd.DataFrame):
        self.teams = sorted(set(train_df["home_team"]) | set(train_df["away_team"]))
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        if "home_goals" not in train_df.columns:
            raise ValueError("DixonColesModel.fit requires 'home_goals'/'away_goals' columns")

        home_idx = train_df["home_team"].map(idx).to_numpy()
        away_idx = train_df["away_team"].map(idx).to_numpy()
        hg = train_df["home_goals"].to_numpy()
        ag = train_df["away_goals"].to_numpy()

        # Parameters: attack[0..n-1], defense[0..n-1], home_adv, rho.
        # Attack is fixed to sum to zero (via reparametrization at unpack
        # time) so the model is identifiable — otherwise attack and defense
        # could both drift by an arbitrary constant with no effect on fit.
        def unpack(params):
            attack = params[:n - 1]
            attack = np.append(attack, -attack.sum())  # last team's attack fixed by the zero-sum constraint
            defense = params[n - 1:2 * n - 2]
            defense = np.append(defense, -defense.sum())
            home_adv, rho = params[-2], params[-1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])
            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            tau = np.array([_dc_tau(int(x), int(y), l, m, rho) for x, y, l, m in zip(hg, ag, lam, mu)])
            tau = np.clip(tau, 1e-6, None)  # guard against invalid rho pushing tau <= 0 mid-optimization
            ll = ll + np.log(tau)
            return -ll.sum()

        x0 = np.zeros(2 * (n - 1) + 2)
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        attack, defense, home_adv, rho = unpack(result.x)

        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home_adv = float(home_adv)
        self.rho = float(np.clip(rho, -1, 1))
        return self

    def _team_strength(self, team: str) -> tuple[float, float]:
        # Unseen team (e.g. newly promoted, not in the training window):
        # fall back to average (zero) attack/defense rather than erroring.
        return self.attack.get(team, 0.0), self.defense.get(team, 0.0)

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        goals = np.arange(self.MAX_GOALS + 1)
        proba = np.zeros((len(eval_df), 3))

        for row_i, row in enumerate(eval_df.itertuples(index=False)):
            home_attack, home_defense = self._team_strength(row.home_team)
            away_attack, away_defense = self._team_strength(row.away_team)
            lam = np.exp(home_attack + away_defense + self.home_adv)
            mu = np.exp(away_attack + home_defense)

            p_home = poisson.pmf(goals, lam)
            p_away = poisson.pmf(goals, mu)
            grid = np.outer(p_home, p_away)  # grid[x, y] = P(home=x, away=y), independence part

            for x in (0, 1):
                for y in (0, 1):
                    grid[x, y] *= _dc_tau(x, y, lam, mu, self.rho)

            p_h = np.tril(grid, k=-1).sum()   # home_goals > away_goals
            p_d = np.trace(grid)              # home_goals == away_goals
            p_a = np.triu(grid, k=1).sum()    # home_goals < away_goals
            total = p_h + p_d + p_a           # renormalize (grid truncated at MAX_GOALS, tau adjustment)
            proba[row_i] = [p_h / total, p_d / total, p_a / total]

        return proba


# ---------------------------------------------------------------------------
# Hyperparameter tuning via walk-forward CV
# ---------------------------------------------------------------------------

def tune_via_walk_forward(model_factory, param_grid: list[dict], df: pd.DataFrame,
                           seasons: list[str], min_train_seasons: int = 4) -> tuple[dict, pd.DataFrame]:
    """For each candidate param dict, fit on the expanding training window
    of every fold and score log loss on that fold's validation season.
    Returns the best params (lowest mean log loss across folds) and a
    results table for transparency.
    """
    rows = []
    mean_losses = []
    for params in param_grid:
        fold_losses = []
        for train_idx, val_idx, val_season in walk_forward_splits(df, seasons, min_train_seasons):
            model = model_factory(**params)
            model.fit(df.loc[train_idx])
            proba = model.predict_proba(df.loc[val_idx])
            loss = log_loss(encode_labels(df.loc[val_idx]), proba, labels=[0, 1, 2])
            fold_losses.append(loss)
        mean_loss = np.mean(fold_losses)
        mean_losses.append(mean_loss)
        rows.append({**params, "mean_log_loss": mean_loss, "n_folds": len(fold_losses)})
        logger.info("params=%s -> mean_log_loss=%.4f", params, mean_loss)

    # Pull the winning params from the original grid (preserving int/float
    # types as the caller specified them) rather than round-tripping through
    # a DataFrame row, which would silently upcast e.g. max_depth to float
    # (a pandas Series has one dtype, so mixing it with mean_log_loss coerces
    # every value to float64) and break XGBClassifier's param validation.
    best_idx = int(np.argmin(mean_losses))
    best_params = param_grid[best_idx]
    results = pd.DataFrame(rows).sort_values("mean_log_loss").reset_index(drop=True)
    return best_params, results


XGB_PARAM_GRID = [
    {"max_depth": d, "learning_rate": lr, "n_estimators": n}
    for d, lr, n in itertools.product([2, 3, 4], [0.03, 0.1], [100, 300])
]
LOGISTIC_PARAM_GRID = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]


def tune_and_save(df: pd.DataFrame) -> dict:
    train_df = df[df["season"].isin(TRAIN_SEASONS)]

    logger.info("Tuning XGBoost (%d candidates x walk-forward folds)...", len(XGB_PARAM_GRID))
    xgb_best, xgb_results = tune_via_walk_forward(
        lambda **p: XGBoostModel(**p), XGB_PARAM_GRID, train_df, TRAIN_SEASONS)

    logger.info("Tuning logistic regression (%d candidates x walk-forward folds)...", len(LOGISTIC_PARAM_GRID))
    log_best, log_results = tune_via_walk_forward(
        lambda **p: LogisticRegressionModel(**p), LOGISTIC_PARAM_GRID, train_df, TRAIN_SEASONS)

    best = {"xgboost": xgb_best, "logistic_regression": log_best}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    xgb_results.to_csv(MODELS_DIR / "xgb_tuning_results.csv", index=False)
    log_results.to_csv(MODELS_DIR / "logistic_tuning_results.csv", index=False)

    logger.info("Best XGBoost params: %s", xgb_best)
    logger.info("Best logistic regression params: %s", log_best)
    return best


if __name__ == "__main__":
    features = load_feature_table()
    tune_and_save(features)
