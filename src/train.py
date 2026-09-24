"""Model training: baselines, logistic regression, Dixon-Coles, XGBoost,
an LR + Dixon-Coles blend, and post-hoc calibration wrappers.

Every model here shares one interface: `.fit(train_df)` then
`.predict_proba(eval_df) -> (n, 3)` array in CLASSES order (H, D, A), so
evaluate.py can loop over them identically regardless of what's underneath.

Everything tunable (Elo parameters, XGBoost/logistic hyperparameters, the
Dixon-Coles time-decay rate, the blend weight) is chosen by walk-forward
(expanding-window, season-by-season) cross-validation on TRAIN_SEASONS
only, never a random split, and never touching TEST_SEASONS. The same CV
also scores every model variant, and the lowest-CV-log-loss model is named
the "primary" model *before* anything is scored on the test seasons. See
`tune_and_save`.
"""
import itertools
import json
import logging
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SEASON_CODES, TEST_SEASONS
from src.features import FEATURE_COLUMNS, build_feature_table, parse_feature_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["H", "D", "A"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
TRAIN_SEASONS = [s for s in SEASON_CODES if s not in TEST_SEASONS]

# Feature subsets for the ablation study (Elo only -> Elo + overall rolling
# form -> all 59 features). "Form" here is the overall last-5/last-10
# rolling stats; the home-only/away-only splits and rest days only enter
# with the full set.
ELO_FEATURES = ["elo_home_pre", "elo_away_pre", "elo_diff"]
FORM_FEATURES = [c for c in FEATURE_COLUMNS if (p := parse_feature_name(c)) and p[1] == "form"]
FEATURE_SETS = {
    "elo_only": ELO_FEATURES,
    "elo_form": ELO_FEATURES + FORM_FEATURES,
    "all": FEATURE_COLUMNS,
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_best_params() -> dict:
    path = MODELS_DIR / "best_params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_feature_table(elo_params: dict | None = None) -> pd.DataFrame:
    """Feature table built with the tuned Elo parameters from
    best_params.json (config defaults if tuning hasn't been run). Rebuilt
    from the cached raw CSVs every call (about a second), so it can never
    silently disagree with the tuned Elo settings; the parquet copy is
    written for inspection only."""
    from src.config import PROCESSED_DATA_DIR
    from src.data import load_raw_matches

    if elo_params is None:
        elo_params = load_best_params().get("elo")
    df = build_feature_table(load_raw_matches(), elo_params=elo_params)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(LABEL_TO_IDX).to_numpy()


def _seasons_in_order(df: pd.DataFrame) -> list[str]:
    return list(df.groupby("season")["date"].min().sort_values().index)


# ---------------------------------------------------------------------------
# Walk-forward (expanding window) splitting: the only validation strategy
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


def walk_forward_oof(model_factory, df: pd.DataFrame, seasons: list[str],
                     min_train_seasons: int = 4) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per-fold (val_season, y_true, proba) for one model configuration."""
    folds = []
    for train_idx, val_idx, val_season in walk_forward_splits(df, seasons, min_train_seasons):
        model = model_factory()
        model.fit(df.loc[train_idx])
        folds.append((val_season, encode_labels(df.loc[val_idx]), model.predict_proba(df.loc[val_idx])))
    return folds


def mean_fold_log_loss(folds) -> float:
    return float(np.mean([log_loss(y, p, labels=[0, 1, 2]) for _, y, p in folds]))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class HomeWinBaseline:
    """Always predicts a home win. No fitting, no probabilities beyond a
    degenerate [1, 0, 0]. Accuracy is the fair metric for this baseline;
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
    Not fit on training data at all, it's the market's own forecast.
    `closing=False` uses the pre-closing B365H/D/A prices (every season);
    `closing=True` uses the B365CH/CD/CA closing prices (2019/20 onward)."""

    def __init__(self, closing: bool = False):
        self.prefix = "implied_close_prob" if closing else "implied_prob"

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return eval_df[[f"{self.prefix}_{o}" for o in ("home", "draw", "away")]].to_numpy()


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
    def __init__(self, C: float = 1.0, features: list[str] | None = None):
        self.C = C
        self.features = features or FEATURE_COLUMNS
        self.pipeline = make_logistic_model(C)

    def fit(self, train_df: pd.DataFrame):
        self.pipeline.fit(train_df[self.features], encode_labels(train_df))
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        # pipeline.classes_ is sorted [0, 1, 2] == CLASSES order already.
        return self.pipeline.predict_proba(eval_df[self.features])


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
)


class XGBoostModel:
    def __init__(self, features: list[str] | None = None, **params):
        self.features = features or FEATURE_COLUMNS
        self.params = {**DEFAULT_XGB_PARAMS, **params}
        self.model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, **self.params,
        )

    def fit(self, train_df: pd.DataFrame):
        # XGBoost handles NaN natively (learns a default split direction for
        # missing values), no imputation, unlike the logistic pipeline.
        self.model.fit(train_df[self.features], encode_labels(train_df))
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(eval_df[self.features])


# ---------------------------------------------------------------------------
# Post-hoc calibration, fitted on out-of-fold predictions only.
#
# The earlier version fit isotonic regression on the last 20% of the
# training window, with the base model trained on the first 80%, then
# scored that 80%-data model on the test season. Two problems: the
# served model saw less recent data, and isotonic's step function (with
# ~1000 calibration points spread across 3 classes) is a lot of freedom
# for so little data. Here, the calibrator is fit on walk-forward
# out-of-fold predictions for the last `n_calib_seasons` seasons of the
# training window (each predicted by a model trained only on seasons
# before it), and the served base model is then refit on the whole
# window. Nothing from the evaluation season is ever used.
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """One parameter T: p' = softmax(log(p) / T). T > 1 softens
    over-confident probabilities, T < 1 sharpens under-confident ones.
    Can't reorder outcomes, so it can't overfit much."""

    def fit(self, proba: np.ndarray, y: np.ndarray):
        log_p = np.log(np.clip(proba, _EPS, 1))
        result = minimize_scalar(
            lambda t: log_loss(y, _softmax(log_p / t), labels=[0, 1, 2]),
            bounds=(0.2, 5.0), method="bounded",
        )
        self.temperature = float(result.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return _softmax(np.log(np.clip(proba, _EPS, 1)) / self.temperature)


class MultinomialScaler:
    """Multinomial (matrix) scaling, the 3-class generalisation of Platt
    scaling: a multinomial logistic regression on the base model's log
    probabilities (3x3 weights + 3 biases), lightly L2-regularised."""

    def fit(self, proba: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(C=1.0, max_iter=2000)
        self.model.fit(np.log(np.clip(proba, _EPS, 1)), y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.log(np.clip(proba, _EPS, 1)))


class IsotonicScaler:
    """One-vs-rest isotonic regression per class, renormalised to sum to 1
    (what sklearn's CalibratedClassifierCV(method="isotonic") does)."""

    def fit(self, proba: np.ndarray, y: np.ndarray):
        self.models = [
            IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba[:, k], (y == k).astype(float))
            for k in range(3)
        ]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(self.models)])
        total = out.sum(axis=1, keepdims=True)
        return np.where(total > 0, out / np.where(total > 0, total, 1), 1 / 3)


CALIBRATORS = {"temperature": TemperatureScaler, "multinomial": MultinomialScaler, "isotonic": IsotonicScaler}


class CalibratedModel:
    def __init__(self, base_factory, method: str, n_calib_seasons: int = 3):
        self.base_factory = base_factory
        self.method = method
        self.n_calib_seasons = n_calib_seasons

    def fit(self, train_df: pd.DataFrame):
        seasons = _seasons_in_order(train_df)
        # Always leave at least one season to train the first OOF model on.
        calib_seasons = seasons[max(1, len(seasons) - self.n_calib_seasons):]
        oof_proba, oof_y = [], []
        for season in calib_seasons:
            prior = seasons[:seasons.index(season)]
            model = self.base_factory().fit(train_df[train_df["season"].isin(prior)])
            season_df = train_df[train_df["season"] == season]
            oof_proba.append(model.predict_proba(season_df))
            oof_y.append(encode_labels(season_df))
        self.calibrator = CALIBRATORS[self.method]().fit(np.vstack(oof_proba), np.concatenate(oof_y))
        self.base_model = self.base_factory().fit(train_df)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.base_model.predict_proba(eval_df))


# ---------------------------------------------------------------------------
# Dixon-Coles (Dixon & Coles, 1997): a domain-specific alternative to
# treating this as a plain classification problem. Each team gets an attack
# and defense strength; home/away goals are modeled as (near-)independent
# Poisson variables built from those strengths, with a small correction
# (tau/rho) for the historically-observed excess of low-scoring draws.
#
# Time decay (the paper's own weighting): each training match's
# log-likelihood is weighted by exp(-xi * days_before_last_training_match),
# so recent form counts more than matches from years ago. xi = 0 is the
# unweighted fit. xi is tuned by walk-forward CV like everything else.
# ---------------------------------------------------------------------------

def _dc_tau_and_grads(x, y, lam, mu, rho):
    """Vectorised tau correction, plus its derivatives w.r.t. log(lam),
    log(mu) and rho (for the analytic likelihood gradient)."""
    tau = np.ones_like(lam)
    d_loglam = np.zeros_like(lam)
    d_logmu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m = (x == 0) & (y == 0)
    tau[m] = 1 - lam[m] * mu[m] * rho
    d_loglam[m] = d_logmu[m] = -lam[m] * mu[m] * rho
    d_rho[m] = -lam[m] * mu[m]

    m = (x == 0) & (y == 1)
    tau[m] = 1 + lam[m] * rho
    d_loglam[m] = lam[m] * rho
    d_rho[m] = lam[m]

    m = (x == 1) & (y == 0)
    tau[m] = 1 + mu[m] * rho
    d_logmu[m] = mu[m] * rho
    d_rho[m] = mu[m]

    m = (x == 1) & (y == 1)
    tau[m] = 1 - rho
    d_rho[m] = -1.0
    return tau, d_loglam, d_logmu, d_rho


class DixonColesModel:
    MAX_GOALS = 10

    def __init__(self, xi: float = 0.0):
        self.xi = xi
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def _unpack(self, params: np.ndarray, n: int):
        # Attack and defense are each fixed to sum to zero (the last team's
        # value is minus the sum of the rest) so the model is identifiable,
        # otherwise attack and defense could both drift by an arbitrary
        # constant with no effect on fit.
        attack = np.append(params[:n - 1], -params[:n - 1].sum())
        defense = np.append(params[n - 1:2 * n - 2], -params[n - 1:2 * n - 2].sum())
        return attack, defense, params[-2], params[-1]

    def neg_log_likelihood(self, params, home_idx, away_idx, hg, ag, weights, n):
        """Weighted negative log-likelihood and its analytic gradient.
        (Poisson log-factorial constants are dropped: they don't depend on
        the parameters.)"""
        attack, defense, home_adv, rho = self._unpack(params, n)
        log_lam = attack[home_idx] + defense[away_idx] + home_adv
        log_mu = attack[away_idx] + defense[home_idx]
        lam, mu = np.exp(log_lam), np.exp(log_mu)

        tau, d_loglam, d_logmu, d_rho = _dc_tau_and_grads(hg, ag, lam, mu, rho)
        tau = np.clip(tau, 1e-6, None)
        ll = hg * log_lam - lam + ag * log_mu - mu + np.log(tau)

        g_lam = weights * (hg - lam + d_loglam / tau)
        g_mu = weights * (ag - mu + d_logmu / tau)
        g_attack = np.bincount(home_idx, g_lam, n) + np.bincount(away_idx, g_mu, n)
        g_defense = np.bincount(away_idx, g_lam, n) + np.bincount(home_idx, g_mu, n)
        grad = np.concatenate([
            g_attack[:-1] - g_attack[-1],
            g_defense[:-1] - g_defense[-1],
            [g_lam.sum(), np.sum(weights * d_rho / tau)],
        ])
        return -np.sum(weights * ll), -grad

    def fit(self, train_df: pd.DataFrame):
        if "home_goals" not in train_df.columns:
            raise ValueError("DixonColesModel.fit requires 'home_goals'/'away_goals' columns")

        self.teams = sorted(set(train_df["home_team"]) | set(train_df["away_team"]))
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}
        home_idx = train_df["home_team"].map(idx).to_numpy()
        away_idx = train_df["away_team"].map(idx).to_numpy()
        hg = train_df["home_goals"].to_numpy(dtype=float)
        ag = train_df["away_goals"].to_numpy(dtype=float)
        days_ago = (train_df["date"].max() - train_df["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days_ago)

        x0 = np.zeros(2 * (n - 1) + 2)
        # rho bounded to a range that keeps tau positive for realistic
        # scoring rates; fitted values sit well inside it (around -0.05).
        bounds = [(None, None)] * (2 * (n - 1) + 1) + [(-0.3, 0.3)]
        result = minimize(self.neg_log_likelihood, x0, args=(home_idx, away_idx, hg, ag, weights, n),
                          jac=True, method="L-BFGS-B", bounds=bounds)
        attack, defense, home_adv, rho = self._unpack(result.x, n)

        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        # Unseen team (e.g. newly promoted, not in the training window):
        # fall back to average (zero) attack/defense rather than erroring.
        home_attack = eval_df["home_team"].map(self.attack).fillna(0.0).to_numpy()
        home_defense = eval_df["home_team"].map(self.defense).fillna(0.0).to_numpy()
        away_attack = eval_df["away_team"].map(self.attack).fillna(0.0).to_numpy()
        away_defense = eval_df["away_team"].map(self.defense).fillna(0.0).to_numpy()
        lam = np.exp(home_attack + away_defense + self.home_adv)
        mu = np.exp(away_attack + home_defense)

        goals = np.arange(self.MAX_GOALS + 1)
        # grid[i, x, y] = P(home=x, away=y) for match i, independence part
        grid = poisson.pmf(goals[None, :], lam[:, None])[:, :, None] * poisson.pmf(goals[None, :], mu[:, None])[:, None, :]
        for x in (0, 1):
            for y in (0, 1):
                tau, *_ = _dc_tau_and_grads(np.full_like(lam, x), np.full_like(lam, y), lam, mu, self.rho)
                grid[:, x, y] *= tau

        p_h = np.tril(grid, k=-1).sum(axis=(1, 2))   # home_goals > away_goals
        p_d = np.trace(grid, axis1=1, axis2=2)       # home_goals == away_goals
        p_a = np.triu(grid, k=1).sum(axis=(1, 2))    # home_goals < away_goals
        proba = np.column_stack([p_h, p_d, p_a])
        return proba / proba.sum(axis=1, keepdims=True)  # renormalize (grid truncated at MAX_GOALS, tau)


# ---------------------------------------------------------------------------
# Blend: a linear pool of logistic regression and Dixon-Coles probabilities.
# They get at the problem from different directions (engineered form/Elo
# features vs. a goals model with its own team ratings), so their errors
# are only partly correlated. The weight is tuned by walk-forward CV.
# ---------------------------------------------------------------------------

class BlendModel:
    def __init__(self, weight: float, lr_params: dict, dc_params: dict):
        self.weight = weight
        self.lr = LogisticRegressionModel(**lr_params)
        self.dc = DixonColesModel(**dc_params)

    def fit(self, train_df: pd.DataFrame):
        self.lr.fit(train_df)
        self.dc.fit(train_df)
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.weight * self.lr.predict_proba(eval_df) + (1 - self.weight) * self.dc.predict_proba(eval_df)


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
        folds = walk_forward_oof(lambda: model_factory(**params), df, seasons, min_train_seasons)
        mean_loss = mean_fold_log_loss(folds)
        mean_losses.append(mean_loss)
        rows.append({**params, "mean_log_loss": mean_loss, "n_folds": len(folds)})
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
    for d, lr, n in itertools.product([1, 2, 3], [0.03, 0.1], [50, 100, 200, 300])
]
LOGISTIC_PARAM_GRID = [{"C": c} for c in [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]]
# Per-day decay rates. 0.0019/day is roughly the paper's optimum
# (0.0065 per half-week); a half-life of ln(2)/xi days, i.e. from never
# (xi=0) down to about 6 months (xi=0.004).
DIXON_COLES_PARAM_GRID = [{"xi": x} for x in [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]]
ELO_PARAM_GRID = [
    {"k_factor": k, "home_advantage": h, "carryover": c}
    for k, h, c in itertools.product([10, 15, 20, 25, 30, 40], [30, 60, 90, 120, 150, 180], [0.5, 0.75, 1.0])
]
BLEND_WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]


def tune_elo_params(raw_tuning: pd.DataFrame, seasons: list[str]) -> tuple[dict, pd.DataFrame]:
    """Pick Elo K-factor, home advantage and season carryover by how well
    the Elo features alone predict results: walk-forward CV log loss of a
    logistic regression on just the three Elo features, over the tuning
    seasons. `raw_tuning` must contain tuning-season matches only."""
    rows = []
    for params in ELO_PARAM_GRID:
        feats = build_feature_table(raw_tuning, elo_params=params)
        folds = walk_forward_oof(lambda: LogisticRegressionModel(C=1.0, features=ELO_FEATURES), feats, seasons)
        rows.append({**params, "mean_log_loss": mean_fold_log_loss(folds)})
        logger.info("elo %s -> %.4f", params, rows[-1]["mean_log_loss"])
    results = pd.DataFrame(rows)
    best_idx = int(results["mean_log_loss"].idxmin())
    return ELO_PARAM_GRID[best_idx], results.sort_values("mean_log_loss").reset_index(drop=True)


def tune_blend_weight(lr_folds, dc_folds) -> tuple[float, pd.DataFrame]:
    """Blend weight from the two models' existing walk-forward fold
    predictions (no refitting needed: the weight only mixes them)."""
    rows = []
    for w in BLEND_WEIGHTS:
        blended = [(s, y, w * p_lr + (1 - w) * p_dc) for (s, y, p_lr), (_, _, p_dc) in zip(lr_folds, dc_folds)]
        rows.append({"weight": w, "mean_log_loss": mean_fold_log_loss(blended)})
    results = pd.DataFrame(rows)
    return float(results.loc[results["mean_log_loss"].idxmin(), "weight"]), results


# ---------------------------------------------------------------------------
# The model registry shared by CV model selection, evaluate.py and the app.
# ---------------------------------------------------------------------------

def model_factories(best: dict) -> dict:
    lr, xg, dc = best["logistic_regression"], best["xgboost"], best["dixon_coles"]
    factories = {
        "home_win_baseline": lambda: HomeWinBaseline(),
        "bookmaker_baseline": lambda: BookmakerBaseline(),
        "bookmaker_closing": lambda: BookmakerBaseline(closing=True),
        "logistic_regression": lambda: LogisticRegressionModel(**lr),
        "xgboost": lambda: XGBoostModel(**xg),
        "dixon_coles": lambda: DixonColesModel(**dc),
        "dixon_coles_no_decay": lambda: DixonColesModel(xi=0.0),
        "blend_lr_dixon_coles": lambda: BlendModel(best["blend"]["weight"], lr, dc),
    }
    for method in CALIBRATORS:
        # functools.partial rather than a lambda so a fitted CalibratedModel
        # can be pickled (the app loads a saved model artifact).
        factories[f"xgboost_{method}"] = lambda m=method: CalibratedModel(partial(XGBoostModel, **xg), m)
        factories[f"logistic_{method}"] = lambda m=method: CalibratedModel(partial(LogisticRegressionModel, **lr), m)
    return factories


def ablation_factories(best: dict) -> dict:
    factories = {}
    for set_name, cols in FEATURE_SETS.items():
        params = best["ablation"][set_name]
        factories[f"logistic_{set_name}"] = lambda c=cols, p=params["logistic_regression"]: LogisticRegressionModel(features=c, **p)
        factories[f"xgboost_{set_name}"] = lambda c=cols, p=params["xgboost"]: XGBoostModel(features=c, **p)
    return factories


# Not candidates for "primary model": baselines, the bookmaker itself, and
# the closing-odds benchmark (which only exists from 2019/20).
NON_CANDIDATES = {"home_win_baseline", "bookmaker_baseline", "bookmaker_closing"}


def tune_and_save() -> dict:
    from src.data import load_raw_matches

    raw = load_raw_matches()
    raw_tuning = raw[raw["season"].isin(TRAIN_SEASONS)]

    logger.info("Tuning Elo parameters (%d candidates)...", len(ELO_PARAM_GRID))
    elo_best, elo_results = tune_elo_params(raw_tuning, TRAIN_SEASONS)
    logger.info("Best Elo params: %s", elo_best)

    df = load_feature_table(elo_params=elo_best)
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    tuned = {"elo": elo_best}
    tables = {"elo": elo_results}

    logger.info("Tuning XGBoost...")
    tuned["xgboost"], tables["xgb"] = tune_via_walk_forward(
        lambda **p: XGBoostModel(**p), XGB_PARAM_GRID, train_df, TRAIN_SEASONS)
    logger.info("Tuning logistic regression...")
    tuned["logistic_regression"], tables["logistic"] = tune_via_walk_forward(
        lambda **p: LogisticRegressionModel(**p), LOGISTIC_PARAM_GRID, train_df, TRAIN_SEASONS)
    logger.info("Tuning Dixon-Coles time decay...")
    tuned["dixon_coles"], tables["dixon_coles"] = tune_via_walk_forward(
        lambda **p: DixonColesModel(**p), DIXON_COLES_PARAM_GRID, train_df, TRAIN_SEASONS)

    lr_folds = walk_forward_oof(lambda: LogisticRegressionModel(**tuned["logistic_regression"]), train_df, TRAIN_SEASONS)
    dc_folds = walk_forward_oof(lambda: DixonColesModel(**tuned["dixon_coles"]), train_df, TRAIN_SEASONS)
    weight, tables["blend"] = tune_blend_weight(lr_folds, dc_folds)
    tuned["blend"] = {"weight": weight}

    logger.info("Tuning ablation feature sets...")
    tuned["ablation"] = {}
    for set_name, cols in FEATURE_SETS.items():
        if set_name == "all":
            tuned["ablation"][set_name] = {k: tuned[k] for k in ("logistic_regression", "xgboost")}
            continue
        lr_p, tables[f"ablation_{set_name}_logistic"] = tune_via_walk_forward(
            lambda **p: LogisticRegressionModel(features=cols, **p), LOGISTIC_PARAM_GRID, train_df, TRAIN_SEASONS)
        xgb_p, tables[f"ablation_{set_name}_xgb"] = tune_via_walk_forward(
            lambda **p: XGBoostModel(features=cols, **p), XGB_PARAM_GRID, train_df, TRAIN_SEASONS)
        tuned["ablation"][set_name] = {"logistic_regression": lr_p, "xgboost": xgb_p}

    # Score every model variant with the same walk-forward CV, then name the
    # primary model from these tuning-season scores alone. (Tuned models
    # get a slightly optimistic CV score, since their hyperparameters were
    # picked on these same folds; that applies to all of them alike.)
    logger.info("Cross-validating every model variant for model selection...")
    cv_rows = []
    for name, factory in model_factories(tuned).items():
        if name in ("home_win_baseline", "bookmaker_closing"):
            continue
        folds = walk_forward_oof(factory, train_df, TRAIN_SEASONS)
        cv_rows.append({"model": name, "cv_mean_log_loss": mean_fold_log_loss(folds), "n_folds": len(folds)})
        logger.info("CV %-28s %.4f", name, cv_rows[-1]["cv_mean_log_loss"])
    cv_summary = pd.DataFrame(cv_rows).sort_values("cv_mean_log_loss").reset_index(drop=True)
    candidates = cv_summary[~cv_summary["model"].isin(NON_CANDIDATES)]
    tuned["primary_model"] = candidates.iloc[0]["model"]
    logger.info("Primary model (lowest CV log loss on tuning seasons): %s", tuned["primary_model"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(tuned, f, indent=2)
    tables["cv_summary"] = cv_summary
    for name, table in tables.items():
        table.to_csv(MODELS_DIR / f"{name}_tuning_results.csv" if name != "cv_summary" else MODELS_DIR / "cv_summary.csv",
                     index=False)
    return tuned


if __name__ == "__main__":
    tune_and_save()
