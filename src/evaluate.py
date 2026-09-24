"""Final walk-forward evaluation on the held-out test seasons.

This is the only place TEST_SEASONS gets touched. Evaluation is genuine
walk-forward, not a single train/test split: to predict 2023/24 we train on
everything before it; to predict 2024/25 we retrain including 2023/24; to
predict 2025/26 we retrain including 2024/25 too. Each season is always
predicted using only data that would actually have been available before
it kicked off.

Nothing here is tuned or selected on the test seasons: every
hyperparameter, the blend weight, and the choice of "primary model" come
from models/best_params.json, written by train.py from tuning-season CV.
The edge thresholds in the Kelly backtest are fixed in advance (0 and 5%).
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import MODELS_DIR, RAW_DATA_DIR, ROOT_DIR, SEASON_CODES, TEST_SEASONS
from src.features import FEATURE_COLUMNS
from src.train import (
    CLASSES,
    LABEL_TO_IDX,
    TRAIN_SEASONS,
    XGBoostModel,
    ablation_factories,
    load_best_params,
    load_feature_table,
    model_factories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT_DIR / "reports"
PROBA_COLS = [f"proba_{c}" for c in CLASSES]
N_BOOTSTRAP = 5000
# Same probability floor sklearn's log_loss uses, so per-match losses
# average to exactly what log_loss() reports.
_LOG_EPS = np.finfo(float).eps

# The models that get pairwise bootstrap comparisons and appear in the
# README's main table. Calibration variants and the no-decay Dixon-Coles
# are reported in the full table and the report file.
MAIN_MODELS = [
    "bookmaker_closing", "bookmaker_baseline", "blend_lr_dixon_coles", "logistic_regression",
    "xgboost", "dixon_coles",
]


# ---------------------------------------------------------------------------
# Walk-forward test predictions
# ---------------------------------------------------------------------------

def walk_forward_test_predictions(df: pd.DataFrame, factories: dict) -> dict:
    predictions = {name: [] for name in factories}

    for test_season in TEST_SEASONS:
        train_seasons = SEASON_CODES[:SEASON_CODES.index(test_season)]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        logger.info("Test season %s: training on %d matches (%s .. %s), predicting %d matches",
                    test_season, len(train_df), train_seasons[0], train_seasons[-1], len(test_df))

        for name, factory in factories.items():
            model = factory()
            model.fit(train_df)
            proba = model.predict_proba(test_df)
            pred = pd.DataFrame(proba, columns=PROBA_COLS, index=test_df.index)
            for col in ["season", "date", "home_team", "away_team", "result",
                        "odds_home", "odds_draw", "odds_away",
                        "odds_close_home", "odds_close_draw", "odds_close_away",
                        "implied_prob_home", "implied_prob_draw", "implied_prob_away",
                        "implied_close_prob_home", "implied_close_prob_draw", "implied_close_prob_away"]:
                pred[col] = test_df[col].values
            predictions[name].append(pred)

    # Stable sort so every model's rows end up in the identical order,
    # which the paired bootstrap relies on.
    return {name: pd.concat(parts).sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
            for name, parts in predictions.items()}


# ---------------------------------------------------------------------------
# Per-match scores and metrics
# ---------------------------------------------------------------------------

def per_match_scores(pred_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Log loss, Brier and RPS for every match individually, so means can
    be bootstrapped. RPS (ranked probability score) treats H > D > A as
    ordered: predicting a draw when the home team wins is penalised less
    than predicting an away win. Scaled to [0, 1] by 1 / (K - 1)."""
    y_idx = pred_df["result"].map(LABEL_TO_IDX).to_numpy()
    proba = pred_df[PROBA_COLS].to_numpy()
    proba = proba / proba.sum(axis=1, keepdims=True)
    onehot = np.eye(len(CLASSES))[y_idx]

    p_true = np.clip(proba[np.arange(len(y_idx)), y_idx], _LOG_EPS, 1 - _LOG_EPS)
    cum_diff = np.cumsum(proba, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return {
        "log_loss": -np.log(p_true),
        "brier": np.sum((proba - onehot) ** 2, axis=1),
        "rps": np.sum(cum_diff ** 2, axis=1) / (len(CLASSES) - 1),
        "correct": (proba.argmax(axis=1) == y_idx).astype(float),
    }


def multiclass_brier_score(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared distance between predicted probability vectors and the
    one-hot true outcome, summed across classes. 0 = perfect, 2 = maximally
    wrong with full confidence (for 3 classes)."""
    onehot = np.eye(len(CLASSES))[y_true_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ranked_probability_score(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(len(CLASSES))[y_true_idx]
    cum_diff = np.cumsum(proba, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum_diff ** 2, axis=1) / (len(CLASSES) - 1)))


# ---------------------------------------------------------------------------
# Paired bootstrap. Every model is scored on the same 1,140 matches, so a
# difference between two models is bootstrapped by resampling matches
# (the same resampled matches for both), which is much tighter than
# comparing two separate intervals. A second, block version resamples
# whole calendar weeks instead of single matches, as a check that
# same-weekend matches being correlated doesn't change the conclusions.
# (Resampling whole seasons isn't meaningful with only three of them.)
# ---------------------------------------------------------------------------

def bootstrap_indices(n: int, groups: np.ndarray | None = None, n_boot: int = N_BOOTSTRAP,
                      seed: int = 0) -> list[np.ndarray] | np.ndarray:
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.integers(0, n, size=(n_boot, n))
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    picks = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    return [np.concatenate([members[j] for j in row]) for row in picks]


def bootstrap_ci(values: np.ndarray, indices) -> tuple[float, float]:
    if isinstance(indices, np.ndarray):
        means = values[indices].mean(axis=1)
    else:
        means = np.array([values[idx].mean() for idx in indices])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def build_results_table(all_predictions: dict, indices, reference: str = "bookmaker_baseline") -> pd.DataFrame:
    scores = {name: per_match_scores(pred) for name, pred in all_predictions.items()}
    rows = {}
    for name, s in scores.items():
        row = {"accuracy": s["correct"].mean(), "n_matches": len(s["correct"])}
        for metric in ("log_loss", "brier", "rps"):
            row[metric] = s[metric].mean()
            row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_ci(s[metric], indices)
            diff = s[metric] - scores[reference][metric]
            row[f"{metric}_diff_vs_bookmaker"] = diff.mean()
            row[f"{metric}_diff_lo"], row[f"{metric}_diff_hi"] = bootstrap_ci(diff, indices)
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    return table.sort_values("log_loss")


def pairwise_differences(all_predictions: dict, models: list[str], indices, metric: str) -> pd.DataFrame:
    """Row model minus column model; negative = row model is better."""
    scores = {m: per_match_scores(all_predictions[m])[metric] for m in models}
    rows = []
    for a in models:
        for b in models:
            if a == b:
                continue
            diff = scores[a] - scores[b]
            lo, hi = bootstrap_ci(diff, indices)
            rows.append({"model_a": a, "model_b": b, "metric": metric, "mean_diff_a_minus_b": diff.mean(),
                         "ci_lo": lo, "ci_hi": hi, "significant": bool(hi < 0 or lo > 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration(all_predictions: dict, models_to_plot: list[str], n_bins: int = 8):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 5), sharey=True)
    for class_idx, (cls, ax) in enumerate(zip(CLASSES, axes)):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        for model_name in models_to_plot:
            pred_df = all_predictions[model_name]
            y_true = (pred_df["result"] == cls).astype(int)
            frac_pos, mean_pred = calibration_curve(y_true, pred_df[f"proba_{cls}"], n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
        ax.set_title(f"Outcome = {cls}")
        ax.set_xlabel("Mean predicted probability")
        if class_idx == 0:
            ax.set_ylabel("Observed frequency")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration (reliability) curves, held-out test seasons")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_plot.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP feature importance (static snapshot: XGBoost fit on TRAIN_SEASONS
# only, matching the hyperparameter-tuning data, explained on TEST_SEASONS)
# ---------------------------------------------------------------------------

def _normalize_shap_array(raw_shap) -> np.ndarray:
    # shap_values shape varies by version/objective: list-of-arrays (one per
    # class) for older APIs, or a single (n, features, classes) array for
    # newer ones. Normalize to (n, features, classes).
    if isinstance(raw_shap, list):
        return np.stack(raw_shap, axis=-1)
    return raw_shap


def shap_feature_importance(df: pd.DataFrame, best_params: dict, top_n: int = 15) -> pd.DataFrame:
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    model = XGBoostModel(**best_params["xgboost"])
    model.fit(train_df)

    explainer = shap.TreeExplainer(model.model)
    shap_array = _normalize_shap_array(explainer.shap_values(test_df[FEATURE_COLUMNS]))

    mean_abs = np.abs(shap_array).mean(axis=(0, 2))  # mean over samples AND classes
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value| (avg over H/D/A classes)")
    ax.set_title(f"Top {top_n} features: XGBoost, test seasons")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    return importance


# ---------------------------------------------------------------------------
# Where the model fails
# ---------------------------------------------------------------------------

def failure_analysis(pred_df: pd.DataFrame) -> dict:
    y_idx = pred_df["result"].map(LABEL_TO_IDX).to_numpy()
    pred_idx = pred_df[PROBA_COLS].to_numpy().argmax(axis=1)

    cm = confusion_matrix(y_idx, pred_idx, labels=[0, 1, 2])
    report = classification_report(y_idx, pred_idx, labels=[0, 1, 2], target_names=CLASSES,
                                    output_dict=True, zero_division=0)
    return {
        "confusion_matrix": pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASSES],
                                          columns=[f"pred_{c}" for c in CLASSES]),
        "classification_report": pd.DataFrame(report).T,
        "n_actual_draws": int((y_idx == LABEL_TO_IDX["D"]).sum()),
        "n_predicted_draws": int((pred_idx == LABEL_TO_IDX["D"]).sum()),
    }


# ---------------------------------------------------------------------------
# Kelly-criterion backtest, paper money only. A way of asking "are these
# probabilities decision-useful", not betting advice. Stakes are placed at
# Bet365's PRE-CLOSING prices (B365H/D/A: collected Friday afternoon for
# weekend games, Tuesday afternoon for midweek), not closing odds.
# ---------------------------------------------------------------------------

ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]
IMPLIED_COLS = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]
IMPLIED_CLOSE_COLS = ["implied_close_prob_home", "implied_close_prob_draw", "implied_close_prob_away"]
ODDS_BUCKETS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, np.inf]


def select_bets(pred_df: pd.DataFrame, min_edge: float = 0.0) -> pd.DataFrame:
    """For every match, the max-EV outcome under the model's probabilities,
    and whether it clears `min_edge`. One row per match."""
    odds = pred_df[ODDS_COLS].to_numpy()
    proba = pred_df[PROBA_COLS].to_numpy()
    ev = proba * odds - 1
    pick = ev.argmax(axis=1)
    rows = np.arange(len(pred_df))
    out = pred_df[["date", "season", "home_team", "away_team", "result"]].copy()
    out["pick"] = [CLASSES[k] for k in pick]
    out["odds"] = odds[rows, pick]
    out["model_p"] = proba[rows, pick]
    out["market_p"] = pred_df[IMPLIED_COLS].to_numpy()[rows, pick]
    out["close_market_p"] = pred_df[IMPLIED_CLOSE_COLS].to_numpy()[rows, pick]
    out["claimed_ev"] = ev[rows, pick]
    # What the market's own (de-vigged) probabilities say this bet is worth:
    out["market_ev"] = out["market_p"] * out["odds"] - 1
    # Closing-line value: price taken vs. the de-vigged closing fair price.
    # Consistently positive CLV is the standard evidence of a real edge.
    out["clv"] = out["odds"] * out["close_market_p"] - 1
    out["bet"] = out["claimed_ev"] > min_edge
    out["won"] = out["pick"] == out["result"]
    out["flat_return"] = np.where(out["won"], out["odds"] - 1, -1.0)
    return out


def kelly_backtest(pred_df: pd.DataFrame, kelly_fraction: float = 0.5, max_stake: float = 0.2,
                    min_edge: float = 0.0) -> pd.DataFrame:
    """`min_edge` is a materiality threshold: only bet when claimed EV
    exceeds it. Fixed in advance at 0 or 5%, never tuned on test seasons."""
    bets = select_bets(pred_df, min_edge)
    bankroll = 1.0
    history = []
    for row in bets.itertuples(index=False):
        if row.bet:
            b = row.odds - 1
            stake = min(max(row.claimed_ev / b * kelly_fraction, 0), max_stake)
            bankroll *= (1 + stake * b) if row.won else (1 - stake)
        history.append(bankroll)

    result = pred_df[["date", "season"]].copy()
    result["bankroll"] = history
    placed = bets[bets["bet"]]
    result.attrs["n_bets"] = len(placed)
    result.attrs["win_rate"] = placed["won"].mean() if len(placed) else float("nan")
    return result


def summarize_bets(bets: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    placed = bets[bets["bet"]]
    agg = dict(
        n_bets=("won", "size"), win_rate=("won", "mean"), mean_odds=("odds", "mean"),
        mean_model_p=("model_p", "mean"), mean_market_p=("market_p", "mean"),
        mean_claimed_ev=("claimed_ev", "mean"), mean_market_ev=("market_ev", "mean"),
        flat_stake_roi=("flat_return", "mean"), mean_clv=("clv", "mean"),
    )
    if by is None:
        return pd.DataFrame({k: [placed[col].agg(func)] for k, (col, func) in agg.items()}, index=["all"])
    return placed.groupby(by, observed=True).agg(**agg)


def market_longshot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Favourite-longshot bias in the market itself: for every outcome of
    every match, bucket by the offered odds and compare the de-vigged
    implied probability with how often it actually happened, plus the
    flat-stake return from backing every outcome in that bucket. Purely
    descriptive of the bookmaker's prices; nothing is fitted."""
    rows = []
    for k, cls in enumerate(CLASSES):
        rows.append(pd.DataFrame({
            "odds": df[ODDS_COLS[k]].to_numpy(),
            "implied_p": df[IMPLIED_COLS[k]].to_numpy(),
            "won": (df["result"] == cls).to_numpy(),
        }))
    long = pd.concat(rows).dropna()
    long["odds_bucket"] = pd.cut(long["odds"], ODDS_BUCKETS)
    long["flat_return"] = np.where(long["won"], long["odds"] - 1, -1.0)
    return long.groupby("odds_bucket", observed=True).agg(
        n=("won", "size"), mean_implied_p=("implied_p", "mean"), actual_freq=("won", "mean"),
        flat_stake_roi=("flat_return", "mean"))


def winners_curse_simulation(model_pred: pd.DataFrame, n_sims: int = 200, seed: int = 0) -> dict:
    """If the market's de-vigged probabilities were exactly the truth, and
    a 'model' were just those probabilities plus random noise of the same
    size as the real model's disagreement with the market, how often would
    max-EV selection find a 'positive edge', and how big would it claim
    that edge is? If the answer looks like the real model's numbers, noise
    plus picking the maximum explains them, with no real edge needed."""
    rng = np.random.default_rng(seed)
    q = model_pred[IMPLIED_COLS].to_numpy()
    odds = model_pred[ODDS_COLS].to_numpy()
    p = model_pred[PROBA_COLS].to_numpy()
    # Log-probability disagreement, centred per match (log-probs are only
    # defined up to a per-match constant under softmax).
    disagreement = np.log(p) - np.log(q)
    disagreement -= disagreement.mean(axis=1, keepdims=True)
    noise_sd = float(disagreement.std())

    share_positive, mean_claimed, mean_true = [], [], []
    for _ in range(n_sims):
        logits = np.log(q) + rng.normal(0, noise_sd, size=q.shape)
        sim = np.exp(logits - logits.max(axis=1, keepdims=True))
        sim /= sim.sum(axis=1, keepdims=True)
        ev = sim * odds - 1
        pick = ev.argmax(axis=1)
        rows = np.arange(len(q))
        positive = ev[rows, pick] > 0
        share_positive.append(positive.mean())
        mean_claimed.append(ev[rows, pick][positive].mean())
        mean_true.append((q[rows, pick] * odds[rows, pick] - 1)[positive].mean())
    return {
        "noise_sd_log_prob": noise_sd,
        "sim_share_positive_ev": float(np.mean(share_positive)),
        "sim_mean_claimed_ev": float(np.mean(mean_claimed)),
        "sim_mean_true_ev": float(np.mean(mean_true)),
    }


def verify_odds_alignment(pred_df: pd.DataFrame) -> dict:
    """Independent check that each prediction row carries the odds and
    result of the right match: re-read the raw football-data.co.uk CSVs
    with their original column names (bypassing data.py entirely), join
    on (date, home team, away team), and compare."""
    frames = []
    for season in TEST_SEASONS:
        raw = pd.read_csv(RAW_DATA_DIR / f"E0_{season}.csv", encoding="latin-1",
                          usecols=["Date", "HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"])
        raw["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
        frames.append(raw.drop(columns="Date"))
    raw = pd.concat(frames).rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})
    joined = pred_df.merge(raw, on=["date", "home_team", "away_team"], how="left", validate="one_to_one")
    ok_odds = np.isclose(joined[ODDS_COLS].to_numpy(), joined[["B365H", "B365D", "B365A"]].to_numpy()).all(axis=1)
    favourite = joined[ODDS_COLS].to_numpy().argmin(axis=1)
    return {
        "n_rows": len(joined),
        "n_unmatched": int(joined["FTR"].isna().sum()),
        "n_odds_match": int(ok_odds.sum()),
        "n_result_match": int((joined["FTR"] == joined["result"]).sum()),
        # Sanity: the shortest-priced outcome should win far more often than 1/3.
        "favourite_win_rate": float((favourite == joined["result"].map(LABEL_TO_IDX).to_numpy()).mean()),
        "home_fav_share": float((favourite == 0).mean()),
    }


def plot_kelly_backtest(results: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in results.items():
        ax.plot(curve["date"], curve["bankroll"], label=name)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Bankroll (log scale, start = 1.0, half Kelly, paper money)")
    ax.set_xlabel("Date")
    ax.set_title("Kelly-criterion backtest at Bet365 pre-closing odds, test seasons")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "kelly_backtest.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown table (avoids a tabulate dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)
    index_names = [n or "" for n in df.index.names]
    header = index_names + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        lines.append("| " + " | ".join([cell(i) for i in idx] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)


def _fmt_ci(mean, lo, hi, digits=3):
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_diff(mean, lo, hi):
    return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def results_markdown(table: pd.DataFrame, cv: pd.DataFrame | None) -> str:
    cv_map = dict(zip(cv["model"], cv["cv_mean_log_loss"])) if cv is not None else {}
    lines = [
        "| Model | CV log loss (tuning) | Accuracy | Log loss [95% CI] | Brier [95% CI] | RPS [95% CI] | "
        "Δ log loss vs bookmaker [95% CI] | Δ RPS vs bookmaker [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in table.iterrows():
        cv_val = f"{cv_map[name]:.4f}" if name in cv_map else "n/a"
        lines.append(
            f"| {name} | {cv_val} | {r['accuracy']:.3f} | {_fmt_ci(r['log_loss'], r['log_loss_lo'], r['log_loss_hi'])} | "
            f"{_fmt_ci(r['brier'], r['brier_lo'], r['brier_hi'])} | {_fmt_ci(r['rps'], r['rps_lo'], r['rps_hi'], 4)} | "
            f"{_fmt_diff(r['log_loss_diff_vs_bookmaker'], r['log_loss_diff_lo'], r['log_loss_diff_hi'])} | "
            f"{_fmt_diff(r['rps_diff_vs_bookmaker'], r['rps_diff_lo'], r['rps_diff_hi'])} |"
        )
    return "\n".join(lines)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = load_best_params()
    df = load_feature_table()
    cv_path = MODELS_DIR / "cv_summary.csv"
    cv = pd.read_csv(cv_path) if cv_path.exists() else None
    primary = best["primary_model"]

    factories = {**model_factories(best), **ablation_factories(best)}
    logger.info("Running walk-forward evaluation across test seasons: %s", TEST_SEASONS)
    all_predictions = walk_forward_test_predictions(df, factories)
    ablation_names = list(ablation_factories(best))
    main_predictions = {k: v for k, v in all_predictions.items() if k not in ablation_names}

    reference = all_predictions["bookmaker_baseline"]
    for name, pred in all_predictions.items():
        assert (pred[["date", "home_team", "away_team"]].values == reference[["date", "home_team", "away_team"]].values).all(), name

    long = pd.concat([p[["season", "date", "home_team", "away_team", "result", *PROBA_COLS]].assign(model=n)
                      for n, p in all_predictions.items()])
    long.to_csv(REPORTS_DIR / "test_predictions.csv.gz", index=False)

    match_idx = bootstrap_indices(len(reference))
    week_groups = reference["date"].dt.strftime("%G-%V").to_numpy()
    week_idx = bootstrap_indices(len(reference), groups=week_groups)

    results_table = build_results_table(main_predictions, match_idx)
    results_table.to_csv(REPORTS_DIR / "results_table.csv")
    results_table_weekly = build_results_table(main_predictions, week_idx)
    logger.info("\n%s", results_table[["accuracy", "log_loss", "brier", "rps", "log_loss_diff_lo", "log_loss_diff_hi"]].to_string())

    models_for_pairs = [m for m in dict.fromkeys(MAIN_MODELS + [primary])]
    pairwise = pd.concat([pairwise_differences(main_predictions, models_for_pairs, match_idx, m)
                          for m in ("log_loss", "brier", "rps")])
    pairwise.to_csv(REPORTS_DIR / "bootstrap_pairwise.csv", index=False)

    ablation_table = build_results_table({k: all_predictions[k] for k in ablation_names + ["bookmaker_baseline"]}, match_idx)
    ablation_pairs = pd.concat([
        pairwise_differences(all_predictions, [f"{fam}_elo_only", f"{fam}_elo_form", f"{fam}_all"], match_idx, "log_loss")
        for fam in ("logistic", "xgboost")])
    ablation_table.to_csv(REPORTS_DIR / "ablation_table.csv")

    # Each calibration method against its own uncalibrated base model.
    calibration_pairs = pd.concat([
        pairwise_differences(all_predictions, [f"{base}_{method}", base_name], match_idx, metric)
        for base, base_name in (("xgboost", "xgboost"), ("logistic", "logistic_regression"))
        for method in ("temperature", "multinomial", "isotonic")
        for metric in ("log_loss", "rps")
    ])
    calibration_pairs = calibration_pairs[calibration_pairs["model_b"].isin(["xgboost", "logistic_regression"])]
    calibration_pairs.to_csv(REPORTS_DIR / "calibration_comparison.csv", index=False)

    plot_calibration(all_predictions, list(dict.fromkeys(
        [primary, "logistic_regression", "xgboost", "xgboost_temperature", "bookmaker_baseline"])))

    importance = shap_feature_importance(df, best)
    failures = {}
    for name in dict.fromkeys(["xgboost", primary]):
        failures[name] = failure_analysis(all_predictions[name])
        failures[name]["confusion_matrix"].to_csv(REPORTS_DIR / f"confusion_matrix_{name}.csv")
        failures[name]["classification_report"].to_csv(REPORTS_DIR / f"classification_report_{name}.csv")

    # --- Kelly audit ---
    alignment = verify_odds_alignment(reference)
    kelly_models = list(dict.fromkeys(["xgboost", "xgboost_isotonic", primary]))
    kelly_results = {}
    for name in kelly_models:
        for edge in (0.0, 0.05):
            kelly_results[f"{name} (edge>{edge:.0%})"] = kelly_backtest(all_predictions[name], min_edge=edge)
    kelly_results["bookmaker_baseline"] = kelly_backtest(reference, min_edge=0.0)
    plot_kelly_backtest(kelly_results)

    raw_bets = select_bets(all_predictions["xgboost"], min_edge=0.0)
    primary_bets = select_bets(all_predictions[primary], min_edge=0.0)
    for bets in (raw_bets, primary_bets):
        bets["odds_bucket"] = pd.cut(bets["odds"], ODDS_BUCKETS)
    curse = winners_curse_simulation(all_predictions["xgboost"])
    longshot_test = market_longshot_table(reference)
    longshot_all = market_longshot_table(df[df["season"].isin(SEASON_CODES)])

    kelly_summary = pd.DataFrame([
        {"strategy": name, "final_bankroll": c["bankroll"].iloc[-1], "n_bets": c.attrs["n_bets"],
         "win_rate": c.attrs["win_rate"]} for name, c in kelly_results.items()])
    kelly_summary.to_csv(REPORTS_DIR / "kelly_summary.csv", index=False)
    summarize_bets(raw_bets, "pick").to_csv(REPORTS_DIR / "kelly_xgboost_by_outcome.csv")
    summarize_bets(raw_bets, "odds_bucket").to_csv(REPORTS_DIR / "kelly_xgboost_by_odds.csv")

    # --- Markdown report ---
    fmt = _to_markdown
    report = [
        "# Evaluation report (generated by `python -m src.evaluate`)",
        "",
        f"Held-out test seasons: {', '.join(TEST_SEASONS)} ({len(reference)} matches). "
        f"Primary model, chosen by tuning-season CV before scoring the test seasons: **{primary}**.",
        "",
        f"Tuned parameters (walk-forward CV on {TRAIN_SEASONS[0]}..{TRAIN_SEASONS[-1]}): "
        f"Elo {best['elo']}, logistic {best['logistic_regression']}, XGBoost {best['xgboost']}, "
        f"Dixon-Coles {best['dixon_coles']}, blend weight on LR {best['blend']['weight']}.",
        "",
        "## Results (95% CIs: paired bootstrap over matches, 5,000 resamples)",
        "",
        "Δ columns are model minus pre-closing bookmaker; negative means the model scored better. "
        "A CI that contains 0 means no detectable difference.",
        "",
        results_markdown(results_table, cv),
        "",
        "## Same table, block bootstrap over calendar weeks",
        "",
        results_markdown(results_table_weekly, cv),
        "",
        "## Pairwise differences between models (row minus column, paired bootstrap over matches)",
        "",
        fmt(pairwise[pairwise["model_a"] < pairwise["model_b"]].set_index(["metric", "model_a", "model_b"])),
        "",
        "## Feature ablation (each set's hyperparameters tuned separately by tuning-season CV)",
        "",
        fmt(ablation_table[["accuracy", "log_loss", "log_loss_lo", "log_loss_hi", "rps", "log_loss_diff_vs_bookmaker",
                            "log_loss_diff_lo", "log_loss_diff_hi"]]),
        "",
        fmt(ablation_pairs[ablation_pairs["model_a"] > ablation_pairs["model_b"]].set_index(["model_a", "model_b"])),
        "",
        "## Calibration: each method minus its uncalibrated base model (paired bootstrap)",
        "",
        "Calibrators are fit on out-of-fold predictions for the last 3 seasons of each training window; "
        "negative = calibration helped.",
        "",
        fmt(calibration_pairs.set_index(["metric", "model_a", "model_b"])),
        "",
        "## Kelly backtest (Bet365 pre-closing odds, half Kelly, max 20% stake)",
        "",
        "### Odds/outcome alignment check (raw CSVs re-read independently)",
        "",
        fmt(pd.Series(alignment).to_frame("value")),
        "",
        fmt(kelly_summary.set_index("strategy")),
        "",
        "### Raw XGBoost, max-EV bet on every match with claimed EV > 0",
        "",
        f"Share of matches with a claimed positive-EV outcome: {raw_bets['bet'].mean():.1%}.",
        "",
        fmt(summarize_bets(raw_bets)),
        "",
        "By outcome picked:",
        "",
        fmt(summarize_bets(raw_bets, "pick")),
        "",
        "By odds bucket:",
        "",
        fmt(summarize_bets(raw_bets, "odds_bucket")),
        "",
        f"### {primary}, same analysis",
        "",
        f"Share of matches with a claimed positive-EV outcome: {primary_bets['bet'].mean():.1%}.",
        "",
        fmt(summarize_bets(primary_bets)),
        "",
        fmt(summarize_bets(primary_bets, "pick")),
        "",
        "### Favourite-longshot bias in Bet365's own prices (every outcome of every match)",
        "",
        "Test seasons:",
        "",
        fmt(longshot_test),
        "",
        "All 16 seasons (descriptive only, nothing fitted):",
        "",
        fmt(longshot_all),
        "",
        "### Winner's-curse simulation (market probabilities + noise, max-EV selection)",
        "",
        fmt(pd.Series(curse).to_frame("value")),
        "",
        "## Draws",
        "",
        *[f"- {name}: {f['n_predicted_draws']} predicted draws vs {f['n_actual_draws']} actual." for name, f in failures.items()],
        "",
        "## Top SHAP features (XGBoost trained on tuning seasons, explained on test seasons)",
        "",
        fmt(importance.head(10).set_index("feature")),
        "",
    ]
    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Kelly:\n%s", kelly_summary.to_string())
    logger.info("All reports written to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
