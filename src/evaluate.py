"""Final walk-forward evaluation on the held-out test seasons.

This is the only place TEST_SEASONS gets touched. Evaluation is genuine
walk-forward, not a single train/test split: to predict 2023/24 we train on
everything before it; to predict 2024/25 we retrain including 2023/24; to
predict 2025/26 we retrain including 2024/25 too. Each season is always
predicted using only data that would actually have been available before
it kicked off — this mirrors how the model would be deployed and retrained
in practice, and is strictly more honest than fitting once and scoring all
three seasons with a single stale model.
"""
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from src.config import MODELS_DIR, ROOT_DIR, SEASON_CODES, TEST_SEASONS
from src.features import FEATURE_COLUMNS
from src.train import (
    BookmakerBaseline,
    CalibratedXGBoostModel,
    CLASSES,
    DixonColesModel,
    HomeWinBaseline,
    LABEL_TO_IDX,
    LogisticRegressionModel,
    TRAIN_SEASONS,
    XGBoostModel,
    encode_labels,
    load_feature_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT_DIR / "reports"
PROBA_COLS = [f"proba_{c}" for c in CLASSES]


def load_best_params() -> dict:
    with open(MODELS_DIR / "best_params.json") as f:
        return json.load(f)


def model_factories(best_params: dict) -> dict:
    return {
        "home_win_baseline": lambda: HomeWinBaseline(),
        "bookmaker_baseline": lambda: BookmakerBaseline(),
        "logistic_regression": lambda: LogisticRegressionModel(**best_params["logistic_regression"]),
        "dixon_coles": lambda: DixonColesModel(),
        "xgboost": lambda: XGBoostModel(**best_params["xgboost"]),
        "xgboost_calibrated": lambda: CalibratedXGBoostModel(**best_params["xgboost"]),
    }


def multiclass_brier_score(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared distance between predicted probability vectors and the
    one-hot true outcome, summed across classes. 0 = perfect, 2 = maximally
    wrong with full confidence (for 3 classes)."""
    onehot = np.eye(len(CLASSES))[y_true_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


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
                        "odds_home", "odds_draw", "odds_away"]:
                pred[col] = test_df[col].values
            predictions[name].append(pred)

    return {name: pd.concat(parts).sort_values("date").reset_index(drop=True)
            for name, parts in predictions.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_df: pd.DataFrame) -> dict:
    y_idx = pred_df["result"].map(LABEL_TO_IDX).to_numpy()
    proba = pred_df[PROBA_COLS].to_numpy()
    pred_idx = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_idx, pred_idx),
        "log_loss": log_loss(y_idx, proba, labels=[0, 1, 2]),
        "brier_score": multiclass_brier_score(y_idx, proba),
        "n_matches": len(pred_df),
    }


def build_results_table(all_predictions: dict) -> pd.DataFrame:
    rows = {name: compute_metrics(pred_df) for name, pred_df in all_predictions.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    return table.sort_values("log_loss")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def plot_calibration(all_predictions: dict,
                      models_to_plot=("xgboost", "xgboost_calibrated", "bookmaker_baseline"),
                      n_bins: int = 8):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 5), sharey=True)
    for class_idx, (cls, ax) in enumerate(zip(CLASSES, axes)):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        for model_name in models_to_plot:
            pred_df = all_predictions[model_name]
            y_true = (pred_df["result"] == cls).astype(int)
            y_prob = pred_df[f"proba_{cls}"]
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
        ax.set_title(f"Outcome = {cls}")
        ax.set_xlabel("Mean predicted probability")
        if class_idx == 0:
            ax.set_ylabel("Observed frequency")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration (reliability) curves, held-out test seasons")
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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


def explain_single_match(base_xgb_model, X_row: pd.DataFrame, class_idx: int, top_n: int = 8) -> pd.DataFrame:
    """Local SHAP explanation for one hypothetical match, used by app.py.
    Explains the underlying (uncalibrated) XGBoost model directly —
    isotonic calibration is a monotonic per-class rescaling and doesn't
    change which features drove the prediction, just how confidently the
    probability is stated."""
    explainer = shap.TreeExplainer(base_xgb_model)
    shap_array = _normalize_shap_array(explainer.shap_values(X_row))
    values = shap_array[0, :, class_idx]
    out = pd.DataFrame({"feature": FEATURE_COLUMNS, "shap_value": values})
    out["abs_shap"] = out["shap_value"].abs()
    return out.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")


def shap_feature_importance(df: pd.DataFrame, best_params: dict, top_n: int = 15) -> pd.DataFrame:
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    model = XGBoostModel(**best_params["xgboost"])
    model.fit(train_df)

    X_test = test_df[FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(model.model)
    shap_array = _normalize_shap_array(explainer.shap_values(X_test))

    mean_abs = np.abs(shap_array).mean(axis=(0, 2))  # mean over samples AND classes
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value| (avg over H/D/A classes)")
    ax.set_title(f"Top {top_n} features — XGBoost, test seasons")
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    return importance


# ---------------------------------------------------------------------------
# Where the model fails
# ---------------------------------------------------------------------------

def failure_analysis(pred_df: pd.DataFrame) -> dict:
    y_idx = pred_df["result"].map(LABEL_TO_IDX).to_numpy()
    proba = pred_df[PROBA_COLS].to_numpy()
    pred_idx = proba.argmax(axis=1)

    cm = confusion_matrix(y_idx, pred_idx, labels=[0, 1, 2])
    report = classification_report(y_idx, pred_idx, labels=[0, 1, 2], target_names=CLASSES,
                                    output_dict=True, zero_division=0)

    n_actual_draws = int((y_idx == LABEL_TO_IDX["D"]).sum())
    n_predicted_draws = int((pred_idx == LABEL_TO_IDX["D"]).sum())

    return {
        "confusion_matrix": pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASSES],
                                          columns=[f"pred_{c}" for c in CLASSES]),
        "classification_report": pd.DataFrame(report).T,
        "n_actual_draws": n_actual_draws,
        "n_predicted_draws": n_predicted_draws,
    }


# ---------------------------------------------------------------------------
# Kelly-criterion calibration backtest — paper money only. This is a way of
# asking "are these probabilities decision-useful", not betting advice: if
# a model is well-calibrated and occasionally spots a mispriced outcome,
# Kelly staking against the closing odds should show positive long-run
# growth; if it's just noise around the market's own numbers, it won't.
# ---------------------------------------------------------------------------

def kelly_backtest(pred_df: pd.DataFrame, kelly_fraction: float = 0.5, max_stake: float = 0.2,
                    min_edge: float = 0.0) -> pd.DataFrame:
    """`min_edge` is a materiality threshold: only bet when claimed EV
    exceeds it, not just whenever it's nominally positive. Without one,
    Kelly staking bets on every scrap of noise around the model's true
    (unknown) probabilities — see the README for what that costs when
    min_edge=0 against a model with no real edge over the market.
    """
    odds = pred_df[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    proba = pred_df[PROBA_COLS].to_numpy()
    y_idx = pred_df["result"].map(LABEL_TO_IDX).to_numpy()

    bankroll = 1.0
    history = []
    n_bets = 0
    n_wins = 0
    for i in range(len(pred_df)):
        p, o = proba[i], odds[i]
        ev = p * o - 1  # expected return per outcome if we bet on it
        best_outcome = int(np.argmax(ev))
        if ev[best_outcome] <= min_edge:
            history.append(bankroll)
            continue

        b = o[best_outcome] - 1  # net decimal odds
        raw_kelly = (p[best_outcome] * o[best_outcome] - 1) / b
        stake = min(max(raw_kelly * kelly_fraction, 0), max_stake)

        won = y_idx[i] == best_outcome
        n_bets += 1
        n_wins += int(won)
        bankroll *= (1 + stake * b) if won else (1 - stake)
        history.append(bankroll)

    result = pred_df[["date", "season"]].copy()
    result["bankroll"] = history
    result.attrs["n_bets"] = n_bets
    result.attrs["win_rate"] = n_wins / n_bets if n_bets else float("nan")
    return result


def plot_kelly_backtest(results: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in results.items():
        ax.plot(curve["date"], curve["bankroll"], label=name)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("Bankroll (starting at 1.0, fractional Kelly, paper money)")
    ax.set_xlabel("Date")
    ax.set_title("Kelly-criterion calibration backtest — test seasons")
    ax.legend()
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "kelly_backtest.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_feature_table()
    best_params = load_best_params()
    factories = model_factories(best_params)

    logger.info("Running walk-forward evaluation across test seasons: %s", TEST_SEASONS)
    all_predictions = walk_forward_test_predictions(df, factories)

    results_table = build_results_table(all_predictions)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table.to_csv(REPORTS_DIR / "results_table.csv")
    logger.info("\n%s", results_table.to_string())

    xgb_log_loss = results_table.loc["xgboost", "log_loss"]
    bookmaker_log_loss = results_table.loc["bookmaker_baseline", "log_loss"]
    if xgb_log_loss < bookmaker_log_loss:
        logger.info("XGBoost BEATS the bookmaker baseline on log loss (%.4f < %.4f).",
                    xgb_log_loss, bookmaker_log_loss)
    else:
        logger.info("XGBoost does NOT beat the bookmaker baseline on log loss (%.4f >= %.4f). "
                     "This is expected and reported honestly in the README, not hidden.",
                     xgb_log_loss, bookmaker_log_loss)

    plot_calibration(all_predictions)

    importance = shap_feature_importance(df, best_params)
    logger.info("Top 10 SHAP features:\n%s", importance.head(10).to_string(index=False))

    failure = failure_analysis(all_predictions["xgboost"])
    logger.info("Confusion matrix (XGBoost):\n%s", failure["confusion_matrix"].to_string())
    logger.info("Per-class report (XGBoost):\n%s", failure["classification_report"].to_string())
    logger.info("Draws: %d actual, %d predicted by XGBoost", failure["n_actual_draws"], failure["n_predicted_draws"])
    failure["confusion_matrix"].to_csv(REPORTS_DIR / "confusion_matrix_xgboost.csv")
    failure["classification_report"].to_csv(REPORTS_DIR / "classification_report_xgboost.csv")

    kelly_results = {
        "xgboost_calibrated_naive": kelly_backtest(all_predictions["xgboost_calibrated"], min_edge=0.0),
        "xgboost_calibrated_5pct_edge": kelly_backtest(all_predictions["xgboost_calibrated"], min_edge=0.05),
        "bookmaker_baseline": kelly_backtest(all_predictions["bookmaker_baseline"], min_edge=0.0),
    }
    plot_kelly_backtest(kelly_results)
    for name, curve in kelly_results.items():
        logger.info("Kelly backtest (%s): final bankroll = %.4fx starting stake, %d bets placed, win rate %.1f%%",
                    name, curve["bankroll"].iloc[-1], curve.attrs["n_bets"], curve.attrs["win_rate"] * 100)

    logger.info("All reports written to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
