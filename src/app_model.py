"""The model the Streamlit app serves, saved as a small committed artifact.

`python -m src.app_model` fits the primary model (named in
models/best_params.json by tuning-season CV) on every complete season and
saves it to models/app_model.joblib together with the Elo parameters its
features were built with and the library versions it was pickled under.
The app loads that file instead of retraining on every cold start. If the
file is missing, or was saved under different sklearn/xgboost/numpy
versions (pickles aren't portable across versions), the app retrains once
and caches the result for the life of the process.
"""
import logging

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

from src.config import MODELS_DIR, SEASON_CODES
from src.features import FEATURE_COLUMNS
from src.train import (
    BlendModel,
    CalibratedModel,
    LogisticRegressionModel,
    XGBoostModel,
    load_best_params,
    load_feature_table,
    model_factories,
)

logger = logging.getLogger(__name__)

ARTIFACT_PATH = MODELS_DIR / "app_model.joblib"


def _library_versions() -> dict:
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__}


def train_app_bundle(features: pd.DataFrame | None = None) -> dict:
    best = load_best_params()
    name = best["primary_model"]
    if features is None:
        features = load_feature_table(elo_params=best["elo"])
    train_df = features[features["season"].isin(SEASON_CODES)]
    model = model_factories(best)[name]().fit(train_df)
    return {
        "model": model,
        "model_name": name,
        "elo_params": best["elo"],
        "trained_on_seasons": list(SEASON_CODES),
        "versions": _library_versions(),
    }


def save_app_bundle(bundle: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)


def load_app_bundle() -> dict | None:
    """The saved bundle, or None if it's missing or unusable here."""
    if not ARTIFACT_PATH.exists():
        return None
    try:
        bundle = joblib.load(ARTIFACT_PATH)
    except Exception:
        logger.warning("Could not load %s", ARTIFACT_PATH, exc_info=True)
        return None
    if bundle.get("versions") != _library_versions():
        logger.warning("Model artifact saved under %s, running %s; retraining instead",
                       bundle.get("versions"), _library_versions())
        return None
    return bundle


# ---------------------------------------------------------------------------
# Per-prediction explanations, whatever the primary model is.
# ---------------------------------------------------------------------------

def _explained_component(model):
    """The part of the model an explanation can be computed for. Calibration
    is a smooth rescaling on top of a base model, so explaining the base
    model shows what drove the prediction; for the blend, only the logistic
    half has per-feature attributions (Dixon-Coles uses team ratings, not
    these features)."""
    if isinstance(model, CalibratedModel):
        return _explained_component(model.base_model)
    if isinstance(model, BlendModel):
        return model.lr
    return model


def explanation_note(model) -> str:
    if isinstance(model, BlendModel):
        return (f"Factors shown are for the logistic-regression part of the blend, which carries "
                f"{model.weight:.0%} of the weight; the Dixon-Coles part rates teams from goals scored "
                "and conceded rather than from these features.")
    if isinstance(model, CalibratedModel):
        return "Factors are computed for the model before its calibration step, which only rescales probabilities."
    return ""


def explain_prediction(model, row: pd.DataFrame, class_idx: int, top_n: int = 8) -> pd.DataFrame:
    """Top features pushing toward (positive) or away from (negative) the
    given outcome for one match, in log-odds units.

    Logistic regression: each feature's exact contribution to that
    outcome's logit, coefficient x standardised value, relative to the
    average across the three outcomes (for a linear model this is its
    SHAP value). XGBoost: TreeExplainer SHAP values.
    """
    component = _explained_component(model)
    if isinstance(component, LogisticRegressionModel):
        steps = component.pipeline.named_steps
        z = steps["scale"].transform(steps["impute"].transform(row[component.features]))[0]
        contrib = steps["clf"].coef_ * z  # (3 classes, n features)
        values = contrib[class_idx] - contrib.mean(axis=0)
        features = component.features
    elif isinstance(component, XGBoostModel):
        raw = shap.TreeExplainer(component.model).shap_values(row[component.features])
        shap_array = np.stack(raw, axis=-1) if isinstance(raw, list) else raw
        values = shap_array[0, :, class_idx]
        features = component.features
    else:
        return pd.DataFrame(columns=["feature", "shap_value"])

    out = pd.DataFrame({"feature": features, "shap_value": values})
    order = out["shap_value"].abs().sort_values(ascending=False).index
    return out.loc[order].head(top_n).reset_index(drop=True)


def match_row(feat: dict, home_team: str, away_team: str, date: pd.Timestamp) -> pd.DataFrame:
    """One-row frame in the shape every model's predict_proba expects:
    the engineered features plus team names/date (Dixon-Coles needs them)."""
    row = pd.DataFrame([feat])[FEATURE_COLUMNS]
    row["home_team"], row["away_team"], row["date"] = home_team, away_team, pd.Timestamp(date)
    return row


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bundle = train_app_bundle()
    save_app_bundle(bundle)
    size_kb = ARTIFACT_PATH.stat().st_size / 1024
    logger.info("Saved %s (%s, %.0f KB)", ARTIFACT_PATH, bundle["model_name"], size_kb)
