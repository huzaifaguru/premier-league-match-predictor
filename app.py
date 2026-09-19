"""Streamlit demo: pick a home/away team, see predicted outcome
probabilities and the top factors behind them.

Deliberately honest about what this is: a portfolio demonstration of a
model that, on held-out historical data, does NOT beat the bookmaker's own
odds (see the results table below). It's shown as a probability estimate
worth inspecting, not a betting signal.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import CURRENT_SEASON, MODELS_DIR, ROOT_DIR, SEASON_CODES
from src.data import load_raw_matches
from src.evaluate import explain_single_match
from src.features import FEATURE_COLUMNS, build_feature_table
from src.train import CLASSES, CalibratedXGBoostModel

CLASS_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

st.set_page_config(page_title="Premier League Match Predictor", layout="wide")


@st.cache_resource(show_spinner="Loading match history and training the model (first run only, ~30s)...")
def load_app_resources():
    raw_all = load_raw_matches(seasons=SEASON_CODES + [CURRENT_SEASON])
    features_all, state = build_feature_table(raw_all, return_state=True)

    # Train on complete seasons only — the partial in-progress season stays
    # out of training (too small, still accumulating) but IS reflected in
    # `state`, so team form/Elo used for live predictions is current.
    train_features = features_all[features_all["season"].isin(SEASON_CODES)]
    with open(MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)
    model = CalibratedXGBoostModel(**best_params["xgboost"])
    model.fit(train_features)

    current_teams = sorted(
        set(raw_all.loc[raw_all["season"] == CURRENT_SEASON, "home_team"])
        | set(raw_all.loc[raw_all["season"] == CURRENT_SEASON, "away_team"])
    )
    if len(current_teams) < 20:
        # Early in a season, not every team may have played yet — fall back
        # to the most recently completed season's roster.
        last_complete = SEASON_CODES[-1]
        current_teams = sorted(
            set(raw_all.loc[raw_all["season"] == last_complete, "home_team"])
            | set(raw_all.loc[raw_all["season"] == last_complete, "away_team"])
        )
    return model, state, current_teams


@st.cache_data
def load_results_table():
    path = ROOT_DIR / "reports" / "results_table.csv"
    return pd.read_csv(path, index_col="model") if path.exists() else None


model, state, teams = load_app_resources()

st.title("Premier League Match Outcome Predictor")
st.caption(
    "XGBoost + isotonic calibration, trained on 15 complete Premier League seasons "
    "(2010/11-2025/26) with leakage-safe rolling form, home/away split form, and Elo features."
)

results_table = load_results_table()
if results_table is not None and "bookmaker_baseline" in results_table.index and "xgboost_calibrated" in results_table.index:
    model_ll = results_table.loc["xgboost_calibrated", "log_loss"]
    market_ll = results_table.loc["bookmaker_baseline", "log_loss"]
    verdict = "does **not** beat" if model_ll >= market_ll else "**beats**"
    st.info(
        f"Honesty check: on held-out test seasons, this model {verdict} the bookmaker's own "
        f"odds (log loss {model_ll:.3f} vs {market_ll:.3f}). Treat the probabilities below as a "
        "reasonable estimate to inspect, not a betting edge — see the README for the full analysis."
    )

col1, col2, col3 = st.columns([2, 2, 1.3])
with col1:
    home_team = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away team", away_options, index=0)
with col3:
    match_date = st.date_input("Match date", value=pd.Timestamp.today())

feat = state.snapshot(home_team, away_team, pd.Timestamp(match_date))
X_row = pd.DataFrame([feat])[FEATURE_COLUMNS]
proba = model.predict_proba(X_row)[0]
pred_idx = int(proba.argmax())

st.subheader("Predicted probabilities")
prob_df = pd.DataFrame({
    "Outcome": [CLASS_LABELS[c] for c in CLASSES],
    "Probability": proba,
}).set_index("Outcome")
pcol, mcol = st.columns([2, 1])
with pcol:
    st.bar_chart(prob_df, horizontal=True)
with mcol:
    for c, p in zip(CLASSES, proba):
        st.metric(CLASS_LABELS[c], f"{p:.1%}")

st.subheader(f"Top factors behind the '{CLASS_LABELS[CLASSES[pred_idx]]}' prediction")
explanation = explain_single_match(model.base_model.model, X_row, class_idx=pred_idx)
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2ca02c" if v > 0 else "#d62728" for v in explanation["shap_value"][::-1]]
ax.barh(explanation["feature"][::-1], explanation["shap_value"][::-1], color=colors)
ax.set_xlabel(f"SHAP value (push toward '{CLASS_LABELS[CLASSES[pred_idx]]}' →, away from it ←)")
fig.tight_layout()
st.pyplot(fig)

with st.expander("Raw feature snapshot used for this prediction"):
    summary_rows = []
    for side, team in [("Home", home_team), ("Away", away_team)]:
        summary_rows.append({
            "Team": f"{team} ({side})",
            "Elo": f"{feat['elo_home_pre' if side == 'Home' else 'elo_away_pre']:.0f}",
            "Last 5 goals for/against": f"{feat[f'{side.lower()}_form5_goals_for']:.2f} / {feat[f'{side.lower()}_form5_goals_against']:.2f}",
            "Last 5 shots on target for/against": f"{feat[f'{side.lower()}_form5_shots_target_for']:.2f} / {feat[f'{side.lower()}_form5_shots_target_against']:.2f}",
            "Rest days": feat[f"rest_days_{side.lower()}"],
        })
    st.table(pd.DataFrame(summary_rows).set_index("Team"))

with st.expander("Compare against bookmaker odds (optional)"):
    st.caption("Enter decimal odds (e.g. Bet365) for this matchup to see the market's own de-vigged view alongside the model's.")
    oc1, oc2, oc3 = st.columns(3)
    odds_home = oc1.number_input("Home odds", min_value=1.01, value=None, step=0.01, format="%.2f")
    odds_draw = oc2.number_input("Draw odds", min_value=1.01, value=None, step=0.01, format="%.2f")
    odds_away = oc3.number_input("Away odds", min_value=1.01, value=None, step=0.01, format="%.2f")
    if odds_home and odds_draw and odds_away:
        inv = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
        overround = sum(inv)
        market_proba = [x / overround for x in inv]
        compare_df = pd.DataFrame({
            "Model": proba,
            "Market (de-vigged)": market_proba,
        }, index=[CLASS_LABELS[c] for c in CLASSES])
        st.bar_chart(compare_df, horizontal=True)

if results_table is not None:
    with st.expander("Full results table (held-out test seasons, 2023/24-2025/26)"):
        st.dataframe(results_table)
