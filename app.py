"""Streamlit demo: pick a home/away team, see predicted outcome
probabilities and the top factors behind them.

Deliberately honest about what this is: a portfolio demonstration of a
model that, on held-out historical data, does NOT beat the bookmaker's own
odds (see the results table below). It's shown as a probability estimate
worth inspecting, not a betting signal.
"""
import hashlib
import html
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import CURRENT_SEASON, MODELS_DIR, ROOT_DIR, SEASON_CODES
from src.data import load_raw_matches
from src.evaluate import explain_single_match
from src.features import FEATURE_COLUMNS, build_feature_table, describe_feature
from src.train import CLASSES, CalibratedXGBoostModel

CLASS_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
BG_COLOR = "#000000"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#fdda2b"
SECONDARY_COLOR = "#2b94fd"
MUTED_COLOR = "#a0a0a0"

BADGE_PALETTE = [
    "#fdda2b", "#2b94fd", "#ff6b6b", "#4caf7d",
    "#b388ff", "#ff9f40", "#40c4c4", "#e0e0e0",
]
RESULT_COLORS = {"W": "#4caf7d", "D": "#8a8f8d", "L": "#ff6b6b"}

st.set_page_config(page_title="Premier League Match Predictor", layout="wide")

# Streamlit's native `[theme] font = "name:url"` config doesn't actually
# inject the stylesheet for an external URL in this Streamlit version
# (verified: no <link>/@import ever appears in the rendered page, font
# silently falls back), so it's loaded the traditional, reliable way instead.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    /* Exclude icon elements. Streamlit renders icons (expander arrows,
       etc.) as ligature text like "keyboard_arrow_right" in the Material
       Symbols font, so overriding font-family on them turns icons into
       visible broken text instead of glyphs. */
    *:not([data-testid="stIconMaterial"]) {
        font-family: 'Inter', sans-serif !important;
    }
    button, input, select, [data-baseweb="select"], [data-testid="stExpander"] summary {
        transition: all 250ms ease !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _dark_figure(figsize):
    """Matplotlib figures don't inherit Streamlit's theme automatically, so
    style them to match rather than render a jarring white box on a black
    page."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#3e4342")
    return fig, ax


def _team_color(team: str) -> str:
    # A stable per-team color, not Python's built-in hash() which is
    # randomized per process and would make badge colors flicker between
    # app restarts.
    idx = int(hashlib.md5(team.encode()).hexdigest(), 16) % len(BADGE_PALETTE)
    return BADGE_PALETTE[idx]


def _team_initials(team: str) -> str:
    words = [w for w in team.replace("'", "").split() if w.lower() != "and"]
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(w[0] for w in words[:3]).upper()


def team_badge_html(team: str, size: int = 40) -> str:
    color = _team_color(team)
    initials = _team_initials(team)
    text_color = "#000000" if color in ("#fdda2b", "#e0e0e0", "#40c4c4", "#ff9f40") else "#ffffff"
    return (
        f"<div style='display:inline-flex; align-items:center; justify-content:center; "
        f"width:{size}px; height:{size}px; border-radius:50%; background:{color}; "
        f"color:{text_color}; font-weight:700; font-size:{size * 0.34:.0f}px; "
        f"border:1px solid #ffffff; flex-shrink:0;'>{initials}</div>"
    )


def recent_form(raw_all: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 5) -> list[dict]:
    """Team's last n results (any venue) strictly before `before_date`, most
    recent last. Used for the form-strip badges, not a model feature."""
    mask = ((raw_all["home_team"] == team) | (raw_all["away_team"] == team)) & (raw_all["date"] < before_date)
    matches = raw_all.loc[mask].sort_values("date").tail(n)
    rows = []
    for row in matches.itertuples(index=False):
        is_home = row.home_team == team
        opponent = row.away_team if is_home else row.home_team
        team_goals = row.home_goals if is_home else row.away_goals
        opp_goals = row.away_goals if is_home else row.home_goals
        if team_goals > opp_goals:
            outcome = "W"
        elif team_goals < opp_goals:
            outcome = "L"
        else:
            outcome = "D"
        rows.append({
            "opponent": opponent, "venue": "H" if is_home else "A",
            "score": f"{int(team_goals)}-{int(opp_goals)}", "outcome": outcome,
        })
    return rows


def form_badges_html(form_rows: list[dict]) -> str:
    if not form_rows:
        return f"<span style='color:{MUTED_COLOR};'>No prior matches on record.</span>"
    chips = []
    for r in form_rows:
        color = RESULT_COLORS[r["outcome"]]
        title = html.escape(f"{r['venue']} vs {r['opponent']}: {r['score']}", quote=True)
        chips.append(
            f'<span title="{title}" '
            f"style='display:inline-flex; align-items:center; justify-content:center; "
            f"width:26px; height:26px; border-radius:6px; background:{color}; "
            f"color:#000000; font-weight:700; font-size:12px; margin-right:6px;'>{r['outcome']}</span>"
        )
    return "".join(chips)


def head_to_head(raw_all: pd.DataFrame, team_a: str, team_b: str, before_date: pd.Timestamp, n: int = 5) -> pd.DataFrame:
    mask = (
        ((raw_all["home_team"] == team_a) & (raw_all["away_team"] == team_b))
        | ((raw_all["home_team"] == team_b) & (raw_all["away_team"] == team_a))
    ) & (raw_all["date"] < before_date)
    return raw_all.loc[mask].sort_values("date", ascending=False).head(n)


def compute_standings(season_df: pd.DataFrame) -> pd.DataFrame:
    """A simple points table (3/1/0) from one season's match results, used
    to show each team's current league position. Not a model feature."""
    teams = pd.unique(season_df[["home_team", "away_team"]].to_numpy().ravel())
    rows = {t: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0} for t in teams}
    for row in season_df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        hg, ag = row.home_goals, row.away_goals
        rows[h]["played"] += 1
        rows[a]["played"] += 1
        rows[h]["gf"] += hg
        rows[h]["ga"] += ag
        rows[a]["gf"] += ag
        rows[a]["ga"] += hg
        if row.result == "H":
            rows[h]["won"] += 1
            rows[h]["points"] += 3
            rows[a]["lost"] += 1
        elif row.result == "A":
            rows[a]["won"] += 1
            rows[a]["points"] += 3
            rows[h]["lost"] += 1
        else:
            rows[h]["drawn"] += 1
            rows[a]["drawn"] += 1
            rows[h]["points"] += 1
            rows[a]["points"] += 1
    table = pd.DataFrame(rows).T
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["points", "gd", "gf"], ascending=False)
    table["position"] = range(1, len(table) + 1)
    return table


@st.cache_resource(show_spinner="Loading match history and training the model (first run only, ~30s)...")
def load_app_resources():
    raw_all = load_raw_matches(seasons=SEASON_CODES + [CURRENT_SEASON])
    features_all, state = build_feature_table(raw_all, return_state=True)

    # Train on complete seasons only. The partial in-progress season stays
    # out of training (too small, still accumulating) but IS reflected in
    # `state`, so team form/Elo used for live predictions is current.
    train_features = features_all[features_all["season"].isin(SEASON_CODES)]
    with open(MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)
    model = CalibratedXGBoostModel(**best_params["xgboost"])
    model.fit(train_features)

    # Same season powers both the team dropdown roster and the standings
    # table: early in a season (fewer than 20 matches played, roughly one
    # full gameweek) there isn't enough data for either to be meaningful,
    # so fall back to the most recently completed season for both.
    if (raw_all["season"] == CURRENT_SEASON).sum() >= 20:
        context_season = CURRENT_SEASON
    else:
        context_season = SEASON_CODES[-1]
    context_matches = raw_all[raw_all["season"] == context_season]
    current_teams = sorted(set(context_matches["home_team"]) | set(context_matches["away_team"]))
    standings_df = compute_standings(context_matches)

    return model, state, current_teams, raw_all, standings_df, context_season


@st.cache_data
def load_results_table():
    path = ROOT_DIR / "reports" / "results_table.csv"
    return pd.read_csv(path, index_col="model") if path.exists() else None


model, state, teams, raw_all, standings_df, standings_season = load_app_resources()

st.title("Premier League Match Outcome Predictor")
st.markdown(
    f"<div style='height:4px; width:72px; background:{ACCENT_COLOR}; "
    "border-radius:2px; margin:4px 0 16px 0;'></div>",
    unsafe_allow_html=True,
)
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
        "reasonable estimate to inspect, not a betting edge. See the README for the full analysis."
    )

with st.container(border=True):
    st.markdown("##### Choose a matchup")
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

with st.container(border=True):
    st.markdown("##### Team comparison")
    bcol1, bcol2 = st.columns(2)
    for col, team, side in [(bcol1, home_team, "home"), (bcol2, away_team, "away")]:
        with col:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px;'>"
                f"{team_badge_html(team)}<div><strong>{team}</strong><br>"
                f"<span style='color:{MUTED_COLOR}; font-size:13px;'>{side.capitalize()}</span></div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if team in standings_df.index:
                s = standings_df.loc[team]
                st.caption(
                    f"League position: #{int(s['position'])} of {len(standings_df)} ({standings_season}) "
                    f"· {int(s['points'])} pts · {int(s['won'])}W {int(s['drawn'])}D {int(s['lost'])}L "
                    f"· GD {int(s['gd']):+d}"
                )
            else:
                st.caption(f"Not in the {standings_season} table (promoted or unavailable).")
            elo_val = feat["elo_home_pre" if side == "home" else "elo_away_pre"]
            st.caption(f"Elo rating: {elo_val:.0f}")
            st.markdown("Last 5 results (oldest to newest):", help="Hover a badge for the opponent and score.")
            st.markdown(form_badges_html(recent_form(raw_all, team, pd.Timestamp(match_date))), unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(f"##### Head-to-head: {home_team} vs {away_team}")
    h2h = head_to_head(raw_all, home_team, away_team, pd.Timestamp(match_date))
    if h2h.empty:
        st.caption("No previous meetings on record in this dataset (2010/11 onward).")
    else:
        display_h2h = pd.DataFrame({
            "Date": h2h["date"].dt.strftime("%Y-%m-%d"),
            "Home": h2h["home_team"],
            "Score": h2h["home_goals"].astype(int).astype(str) + " - " + h2h["away_goals"].astype(int).astype(str),
            "Away": h2h["away_team"],
        })
        st.table(display_h2h.set_index("Date"))

with st.container(border=True):
    st.markdown("##### Predicted probabilities")
    prob_df = pd.DataFrame({
        "Outcome": [CLASS_LABELS[c] for c in CLASSES],
        "Probability": proba,
    }).set_index("Outcome")
    pcol, mcol = st.columns([2, 1])
    with pcol:
        st.bar_chart(prob_df, horizontal=True, color=ACCENT_COLOR)
    with mcol:
        for c, p in zip(CLASSES, proba):
            st.metric(CLASS_LABELS[c], f"{p:.1%}")

with st.container(border=True):
    st.markdown(f"##### Top factors behind the '{CLASS_LABELS[CLASSES[pred_idx]]}' prediction")
    explanation = explain_single_match(model.base_model.model, X_row, class_idx=pred_idx)
    fig, ax = _dark_figure(figsize=(8, 4))
    colors = [ACCENT_COLOR if v > 0 else SECONDARY_COLOR for v in explanation["shap_value"][::-1]]
    ax.barh(explanation["feature"][::-1], explanation["shap_value"][::-1], color=colors)
    ax.set_xlabel(f"SHAP value (push toward '{CLASS_LABELS[CLASSES[pred_idx]]}' →, away from it ←)")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Factor key: what these labels mean", expanded=True):
        for feature_name in explanation["feature"]:
            st.markdown(f"**`{feature_name}`**: {describe_feature(feature_name)}")

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
        st.bar_chart(compare_df, horizontal=True, color=[ACCENT_COLOR, SECONDARY_COLOR])

if results_table is not None:
    with st.expander("Full results table (held-out test seasons, 2023/24-2025/26)"):
        st.dataframe(results_table)
