"""Streamlit demo: pick a real upcoming Premier League fixture (or a
hypothetical matchup), see a predicted result, the probabilities behind
it, and the top factors driving it.

Deliberately honest about what this is: a portfolio demonstration of a
model that, on held-out historical data, does NOT beat the bookmaker's own
odds (see the results table below). It's shown as a probability estimate
worth inspecting, not a betting signal.
"""
import base64
import hashlib
import html
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CURRENT_SEASON, MODELS_DIR, ROOT_DIR, SEASON_CODES
from src.data import load_full_season_fixtures, load_raw_matches, load_upcoming_fixtures
from src.evaluate import explain_single_match
from src.features import (
    FEATURE_COLUMNS,
    STAT_DESCRIPTIONS,
    build_feature_table,
    describe_feature,
    form_scope_text,
    parse_feature_name,
)
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

# Real crest images for all 20 current teams live in assets/crests/
# (see assets/crests/README.md), named <slug>.png using _team_slug(team)
# below (e.g. "Nott'm Forest" -> "nott_m_forest.png"). Any team without a
# matching file (a future promoted/relegated side) falls back to the
# generated shield badge below, nothing else changes.
CRESTS_DIR = ROOT_DIR / "assets" / "crests"
_CREST_MIME = {"png": "image/png", "svg": "image/svg+xml", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}

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


def _team_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")


@st.cache_data
def _load_crest_data_uri(team: str) -> str | None:
    """A base64 data URI for a locally-provided crest image, or None if no
    file exists for this team. Data URI (not a file path/URL) so it embeds
    directly in the markdown HTML with no separate static-file serving to
    configure."""
    slug = _team_slug(team)
    for ext, mime in _CREST_MIME.items():
        path = CRESTS_DIR / f"{slug}.{ext}"
        if path.exists():
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return None


def team_badge_html(team: str, size: int = 40) -> str:
    crest_uri = _load_crest_data_uri(team)
    if crest_uri:
        return (
            f"<img src='{crest_uri}' alt='{html.escape(team)} crest' "
            f"style='width:{size}px; height:{size}px; object-fit:contain; flex-shrink:0;' />"
        )

    # Fallback: a generic shield silhouette (not any real club's actual
    # crest design) reads as "team badge" at a glance without using
    # trademarked club artwork. See CRESTS_DIR above / the README for why
    # real crests aren't fetched automatically.
    color = _team_color(team)
    initials = _team_initials(team)
    text_color = "#000000" if color in ("#fdda2b", "#e0e0e0", "#40c4c4", "#ff9f40") else "#ffffff"
    return (
        f"<div style='display:inline-flex; align-items:center; justify-content:center; "
        f"width:{size}px; height:{size * 1.15:.0f}px; background:{color}; "
        f"clip-path: polygon(0% 0%, 100% 0%, 100% 62%, 50% 100%, 0% 62%); "
        f"color:{text_color}; font-weight:700; font-size:{size * 0.32:.0f}px; "
        f"padding-bottom:{size * 0.12:.0f}px; flex-shrink:0;'>{initials}</div>"
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


def lookup_actual_result(raw_all: pd.DataFrame, full_season_df: pd.DataFrame,
                          home_team: str, away_team: str, kickoff: pd.Timestamp) -> dict | None:
    """If this exact fixture has already been played AND its result has
    synced into one of the two data feeds, return the actual score.
    full_season_df (openfootball) tends to update faster; raw_all
    (football-data.co.uk's historical results) is the fallback. Returns
    None if neither has it yet, since there's nothing honest to show
    until the data catches up, not even a "check back later" placeholder.
    """
    fs_match = full_season_df[
        (full_season_df["home_team"] == home_team) & (full_season_df["away_team"] == away_team)
        & (full_season_df["kickoff"].dt.date == kickoff.date()) & full_season_df["home_goals"].notna()
    ]
    if not fs_match.empty:
        row = fs_match.iloc[0]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        result = "H" if hg > ag else ("A" if ag > hg else "D")
        return {"home_goals": hg, "away_goals": ag, "result": result}

    mask = (
        (raw_all["home_team"] == home_team) & (raw_all["away_team"] == away_team)
        & (raw_all["date"].dt.date == kickoff.date())
    )
    matches = raw_all.loc[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {"home_goals": int(row["home_goals"]), "away_goals": int(row["away_goals"]), "result": row["result"]}


def chart_label(feature_name: str, home_team: str, away_team: str) -> str:
    """Short, team-specific y-axis label for the SHAP chart. The raw
    feature names (and even the default English descriptions) use generic
    home/away roles, which is ambiguous at a glance: a reader has to
    separately remember which selected team is playing which role in this
    specific matchup. Substituting the real team names removes that step.
    """
    special = {
        "elo_diff": f"Elo gap ({home_team} vs {away_team})",
        "elo_home_pre": f"{home_team}'s Elo rating",
        "elo_away_pre": f"{away_team}'s Elo rating",
        "rest_days_home": f"{home_team}: rest days",
        "rest_days_away": f"{away_team}: rest days",
    }
    if feature_name in special:
        return special[feature_name]

    parsed = parse_feature_name(feature_name)
    if not parsed:
        return feature_name
    side, form_type, window, stat = parsed
    team = home_team if side == "home" else away_team
    scope = form_scope_text(form_type, window).replace(", home or away", "").replace(" matches", "")
    if stat == "n":
        return f"{team} ({scope}): # matches in average"
    stat_label = STAT_DESCRIPTIONS.get(stat, stat.replace("_", " "))
    return f"{team} ({scope}): {stat_label}"


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


@st.cache_data(show_spinner="Scoring the model against every match played this season...")
def build_season_comparison_table(_model, features_all: pd.DataFrame, full_season_df: pd.DataFrame) -> pd.DataFrame:
    """One row per match in the current season's full schedule: already-
    played matches get the model's pre-match prediction (from the same
    leakage-safe features used everywhere else in this app) alongside the
    actual result, so a reader can see the model's real season-long track
    record at a glance rather than one match at a time. Upcoming matches
    intentionally show no prediction here: this table is for checking the
    model against outcomes that already happened, not for forecasting
    (that's what the picker above is for).

    `_model` (leading underscore) tells Streamlit's cache not to hash it:
    it's the same fitted model object every rerun, hashing it would be
    both slow and pointless.
    """
    season_features = features_all[features_all["season"] == CURRENT_SEASON].copy()
    if not season_features.empty:
        proba = _model.predict_proba(season_features[FEATURE_COLUMNS])
        pred_idx = proba.argmax(axis=1)
        season_features["predicted_class"] = [CLASSES[i] for i in pred_idx]
        season_features["confidence"] = proba[np.arange(len(proba)), pred_idx]

    rows = []
    for r in full_season_df.itertuples(index=False):
        row = {"Date": r.kickoff.strftime("%Y-%m-%d"), "Home": r.home_team, "Away": r.away_team}
        played = pd.notna(r.home_goals)

        if not played:
            row["Predicted"] = ""
            row["Confidence"] = ""
            row["Actual result"] = "Not played yet"
            row["Correct?"] = ""
            rows.append(row)
            continue

        match = season_features[
            (season_features["home_team"] == r.home_team) & (season_features["away_team"] == r.away_team)
            & (season_features["date"].dt.date == r.kickoff.date())
        ]
        if not match.empty:
            m = match.iloc[0]
            row["Predicted"] = CLASS_LABELS[m["predicted_class"]]
            row["Confidence"] = f"{m['confidence']:.0%}"
            row["Correct?"] = "Yes" if m["predicted_class"] == m["result"] else "No"
        else:
            # Played, but not synced into the leakage-safe feature table yet
            # (same lag as the single-match lookup_actual_result()).
            row["Predicted"] = ""
            row["Confidence"] = ""
            row["Correct?"] = ""
        row["Actual result"] = f"{int(r.home_goals)} - {int(r.away_goals)}"
        rows.append(row)

    return pd.DataFrame(rows)


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

    return model, state, current_teams, raw_all, standings_df, context_season, features_all


@st.cache_data
def load_results_table():
    path = ROOT_DIR / "reports" / "results_table.csv"
    return pd.read_csv(path, index_col="model") if path.exists() else None


@st.cache_data(ttl=3600, show_spinner="Checking for live bookmaker odds...")
def load_odds_fixtures_cached():
    # Cached for an hour, not forever: this genuinely changes (odds move
    # day to day) while the app process stays up. Only used to enrich the
    # full-season fixture list with odds when a selected match falls
    # within its narrow nearest-gameweek window; see load_full_season_fixtures
    # for the actual list of selectable fixtures.
    return load_upcoming_fixtures()


@st.cache_data(ttl=3600, show_spinner="Loading the full season schedule...")
def load_full_season_fixtures_cached():
    return load_full_season_fixtures()


model, state, teams, raw_all, standings_df, standings_season, features_all = load_app_resources()

st.title("Premier League Match Outcome Predictor", anchor=False)
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

now = pd.Timestamp.now()

# football-data.co.uk's fixtures.csv (odds source) only ever lists the
# nearest gameweek, so it's not useful as the fixture LIST once that
# gameweek has kicked off. openfootball's full-season schedule (all ~380
# matches, played and unplayed) is the list source instead; odds get
# merged in from football-data.co.uk only where the two happen to overlap
# (in practice: recently-played and imminent matches, since bookmakers
# don't post odds months ahead).
odds_fixtures_df = load_odds_fixtures_cached()
full_season_df = load_full_season_fixtures_cached()

recent_played = full_season_df[(full_season_df["kickoff"] < now) & (full_season_df["kickoff"] >= now - pd.Timedelta(days=4))]
next_unplayed = full_season_df[(full_season_df["kickoff"] >= now) & full_season_df["home_goals"].isna()].head(10)
selectable_fixtures = pd.concat([recent_played, next_unplayed]).sort_values("kickoff").reset_index(drop=True)

fixture_odds = None
with st.container(border=True):
    st.header("Choose a matchup", anchor=False)
    if not selectable_fixtures.empty:
        source = st.radio(
            "Matchup source",
            ["Real fixture (recent + next gameweek)", "Custom (any two teams, hypothetical)"],
            horizontal=True, label_visibility="collapsed",
        )
    else:
        source = "Custom (any two teams, hypothetical)"
        st.caption(
            "No Premier League fixture data is currently reachable, so only the custom "
            "matchup mode is available right now."
        )

    if source.startswith("Real fixture"):
        labels = []
        for r in selectable_fixtures.itertuples(index=False):
            tag = "" if r.kickoff >= now else " (already played)"
            labels.append(f"{r.home_team} vs {r.away_team}  ·  {r.kickoff.strftime('%a %d %b, %H:%M')}{tag}")
        default_idx = int((selectable_fixtures["kickoff"] >= now).idxmax()) if (selectable_fixtures["kickoff"] >= now).any() else 0
        picked = st.selectbox("Real Premier League fixtures", labels, index=default_idx)
        fixture_row = selectable_fixtures.iloc[labels.index(picked)]
        home_team = fixture_row["home_team"]
        away_team = fixture_row["away_team"]
        match_date = fixture_row["kickoff"]

        odds_match = odds_fixtures_df[
            (odds_fixtures_df["home_team"] == home_team) & (odds_fixtures_df["away_team"] == away_team)
        ] if not odds_fixtures_df.empty else odds_fixtures_df
        if not odds_match.empty and pd.notna(odds_match.iloc[0]["odds_home"]):
            row = odds_match.iloc[0]
            fixture_odds = (row["odds_home"], row["odds_draw"], row["odds_away"])

        st.caption(
            f"Real fixture, full-season schedule via openfootball. Kickoff: "
            f"{match_date.strftime('%A %d %B %Y, %H:%M')} UK time."
        )
    else:
        col1, col2, col3 = st.columns([2, 2, 1.3])
        with col1:
            home_team = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        with col2:
            away_options = [t for t in teams if t != home_team]
            away_team = st.selectbox("Away team", away_options, index=0)
        with col3:
            match_date = pd.Timestamp(st.date_input("Match date", value=pd.Timestamp.today()))
        st.caption("Hypothetical matchup and date, not tied to a real scheduled fixture.")

feat = state.snapshot(home_team, away_team, pd.Timestamp(match_date))
X_row = pd.DataFrame([feat])[FEATURE_COLUMNS]
proba = model.predict_proba(X_row)[0]
pred_idx = int(proba.argmax())
predicted_class = CLASSES[pred_idx]
confidence = proba[pred_idx]
explanation = explain_single_match(model.base_model.model, X_row, class_idx=pred_idx)

if predicted_class == "D":
    headline = "Draw"
else:
    headline = f"{home_team if predicted_class == 'H' else away_team} to win"

top_factor_text = describe_feature(explanation.iloc[0]["feature"], home_team, away_team)

actual = None
if pd.Timestamp(match_date) <= pd.Timestamp.now():
    actual = lookup_actual_result(raw_all, full_season_df, home_team, away_team, pd.Timestamp(match_date))

with st.container(border=True):
    st.header("Prediction", anchor=False)
    # A styled callout, not a heading: this is the tool's single most
    # important output and needs to read as the visual focal point of the
    # page, but it names one specific match's result, it isn't a section
    # of the document outline, so it shouldn't be marked up as one
    # (a raw "# ..." markdown here would render a second, spurious <h1>).
    st.markdown(
        f"<div style='font-size:2.75rem; font-weight:700; line-height:1.2; "
        f"color:{TEXT_COLOR}; border-left:6px solid {ACCENT_COLOR}; "
        f"padding:4px 0 4px 20px; margin:8px 0 16px 0;'>{html.escape(headline)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"Model confidence: **{confidence:.0%}**.")
    st.markdown(f"Single biggest factor: {top_factor_text}")
    if actual is not None:
        correct = actual["result"] == predicted_class
        st.markdown(
            f"**Actual result:** {home_team} {actual['home_goals']} - {actual['away_goals']} {away_team}. "
            f"This pre-match prediction was **{'correct' if correct else 'incorrect'}**."
        )
    elif pd.Timestamp(match_date) <= pd.Timestamp.now():
        st.caption(
            "This match has already been played, but its result hasn't synced into the "
            "historical data feed yet (that usually lags kickoff by about a day)."
        )
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
    st.header("Team comparison", anchor=False)
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
    st.header(f"Head-to-head: {home_team} vs {away_team}", anchor=False)
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
    st.header(f"Top factors behind the '{CLASS_LABELS[CLASSES[pred_idx]]}' prediction", anchor=False)
    st.caption(f"Every bar below is labeled with the specific team it refers to: {home_team} (home) or {away_team} (away).")
    labels = [chart_label(f, home_team, away_team) for f in explanation["feature"]]
    fig, ax = _dark_figure(figsize=(8, 4))
    colors = [ACCENT_COLOR if v > 0 else SECONDARY_COLOR for v in explanation["shap_value"][::-1]]
    ax.barh(labels[::-1], explanation["shap_value"][::-1], color=colors)
    ax.set_xlabel(f"SHAP value (positive bars push toward '{CLASS_LABELS[CLASSES[pred_idx]]}', negative bars push away from it)")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Factor key: full detail on each factor", expanded=True):
        for feature_name, label in zip(explanation["feature"], labels):
            st.markdown(f"**{label}** (`{feature_name}`): {describe_feature(feature_name, home_team, away_team)}")

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

with st.expander("Compare against bookmaker odds", expanded=fixture_odds is not None):
    st.caption(
        "This only builds the market-comparison chart below. It doesn't change the model's own "
        "prediction or probabilities shown above, those are fixed once you pick a matchup."
    )
    if fixture_odds is not None:
        odds_home, odds_draw, odds_away = fixture_odds
        st.caption(
            f"Live Bet365 pre-match odds for this fixture: Home {odds_home:.2f} "
            f"· Draw {odds_draw:.2f} · Away {odds_away:.2f}"
        )
    else:
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

with st.container(border=True):
    st.header("Season schedule: predictions vs. actual results", anchor=False)
    st.caption(
        f"Every {CURRENT_SEASON[:2]}/{CURRENT_SEASON[2:]} Premier League match, played and upcoming. "
        "Already-played matches show what the model predicted beforehand (leakage-safe, computed the "
        "same way as every prediction above) next to what actually happened, so you can check the "
        "model's real track record yourself instead of taking the headline accuracy number on faith. "
        "Upcoming matches show the schedule only: this table is for validating against outcomes that "
        "already happened, not for forecasting."
    )
    season_table = build_season_comparison_table(model, features_all, full_season_df)
    played_rows = season_table[season_table["Actual result"] != "Not played yet"]
    scored_rows = played_rows[played_rows["Correct?"] != ""]
    if not scored_rows.empty:
        n_correct = (scored_rows["Correct?"] == "Yes").sum()
        home_goals = scored_rows["Actual result"].str.split(" - ").str[0].astype(int)
        away_goals = scored_rows["Actual result"].str.split(" - ").str[1].astype(int)
        n_actual_draws = int((home_goals == away_goals).sum())
        n_predicted_draws = int((scored_rows["Predicted"] == "Draw").sum())

        mcol1, mcol2 = st.columns(2)
        mcol1.metric(
            f"Accuracy on this season's {len(scored_rows)} played, scored matches",
            f"{n_correct / len(scored_rows):.0%}",
        )
        mcol2.metric(f"Actual draws vs. predicted draws (of {len(scored_rows)})", f"{n_actual_draws} vs. {n_predicted_draws}")
        if n_actual_draws > 0 and n_predicted_draws == 0:
            st.caption(
                f"Notice the model predicted **zero** draws even though {n_actual_draws} actually "
                "happened. This isn't a bug: it's the same 'draws are hard' finding from the offline "
                "evaluation (see the README). A draw is rarely the single most-likely outcome for any "
                "given match, even when it has a real, meaningful probability, so an argmax classifier "
                "almost never picks it. The model's *probabilities* do carry draw information; what "
                "you're seeing here is a limitation of collapsing those probabilities down to one pick."
            )
    st.dataframe(season_table, hide_index=True, height=400, width="stretch")
