"""Streamlit app: pick a real upcoming Premier League fixture (or any two
teams), see the model's home/draw/away probabilities, the factors behind
them, and how the model performs against bookmaker odds.

Deliberately honest about what this is: a portfolio demonstration of a
model that, on held-out historical data, does NOT beat the bookmaker's own
odds (see the Model performance tab). It's shown as a probability estimate
worth inspecting, not a betting signal.

Layout lives here; styling and HTML builders are in src/ui.py, and the
non-model data helpers (form, standings, head-to-head) in src/app_helpers.py.
"""
import html
import logging

import numpy as np
import pandas as pd
import streamlit as st

from src.app_helpers import chart_label, compute_standings, head_to_head, lookup_actual_result, recent_form
from src.app_model import explain_prediction, explanation_note, load_app_bundle, match_row, train_app_bundle
from src.config import CURRENT_SEASON, ROOT_DIR, SEASON_CODES, TEST_SEASONS
from src.data import load_full_season_fixtures, load_raw_matches, load_upcoming_fixtures
from src.features import FEATURE_COLUMNS, build_feature_table, describe_feature
from src.train import CLASSES, load_best_params
from src import ui

REPO_URL = "https://github.com/huzaifaguru/premier-league-match-predictor"
REPORTS_DIR = ROOT_DIR / "reports"
CLASS_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
MODEL_DISPLAY_NAMES = {
    "bookmaker_closing": "Bookmaker (Bet365 closing odds)",
    "bookmaker_baseline": "Bookmaker (Bet365 pre-closing odds)",
    "blend_lr_dixon_coles": "Logistic regression + Dixon-Coles blend",
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
    "dixon_coles": "Dixon-Coles (time-decayed)",
    "dixon_coles_no_decay": "Dixon-Coles (no time decay)",
    "home_win_baseline": "Always predict home win",
    "xgboost_temperature": "XGBoost + temperature scaling",
    "xgboost_multinomial": "XGBoost + multinomial scaling",
    "xgboost_isotonic": "XGBoost + isotonic calibration",
    "logistic_temperature": "Logistic regression + temperature scaling",
    "logistic_multinomial": "Logistic regression + multinomial scaling",
    "logistic_isotonic": "Logistic regression + isotonic calibration",
    "logistic_elo_only": "Logistic regression, Elo features only",
    "logistic_elo_form": "Logistic regression, Elo + rolling form",
    "logistic_all": "Logistic regression, all features",
    "xgboost_elo_only": "XGBoost, Elo features only",
    "xgboost_elo_form": "XGBoost, Elo + rolling form",
    "xgboost_all": "XGBoost, all features",
}
MAIN_TABLE_MODELS = ["bookmaker_closing", "bookmaker_baseline", "logistic_regression", "blend_lr_dixon_coles",
                     "xgboost", "dixon_coles", "dixon_coles_no_decay", "home_win_baseline"]

st.set_page_config(
    page_title="Premier League match predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " "))


def season_label(code: str) -> str:
    return f"20{code[:2]}/{code[2:]}"


# ---------------------------------------------------------------------------
# Cached data and model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Scoring the model against every match played this season...")
def build_season_comparison_table(_model, features_all: pd.DataFrame, full_season_df: pd.DataFrame) -> pd.DataFrame:
    """One row per match in the current season's full schedule: already-
    played matches get the model's pre-match prediction (from the same
    leakage-safe features used everywhere else in this app) alongside the
    actual result, so a reader can see the model's real season-long track
    record at a glance rather than one match at a time. Upcoming matches
    intentionally show no prediction here: this table is for checking the
    model against outcomes that already happened, not for forecasting.

    `_model` (leading underscore) tells Streamlit's cache not to hash it:
    it's the same fitted model object every rerun, hashing it would be
    both slow and pointless.
    """
    season_features = features_all[features_all["season"] == CURRENT_SEASON].copy()
    if not season_features.empty:
        proba = _model.predict_proba(season_features)
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
            # Icon + text, not color alone: scannable at a glance and still
            # readable/announced correctly without relying on color.
            row["Correct?"] = "✅ Yes" if m["predicted_class"] == m["result"] else "❌ No"
        else:
            # Played, but not synced into the leakage-safe feature table yet
            # (same lag as the single-match lookup_actual_result()).
            row["Predicted"] = ""
            row["Confidence"] = ""
            row["Correct?"] = ""
        row["Actual result"] = f"{int(r.home_goals)} - {int(r.away_goals)}"
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading match history and the model (first run only)...")
def load_app_resources():
    # The committed artifact (models/app_model.joblib) avoids retraining on
    # every cold start; if it's missing or was pickled under different
    # library versions, train once here instead (cached for the process).
    bundle = load_app_bundle()
    elo_params = bundle["elo_params"] if bundle else load_best_params().get("elo")

    raw_all = load_raw_matches(seasons=SEASON_CODES)
    try:
        current = load_raw_matches(seasons=[CURRENT_SEASON])
        raw_all = pd.concat([raw_all, current], ignore_index=True).sort_values("date").reset_index(drop=True)
    except Exception:
        # The in-progress season is a nice-to-have (current form/Elo), not
        # a requirement: predictions still work from last season's end state.
        logging.getLogger(__name__).warning("Could not load the current season", exc_info=True)
    features_all, state = build_feature_table(raw_all, return_state=True, elo_params=elo_params)

    # Trained on complete seasons only. The partial in-progress season stays
    # out of training (too small, still accumulating) but IS reflected in
    # `state`, so team form/Elo used for live predictions is current.
    if bundle is None:
        bundle = train_app_bundle(features_all)
    model = bundle["model"]

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

    return model, bundle["model_name"], state, current_teams, raw_all, standings_df, context_season, features_all


@st.cache_data
def load_report_csv(name: str, index_col: str | None = None) -> pd.DataFrame | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        logging.getLogger(__name__).warning("Could not read %s", path, exc_info=True)
        return None


@st.cache_data(ttl=3600, show_spinner="Checking for live bookmaker odds...")
def load_odds_fixtures_cached():
    # Cached for an hour, not forever: odds move day to day while the app
    # process stays up. Only used to enrich the full-season fixture list
    # with odds when a selected match falls in its nearest-gameweek window.
    return load_upcoming_fixtures()


@st.cache_data(ttl=3600, show_spinner="Loading the full season schedule...")
def load_full_season_fixtures_cached():
    return load_full_season_fixtures()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(ui.hero_html(
    "Premier League match predictor",
    f"Home, draw and away probabilities from {len(SEASON_CODES)} seasons of results, form and Elo ratings, "
    "benchmarked honestly against the bookmakers.",
), unsafe_allow_html=True)

try:
    model, model_name, state, teams, raw_all, standings_df, standings_season, features_all = load_app_resources()
except Exception as exc:  # any startup failure should become a readable message, not a traceback
    logging.getLogger(__name__).exception("Startup failed")
    st.error(
        "The app couldn't load its match data or model, so it can't make predictions right now. "
        "The most common cause is football-data.co.uk rate-limiting or being unreachable; "
        "wait a minute and reload the page.\n\n"
        f"Details: `{type(exc).__name__}: {exc}`"
    )
    st.stop()

results_table = load_report_csv("results_table.csv", index_col="model")


def bookmaker_verdict() -> dict | None:
    """Compare the served model with the bookmaker using the saved results
    (never hardcoded). None if the results file isn't available."""
    if results_table is None or not {"bookmaker_baseline", model_name} <= set(results_table.index):
        return None
    r = results_table.loc[model_name]
    out = {"model_ll": r["log_loss"], "market_ll": results_table.loc["bookmaker_baseline", "log_loss"]}
    if "log_loss_diff_lo" in r:
        out.update(diff=r["log_loss_diff_vs_bookmaker"], lo=r["log_loss_diff_lo"], hi=r["log_loss_diff_hi"])
        out["verdict"] = "worse than" if out["lo"] > 0 else ("better than" if out["hi"] < 0 else "not distinguishable from")
    else:
        out["verdict"] = "worse than" if out["model_ll"] >= out["market_ll"] else "better than"
    return out


verdict = bookmaker_verdict()

tab_predict, tab_perf, tab_how = st.tabs(["⚽ Predict", "\U0001F4CA Performance", "\U0001F4D6 How it works"])

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------

with tab_predict:
    now = pd.Timestamp.now()

    # football-data.co.uk's fixtures.csv (odds source) only ever lists the
    # nearest gameweek, so openfootball's full-season schedule is the list
    # source; odds get merged in only where the two overlap.
    odds_fixtures_df = load_odds_fixtures_cached()
    full_season_df = load_full_season_fixtures_cached()

    recent_played = full_season_df[(full_season_df["kickoff"] < now) & (full_season_df["kickoff"] >= now - pd.Timedelta(days=4))]
    next_unplayed = full_season_df[(full_season_df["kickoff"] >= now) & full_season_df["home_goals"].isna()].head(10)
    selectable_fixtures = pd.concat([recent_played, next_unplayed]).sort_values("kickoff").reset_index(drop=True)

    CUSTOM, FIXTURE = "Pick any two teams", "Upcoming fixtures"
    if not selectable_fixtures.empty:
        source = st.radio("Matchup source", [CUSTOM, FIXTURE], horizontal=True, label_visibility="collapsed")
    else:
        source = CUSTOM
        st.caption("Live fixture data isn't reachable right now, so only custom matchups are available.")

    fixture_odds = None
    home_team = away_team = None
    match_date = None
    when_text = ""

    if source == FIXTURE:
        labels = []
        for r in selectable_fixtures.itertuples(index=False):
            tag = "" if r.kickoff >= now else " (played)"
            labels.append(f"{r.home_team} vs {r.away_team}  ·  {r.kickoff.strftime('%a %d %b, %H:%M')}{tag}")
        default_idx = int((selectable_fixtures["kickoff"] >= now).idxmax()) if (selectable_fixtures["kickoff"] >= now).any() else 0
        picked = st.selectbox("Fixture (recent and next gameweek)", labels, index=default_idx)
        fixture_row = selectable_fixtures.iloc[labels.index(picked)]
        home_team = fixture_row["home_team"]
        away_team = fixture_row["away_team"]
        match_date = fixture_row["kickoff"]
        when_text = match_date.strftime("%a %d %b %Y, %H:%M UK")
    else:
        c_home, c_vs, c_away = st.columns([5, 1, 5], vertical_alignment="center")
        with c_home:
            with st.container(border=True):
                slot = st.empty()
                home_team = st.selectbox("Home team", teams, index=None, placeholder="Choose the home team")
                slot.markdown(ui.team_head_html(home_team, "Home"), unsafe_allow_html=True)
        with c_vs:
            st.markdown(ui.vs_html(), unsafe_allow_html=True)
        with c_away:
            with st.container(border=True):
                slot = st.empty()
                away_options = [t for t in teams if t != home_team]
                away_team = st.selectbox("Away team", away_options, index=None, placeholder="Choose the away team")
                slot.markdown(ui.team_head_html(away_team, "Away"), unsafe_allow_html=True)
        if home_team and away_team:
            # Each team hosts each other team exactly once a season, so the
            # home/away pairing identifies one fixture on the schedule: take
            # its real kickoff instead of asking for a date.
            scheduled = full_season_df[
                (full_season_df["home_team"] == home_team) & (full_season_df["away_team"] == away_team)
            ] if not full_season_df.empty else full_season_df
            if not scheduled.empty:
                fixture_row = scheduled.iloc[0]
                match_date = fixture_row["kickoff"]
                when_text = match_date.strftime("%a %d %b %Y, %H:%M UK")
                status = "Played" if pd.notna(fixture_row["home_goals"]) else "Kickoff"
                st.caption(f"\U0001F4C5 {status}: {when_text}, from the {season_label(CURRENT_SEASON)} "
                           "Premier League fixture list.")
            else:
                # Not on this season's schedule (or the schedule is offline):
                # fall back to asking, and say why.
                st.caption(f"This matchup isn't on the {season_label(CURRENT_SEASON)} fixture list, "
                           "so pick a date for a hypothetical match.")
                d_col, _ = st.columns([1, 2])
                with d_col:
                    match_date = pd.Timestamp(st.date_input("Match date", value=pd.Timestamp.today()))
                when_text = f"{match_date.strftime('%a %d %b %Y')} · hypothetical"

    # Live Bet365 odds, when bookmakers have posted them for this fixture
    # (only the nearest gameweek, in practice).
    if home_team and away_team and not odds_fixtures_df.empty:
        odds_match = odds_fixtures_df[
            (odds_fixtures_df["home_team"] == home_team) & (odds_fixtures_df["away_team"] == away_team)
        ]
        if not odds_match.empty and pd.notna(odds_match.iloc[0]["odds_home"]):
            row = odds_match.iloc[0]
            fixture_odds = (row["odds_home"], row["odds_draw"], row["odds_away"])

    if not (home_team and away_team):
        st.markdown(ui.empty_state_html(
            "Pick two teams to see the prediction",
            "Choose a home and an away team above, or switch to Upcoming fixtures to pick a real match.",
        ), unsafe_allow_html=True)
    else:
        try:
            feat = state.snapshot(home_team, away_team, pd.Timestamp(match_date))
            X_row = match_row(feat, home_team, away_team, pd.Timestamp(match_date))
            proba = model.predict_proba(X_row)[0]
            pred_idx = int(proba.argmax())
            predicted_class = CLASSES[pred_idx]
            explanation = explain_prediction(model, X_row, class_idx=pred_idx)
        except Exception:
            logging.getLogger(__name__).exception("Prediction failed for %s vs %s", home_team, away_team)
            st.error("Something went wrong making this prediction. Try a different matchup or date.")
            st.stop()

        top_factor_text = describe_feature(explanation.iloc[0]["feature"], home_team, away_team) if not explanation.empty else None

        actual = None
        if pd.Timestamp(match_date) <= pd.Timestamp.now():
            actual = lookup_actual_result(raw_all, full_season_df, home_team, away_team, pd.Timestamp(match_date))

        extra = []
        if actual is not None:
            correct = actual["result"] == predicted_class
            extra.append(
                f"<strong>Final score:</strong> {html.escape(home_team)} {actual['home_goals']} - "
                f"{actual['away_goals']} {html.escape(away_team)}. The pre-match pick was "
                f"<strong>{'right' if correct else 'wrong'}</strong>."
            )
        elif pd.Timestamp(match_date) <= pd.Timestamp.now():
            extra.append("Already played; the result usually reaches the data feed about a day after kickoff.")
        if top_factor_text:
            extra.append(f"<strong>Biggest factor:</strong> {html.escape(top_factor_text)}")

        st.markdown(ui.match_card_html(home_team, away_team, when_text, proba, CLASSES, extra), unsafe_allow_html=True)
        if verdict:
            st.caption(
                f"For context: on {len(TEST_SEASONS)} held-out seasons this model scored {verdict['verdict']} "
                "the bookmaker's own odds. See the Performance tab."
            )

        # --- Supporting stats (all already computed for this prediction) ---
        def rest_text(v):
            return "n/a" if pd.isna(v) else f"{int(v)} days"

        def rest_short(v):
            return "n/a" if pd.isna(v) else f"{int(v)}"

        st.markdown(ui.stat_cards_html([
            (f"{home_team} Elo", f"{feat['elo_home_pre']:.0f}", "team strength rating"),
            (f"{away_team} Elo", f"{feat['elo_away_pre']:.0f}", "team strength rating"),
            ("Elo gap", f"{feat['elo_home_pre'] - feat['elo_away_pre']:+.0f}", "home minus away, before home advantage"),
            ("Rest days", f"{rest_short(feat['rest_days_home'])} / {rest_short(feat['rest_days_away'])}",
             f"{home_team} / {away_team}"),
        ]), unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        for col, team in ((f1, home_team), (f2, away_team)):
            with col:
                with st.container(border=True):
                    st.markdown(ui.team_head_html(team, "Last 5 results, oldest to newest"), unsafe_allow_html=True)
                    st.markdown(ui.form_pills_html(recent_form(raw_all, team, pd.Timestamp(match_date))),
                                unsafe_allow_html=True)
                    if team in standings_df.index:
                        s = standings_df.loc[team]
                        st.caption(
                            f"#{int(s['position'])} of {len(standings_df)} in {season_label(standings_season)} · "
                            f"{int(s['points'])} pts · {int(s['won'])}W {int(s['drawn'])}D {int(s['lost'])}L · "
                            f"GD {int(s['gd']):+d}"
                        )
                    else:
                        st.caption(f"Not in the {season_label(standings_season)} table (promoted or unavailable).")

        with st.expander(f"Head-to-head: last meetings of {home_team} and {away_team}"):
            h2h = head_to_head(raw_all, home_team, away_team, pd.Timestamp(match_date))
            if h2h.empty:
                st.caption("No previous meetings on record in this dataset (2010/11 onward).")
            else:
                st.dataframe(pd.DataFrame({
                    "Date": h2h["date"].dt.strftime("%Y-%m-%d"),
                    "Home": h2h["home_team"],
                    "Score": h2h["home_goals"].astype(int).astype(str) + " - " + h2h["away_goals"].astype(int).astype(str),
                    "Away": h2h["away_team"],
                }), hide_index=True, width="stretch")

        with st.expander("Compare with bookmaker odds", expanded=fixture_odds is not None):
            st.caption("This only builds the comparison chart. It doesn't change the model's probabilities above.")
            if fixture_odds is not None:
                odds_home, odds_draw, odds_away = fixture_odds
                st.caption(f"Live Bet365 pre-match odds: home {odds_home:.2f} · draw {odds_draw:.2f} · away {odds_away:.2f}")
            else:
                st.caption("Enter decimal odds (e.g. Bet365) to see the market's de-vigged view next to the model's.")
                oc1, oc2, oc3 = st.columns(3)
                odds_home = oc1.number_input("Home odds", min_value=1.01, value=None, step=0.01, format="%.2f")
                odds_draw = oc2.number_input("Draw odds", min_value=1.01, value=None, step=0.01, format="%.2f")
                odds_away = oc3.number_input("Away odds", min_value=1.01, value=None, step=0.01, format="%.2f")
            if odds_home and odds_draw and odds_away:
                inv = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
                market_proba = [x / sum(inv) for x in inv]
                compare_df = pd.DataFrame({"Model": proba, "Market (de-vigged)": market_proba},
                                          index=[CLASS_LABELS[c] for c in CLASSES])
                st.bar_chart(compare_df, horizontal=True, color=[ui.HOME, ui.AWAY])

        with st.expander("Why this prediction? (model details)"):
            if explanation.empty:
                st.caption("No per-feature explanation is available for this model type.")
            else:
                note = explanation_note(model)
                if note:
                    st.caption(note)
                labels = [chart_label(f, home_team, away_team) for f in explanation["feature"]]
                fig, ax = ui.themed_figure(figsize=(8, 4))
                colors = [ui.HOME if v > 0 else ui.AWAY for v in explanation["shap_value"][::-1]]
                ax.barh(labels[::-1], explanation["shap_value"][::-1], color=colors)
                ax.set_xlabel(f"Contribution in log-odds (right pushes toward '{CLASS_LABELS[predicted_class]}')")
                fig.tight_layout()
                st.pyplot(fig)
                for feature_name, label in zip(explanation["feature"], labels):
                    st.markdown(f"**{label}**: {describe_feature(feature_name, home_team, away_team)}")

            st.markdown("**Raw feature snapshot used for this prediction**")
            summary_rows = []
            for side, team in (("Home", home_team), ("Away", away_team)):
                s_ = side.lower()
                summary_rows.append({
                    "Team": f"{team} ({side})",
                    "Elo": f"{feat[f'elo_{s_}_pre']:.0f}",
                    "Last 5 goals for/against": f"{feat[f'{s_}_form5_goals_for']:.2f} / {feat[f'{s_}_form5_goals_against']:.2f}",
                    "Last 5 shots on target for/against": f"{feat[f'{s_}_form5_shots_target_for']:.2f} / {feat[f'{s_}_form5_shots_target_against']:.2f}",
                    "Rest days": rest_text(feat[f"rest_days_{s_}"]),
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    with st.expander(f"League table ({season_label(standings_season)})"):
        st.caption("Computed from the same results used throughout the app: 3 points for a win, 1 for a draw, "
                   "sorted by points, then goal difference, then goals scored.")
        table_display = standings_df.reset_index().rename(columns={"index": "Team"})
        table_display = table_display[["position", "Team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]]
        table_display.columns = ["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
        table_display["GD"] = table_display["GD"].astype(int).map(lambda x: f"{x:+d}")
        st.dataframe(table_display, hide_index=True, width="stretch", height=740)

# ---------------------------------------------------------------------------
# Model performance tab: everything here is read from reports/ files.
# ---------------------------------------------------------------------------

def fmt_ci(row, metric, digits=3):
    if f"{metric}_lo" in row and pd.notna(row[f"{metric}_lo"]):
        return f"{row[metric]:.{digits}f}  [{row[f'{metric}_lo']:.{digits}f}, {row[f'{metric}_hi']:.{digits}f}]"
    return f"{row[metric]:.{digits}f}"


def fmt_diff(row, metric="log_loss"):
    if f"{metric}_diff_lo" not in row or pd.isna(row.get(f"{metric}_diff_lo")):
        return ""
    return f"{row[f'{metric}_diff_vs_bookmaker']:+.3f}  [{row[f'{metric}_diff_lo']:+.3f}, {row[f'{metric}_diff_hi']:+.3f}]"


def results_display(table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        if m not in table.index:
            continue
        r = table.loc[m]
        degenerate = m == "home_win_baseline"  # 100/0/0 predictions: log loss isn't meaningful
        rows.append({
            "Model": display_name(m) + (" (served in this app)" if m == model_name else ""),
            "Accuracy": f"{r['accuracy']:.1%}",
            "Log loss [95% CI]": "not meaningful" if degenerate else fmt_ci(r, "log_loss"),
            "Brier": f"{r['brier' if 'brier' in r else 'brier_score']:.3f}",
            "RPS": f"{r['rps']:.3f}" if "rps" in r else "",
            "Log loss vs bookmaker [95% CI]": "" if degenerate or m == "bookmaker_baseline" else fmt_diff(r),
        })
    return pd.DataFrame(rows)


with tab_perf:
    if results_table is None:
        st.info("The evaluation results file (reports/results_table.csv) isn't available, so there's nothing to show here.")
    else:
        test_span = f"{season_label(TEST_SEASONS[0])} to {season_label(TEST_SEASONS[-1])}"
        n_matches = int(results_table["n_matches"].iloc[0]) if "n_matches" in results_table else None
        st.subheader("How good is it? Scored on seasons it never saw", anchor=False)
        st.caption(
            f"Every model was tuned on earlier seasons only, then scored walk-forward on {test_span}"
            + (f" ({n_matches:,} matches)." if n_matches else ".")
            + " Lower log loss, Brier and RPS are better."
        )

        others = [m for m in results_table.index if m not in ("bookmaker_baseline", "bookmaker_closing", "home_win_baseline")]
        if "log_loss_diff_lo" in results_table and others:
            n_worse = int((results_table.loc[others, "log_loss_diff_lo"] > 0).sum())
            if n_worse == len(others):
                st.warning(
                    f"**The bookmaker wins.** All {len(others)} models score worse than Bet365's own de-vigged odds, "
                    "and in every case the 95% confidence interval of the difference excludes zero. The model is a "
                    "reasonable estimate, not a way to beat the market."
                )
            else:
                st.info(f"{len(others) - n_worse} of {len(others)} models are not clearly worse than the bookmaker's odds.")

        if verdict:
            k1, k2, k3 = st.columns(3)
            k1.metric(f"{display_name(model_name)}: log loss", f"{verdict['model_ll']:.3f}", border=True)
            k2.metric("Bookmaker odds: log loss", f"{verdict['market_ll']:.3f}", border=True)
            if "diff" in verdict:
                k3.metric("Difference (model minus bookmaker)", f"{verdict['diff']:+.3f}",
                          delta=f"95% CI {verdict['lo']:+.3f} to {verdict['hi']:+.3f}", delta_color="off",
                          border=True)

        st.dataframe(results_display(results_table, MAIN_TABLE_MODELS), hide_index=True, width="stretch")
        st.caption("RPS is the ranked probability score, which treats home / draw / away as ordered. The confidence "
                   "intervals come from a paired bootstrap over matches. The 'always predict home win' baseline "
                   "puts 0% on every draw and away win, so its log loss isn't a fair comparison; its accuracy is.")
        calib_models = [m for m in results_table.index if m not in MAIN_TABLE_MODELS]
        if calib_models:
            with st.expander("Calibration variants"):
                st.dataframe(results_display(results_table, calib_models), hide_index=True, width="stretch")

        ablation = load_report_csv("ablation_table.csv", index_col="model")
        if ablation is not None:
            st.subheader("Where the signal comes from", anchor=False)
            st.caption("A feature ablation: the same models with only Elo ratings, Elo plus rolling form, and all "
                       f"{len(FEATURE_COLUMNS)} features, each tuned separately.")
            st.dataframe(results_display(ablation, [m for m in ablation.index if m != "bookmaker_baseline"]),
                         hide_index=True, width="stretch")

        def report_plot(fname, title, caption):
            path = REPORTS_DIR / fname
            if path.exists():
                st.subheader(title, anchor=False)
                st.image(str(path), width="stretch")
                st.caption(caption)

        # The wide three-panel calibration plot gets the full width; the two
        # taller charts sit side by side on desktop and stack on a phone.
        report_plot("calibration_plot.png", "Calibration: predicted probability vs how often it happened",
                    "Points near the dashed diagonal mean the probabilities can be taken at face value.")
        p1, p2 = st.columns(2)
        with p1:
            report_plot("shap_importance.png", "Feature importance (XGBoost, SHAP)",
                        "The Elo rating gap dominates. SHAP splits credit between correlated Elo features, "
                        "so the ablation above is the cleaner evidence.")
        with p2:
            report_plot("kelly_backtest.png", "Paper-money betting backtest",
                        "Staking whenever the model disagrees with the odds loses steadily: the apparent edges "
                        "are noise.")

        st.markdown(f"Full numbers, including every pairwise comparison: "
                    f"[evaluation report]({REPO_URL}/blob/main/reports/evaluation_report.md).")

    st.subheader(f"This season so far ({season_label(CURRENT_SEASON)})", anchor=False)
    st.caption("Every match this season: already-played matches show the model's pre-match pick next to what "
               "actually happened, so the track record can be checked directly.")
    season_table = build_season_comparison_table(model, features_all, full_season_df)
    played_rows = season_table[season_table["Actual result"] != "Not played yet"] if not season_table.empty else season_table
    scored_rows = played_rows[played_rows["Correct?"] != ""] if not played_rows.empty else played_rows
    if not scored_rows.empty:
        n_correct = scored_rows["Correct?"].str.endswith("Yes").sum()
        home_goals = scored_rows["Actual result"].str.split(" - ").str[0].astype(int)
        away_goals = scored_rows["Actual result"].str.split(" - ").str[1].astype(int)
        s1, s2 = st.columns(2)
        s1.metric(f"Correct picks ({len(scored_rows)} played)", f"{n_correct / len(scored_rows):.0%}", border=True)
        s2.metric("Actual draws vs predicted draws",
                  f"{int((home_goals == away_goals).sum())} vs {int((scored_rows['Predicted'] == 'Draw').sum())}",
                  border=True, help="Draws are rarely the single most likely outcome, so the top pick is almost "
                                    "never a draw even when the draw probability is substantial.")
    if not season_table.empty:
        st.dataframe(season_table, hide_index=True, height=400, width="stretch")
    else:
        st.caption("The season schedule isn't reachable right now.")

# ---------------------------------------------------------------------------
# How it works tab
# ---------------------------------------------------------------------------

with tab_how:
    first, last = season_label(SEASON_CODES[0]), season_label(SEASON_CODES[-1])
    st.subheader("How it works", anchor=False)
    st.markdown(f"""
**Data.** Every Premier League result from {first} to {last} ({len(SEASON_CODES)} seasons) from
[football-data.co.uk](https://www.football-data.co.uk/): goals, shots, shots on target, corners and Bet365 odds.
The current season is added as it's played, so team form and ratings stay up to date.

**Features.** For each match, {len(FEATURE_COLUMNS)} numbers built only from what was known *before* kickoff:
each team's recent form (last 5 and 10 matches), home-only and away-only form, rest days, and an Elo rating
that goes up or down after every result. Tests in the repo check that no match's features can see its own
result or any later one.

**Model.** The app serves the **{display_name(model_name)}**, the model with the best score in walk-forward
cross-validation on the earlier seasons. Walk-forward means always training on the past and testing on the
next season, never a random shuffle.

**Honest testing.** The last {len(TEST_SEASONS)} seasons ({season_label(TEST_SEASONS[0])} to
{season_label(TEST_SEASONS[-1])}) were never used for tuning or choosing the model. They're only used to score
the finished models, against the bookmaker's own odds with the margin removed.
""")
    st.subheader("Limitations", anchor=False)
    st.markdown("""
- **No lineups, injuries or news.** The bookmaker knows who's playing; this model doesn't. That's the main
  reason it can't beat the odds.
- **Draws are almost never the top pick.** They rarely have the single highest probability, so read the full
  home / draw / away bar, not just the headline.
- **Newly promoted teams start from an average rating**, so early-season predictions for them are rough.
- **Picked matchups take their date from the fixture list**, so rest days are real. Pairings that aren't on
  it (or when the schedule is offline) ask for a date instead, and use each team's current form and rating.
- **Not betting advice.** A backtest of betting on the model's "edges" lost money steadily.
""")

st.markdown(ui.footer_html(REPO_URL), unsafe_allow_html=True)
