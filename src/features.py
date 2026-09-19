"""Leakage-safe feature engineering: rolling form, Elo, rest days.

The one rule everything here obeys: a match's features are built ONLY from
information available strictly *before* that match kicked off. We enforce
this structurally, not by hoping. The code makes a single forward pass
through matches in date order, snapshotting each team's rolling stats
*before* folding the current match's result into that team's history. See
tests/test_features.py for explicit leakage-guard assertions.
"""
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_SEASON_CARRYOVER,
    HOME_AWAY_FORM_WINDOW,
    ROLLING_WINDOWS,
)

STAT_KEYS = [
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
]

_MAX_OVERALL_WINDOW = max(ROLLING_WINDOWS)


def _match_stats(row: pd.Series, is_home: bool) -> dict:
    side, other = ("home", "away") if is_home else ("away", "home")
    return {
        "goals_for": row[f"{side}_goals"],
        "goals_against": row[f"{other}_goals"],
        "shots_for": row[f"{side}_shots"],
        "shots_against": row[f"{other}_shots"],
        "shots_target_for": row[f"{side}_shots_target"],
        "shots_target_against": row[f"{other}_shots_target"],
        "corners_for": row[f"{side}_corners"],
        "corners_against": row[f"{other}_corners"],
    }


def _rolling_averages(history: deque, window: int) -> dict:
    items = list(history)[-window:]
    if not items:
        return {**{k: np.nan for k in STAT_KEYS}, "n": 0}
    out = {k: sum(d[k] for d in items) / len(items) for k in STAT_KEYS}
    out["n"] = len(items)
    return out


def _implied_probabilities(row: pd.Series) -> tuple[float, float, float]:
    """De-vig bookmaker odds via the multiplicative method: divide each
    raw implied probability (1/odds) by the overround so the three
    outcomes sum to 1. This is what 'implied probability with the margin
    removed' means: raw 1/odds always sums to >1 because it embeds the
    bookmaker's profit margin.
    """
    inv_home, inv_draw, inv_away = 1 / row["odds_home"], 1 / row["odds_draw"], 1 / row["odds_away"]
    overround = inv_home + inv_draw + inv_away
    return inv_home / overround, inv_draw / overround, inv_away / overround


class LeagueState:
    """Mutable per-team state (rolling histories, Elo, last-played date) as
    of some point in time. `snapshot()` reads it (pure, never mutates) to
    produce a feature dict for a match; `advance()` folds an actual result
    in. build_feature_table() drives one LeagueState forward through
    history, calling snapshot() then advance() for every match in order,
    exactly the leakage-safe sequencing described at the top of this file.

    The app reuses the SAME snapshot() logic to featurize a hypothetical
    future matchup: build a LeagueState from all matches played so far
    (via build_feature_table(..., return_state=True)), then call
    snapshot(home, away, today) with no advance() since there's no real
    result to fold in yet. Sharing this code path (rather than re-deriving "what
    are this team's current rolling stats" separately for the app) is what
    guarantees the live app can't drift out of sync with how the model was
    actually trained.
    """

    def __init__(self):
        self.overall_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_OVERALL_WINDOW))
        self.home_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HOME_AWAY_FORM_WINDOW))
        self.away_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HOME_AWAY_FORM_WINDOW))
        self.last_played: dict[str, pd.Timestamp] = {}
        self.elo: dict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
        self.current_season: str | None = None

    def start_season_if_new(self, season: str):
        if self.current_season is not None and season != self.current_season:
            # New season: regress every known team's Elo toward the mean to
            # approximate squad turnover. Teams not yet seen stay untouched
            # (they'll init at ELO_INITIAL_RATING on first appearance,
            # including newly promoted teams).
            for team in list(self.elo.keys()):
                self.elo[team] = ELO_INITIAL_RATING + ELO_SEASON_CARRYOVER * (self.elo[team] - ELO_INITIAL_RATING)
        self.current_season = season

    def snapshot(self, home: str, away: str, date: pd.Timestamp) -> dict:
        feat: dict = {}

        for window in ROLLING_WINDOWS:
            h = _rolling_averages(self.overall_history[home], window)
            a = _rolling_averages(self.overall_history[away], window)
            for k in STAT_KEYS:
                feat[f"home_form{window}_{k}"] = h[k]
                feat[f"away_form{window}_{k}"] = a[k]
            feat[f"home_form{window}_n"] = h["n"]
            feat[f"away_form{window}_n"] = a["n"]

        hh = _rolling_averages(self.home_history[home], HOME_AWAY_FORM_WINDOW)
        aa = _rolling_averages(self.away_history[away], HOME_AWAY_FORM_WINDOW)
        for k in STAT_KEYS:
            feat[f"home_homeform{HOME_AWAY_FORM_WINDOW}_{k}"] = hh[k]
            feat[f"away_awayform{HOME_AWAY_FORM_WINDOW}_{k}"] = aa[k]
        feat[f"home_homeform{HOME_AWAY_FORM_WINDOW}_n"] = hh["n"]
        feat[f"away_awayform{HOME_AWAY_FORM_WINDOW}_n"] = aa["n"]

        feat["rest_days_home"] = (date - self.last_played[home]).days if home in self.last_played else np.nan
        feat["rest_days_away"] = (date - self.last_played[away]).days if away in self.last_played else np.nan

        elo_home_pre, elo_away_pre = self.elo[home], self.elo[away]
        feat["elo_home_pre"] = elo_home_pre
        feat["elo_away_pre"] = elo_away_pre
        feat["elo_diff"] = elo_home_pre + ELO_HOME_ADVANTAGE - elo_away_pre

        return feat

    def advance(self, row: dict):
        home, away, date = row["home_team"], row["away_team"], row["date"]

        elo_home_pre, elo_away_pre = self.elo[home], self.elo[away]
        expected_home = 1 / (1 + 10 ** (-(elo_home_pre + ELO_HOME_ADVANTAGE - elo_away_pre) / 400))
        actual_home = 1.0 if row["result"] == "H" else (0.5 if row["result"] == "D" else 0.0)
        self.elo[home] = elo_home_pre + ELO_K_FACTOR * (actual_home - expected_home)
        self.elo[away] = elo_away_pre + ELO_K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        self.overall_history[home].append(_match_stats(row, is_home=True))
        self.overall_history[away].append(_match_stats(row, is_home=False))
        self.home_history[home].append(_match_stats(row, is_home=True))
        self.away_history[away].append(_match_stats(row, is_home=False))
        self.last_played[home] = date
        self.last_played[away] = date

    @property
    def known_teams(self) -> list[str]:
        return sorted(self.elo.keys())


def build_feature_table(raw: pd.DataFrame, return_state: bool = False):
    """Walk matches in chronological order, emitting one feature row per
    match built only from that team's history up to (not including) it.
    """
    matches = raw.sort_values(["date", "season"]).reset_index(drop=True)
    state = LeagueState()

    records = []
    for row in matches.itertuples(index=False):
        row = row._asdict()
        home, away, date, season = row["home_team"], row["away_team"], row["date"], row["season"]
        state.start_season_if_new(season)

        feat = state.snapshot(home, away, date)
        feat.update({
            "date": date, "season": season,
            "home_team": home, "away_team": away,
            "result": row["result"],
            # Raw goals are carried through for the Dixon-Coles model and
            # for score-level bookkeeping, NOT included in FEATURE_COLUMNS,
            # since a match's own final score is exactly what we're
            # predicting and would be pure leakage as a model input.
            "home_goals": row["home_goals"], "away_goals": row["away_goals"],
            "odds_home": row.get("odds_home"), "odds_draw": row.get("odds_draw"), "odds_away": row.get("odds_away"),
        })

        # --- de-vigged bookmaker implied probabilities (not a model feature,
        # kept alongside the row so evaluate.py can build the odds baseline
        # without re-joining raw data) ---
        if pd.notna(feat["odds_home"]) and pd.notna(feat["odds_draw"]) and pd.notna(feat["odds_away"]):
            feat["implied_prob_home"], feat["implied_prob_draw"], feat["implied_prob_away"] = _implied_probabilities(row)
        else:
            feat["implied_prob_home"] = feat["implied_prob_draw"] = feat["implied_prob_away"] = np.nan

        records.append(feat)
        state.advance(row)

    df = pd.DataFrame.from_records(records)
    return (df, state) if return_state else df


def _feature_columns() -> list[str]:
    cols = []
    for window in ROLLING_WINDOWS:
        for side in ("home", "away"):
            cols += [f"{side}_form{window}_{k}" for k in STAT_KEYS]
            cols.append(f"{side}_form{window}_n")
    for side, tag in (("home", "homeform"), ("away", "awayform")):
        cols += [f"{side}_{tag}{HOME_AWAY_FORM_WINDOW}_{k}" for k in STAT_KEYS]
        cols.append(f"{side}_{tag}{HOME_AWAY_FORM_WINDOW}_n")
    cols += ["rest_days_home", "rest_days_away", "elo_home_pre", "elo_away_pre", "elo_diff"]
    return cols


FEATURE_COLUMNS = _feature_columns()

STAT_DESCRIPTIONS = {
    "goals_for": "goals scored",
    "goals_against": "goals conceded",
    "shots_for": "shots taken",
    "shots_against": "shots faced",
    "shots_target_for": "shots on target taken",
    "shots_target_against": "shots on target faced",
    "corners_for": "corners won",
    "corners_against": "corners conceded",
}

_SPECIAL_FEATURE_DESCRIPTIONS = {
    "elo_diff": "The gap between the two teams' Elo strength ratings, with home advantage already added in. The single biggest driver of every prediction.",
    "elo_home_pre": "The home team's Elo rating going into this match.",
    "elo_away_pre": "The away team's Elo rating going into this match.",
    "rest_days_home": "Days since the home team's previous match.",
    "rest_days_away": "Days since the away team's previous match.",
}

_FEATURE_NAME_PATTERN = re.compile(r"^(home|away)_(form|homeform|awayform)(\d+)_(.+)$")


def describe_feature(name: str) -> str:
    """Plain-English description of a FEATURE_COLUMNS entry, used to build
    a factor legend in the Streamlit app so a reader doesn't have to guess
    what e.g. 'home_homeform5_shots_target_for' means."""
    if name in _SPECIAL_FEATURE_DESCRIPTIONS:
        return _SPECIAL_FEATURE_DESCRIPTIONS[name]

    match = _FEATURE_NAME_PATTERN.match(name)
    if not match:
        return name

    side, form_type, window, stat = match.groups()
    side_label = "Home team's" if side == "home" else "Away team's"
    if form_type == "form":
        scope = f"last {window} matches, home or away"
    elif form_type == "homeform":
        scope = f"last {window} home matches"
    else:
        scope = f"last {window} away matches"

    if stat == "n":
        return f"{side_label} number of matches this average is based on (fewer than {window} early in a team's history)."
    stat_label = STAT_DESCRIPTIONS.get(stat, stat.replace("_", " "))
    return f"{side_label} average {stat_label} over its {scope}."


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.data import load_raw_matches

    raw = load_raw_matches()
    features = build_feature_table(raw)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Built {len(features)} feature rows, {len(FEATURE_COLUMNS)} features -> {out_path}")
    print(f"Rows with full elo_diff coverage: {features['elo_diff'].notna().mean():.1%}")
    print(f"Rows with full home_form5 coverage: {features['home_form5_goals_for'].notna().mean():.1%}")
