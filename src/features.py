"""Leakage-safe feature engineering: rolling form, Elo, rest days.

The one rule everything here obeys: a match's features are built ONLY from
information available strictly *before* that match kicked off. We enforce
this structurally, not by hoping — the code makes a single forward pass
through matches in date order, snapshotting each team's rolling stats
*before* folding the current match's result into that team's history. See
tests/test_features.py for explicit leakage-guard assertions.
"""
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
    removed' means — raw 1/odds always sums to >1 because it embeds the
    bookmaker's profit margin.
    """
    inv_home, inv_draw, inv_away = 1 / row["odds_home"], 1 / row["odds_draw"], 1 / row["odds_away"]
    overround = inv_home + inv_draw + inv_away
    return inv_home / overround, inv_draw / overround, inv_away / overround


def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Walk matches in chronological order, emitting one feature row per
    match built only from that team's history up to (not including) it.
    """
    matches = raw.sort_values(["date", "season"]).reset_index(drop=True)

    overall_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_OVERALL_WINDOW))
    home_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HOME_AWAY_FORM_WINDOW))
    away_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HOME_AWAY_FORM_WINDOW))
    last_played: dict[str, pd.Timestamp] = {}
    elo: dict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    current_season: str | None = None

    records = []
    for row in matches.itertuples(index=False):
        row = row._asdict()
        home, away, date, season = row["home_team"], row["away_team"], row["date"], row["season"]

        if current_season is not None and season != current_season:
            # New season: regress every known team's Elo toward the mean to
            # approximate squad turnover. Teams not yet seen stay untouched
            # (they'll init at ELO_INITIAL_RATING on first appearance,
            # including newly promoted teams).
            for team in list(elo.keys()):
                elo[team] = ELO_INITIAL_RATING + ELO_SEASON_CARRYOVER * (elo[team] - ELO_INITIAL_RATING)
        current_season = season

        feat: dict = {
            "date": date, "season": season,
            "home_team": home, "away_team": away,
            "result": row["result"],
            "odds_home": row.get("odds_home"), "odds_draw": row.get("odds_draw"), "odds_away": row.get("odds_away"),
        }

        # --- rolling form (snapshot BEFORE this match) ---
        for window in ROLLING_WINDOWS:
            h = _rolling_averages(overall_history[home], window)
            a = _rolling_averages(overall_history[away], window)
            for k in STAT_KEYS:
                feat[f"home_form{window}_{k}"] = h[k]
                feat[f"away_form{window}_{k}"] = a[k]
            feat[f"home_form{window}_n"] = h["n"]
            feat[f"away_form{window}_n"] = a["n"]

        hh = _rolling_averages(home_history[home], HOME_AWAY_FORM_WINDOW)
        aa = _rolling_averages(away_history[away], HOME_AWAY_FORM_WINDOW)
        for k in STAT_KEYS:
            feat[f"home_homeform{HOME_AWAY_FORM_WINDOW}_{k}"] = hh[k]
            feat[f"away_awayform{HOME_AWAY_FORM_WINDOW}_{k}"] = aa[k]
        feat[f"home_homeform{HOME_AWAY_FORM_WINDOW}_n"] = hh["n"]
        feat[f"away_awayform{HOME_AWAY_FORM_WINDOW}_n"] = aa["n"]

        # --- rest days (snapshot BEFORE this match) ---
        feat["rest_days_home"] = (date - last_played[home]).days if home in last_played else np.nan
        feat["rest_days_away"] = (date - last_played[away]).days if away in last_played else np.nan

        # --- Elo (snapshot BEFORE this match) ---
        elo_home_pre, elo_away_pre = elo[home], elo[away]
        expected_home = 1 / (1 + 10 ** (-(elo_home_pre + ELO_HOME_ADVANTAGE - elo_away_pre) / 400))
        feat["elo_home_pre"] = elo_home_pre
        feat["elo_away_pre"] = elo_away_pre
        feat["elo_diff"] = elo_home_pre + ELO_HOME_ADVANTAGE - elo_away_pre

        # --- de-vigged bookmaker implied probabilities (not a model feature —
        # kept alongside the row so evaluate.py can build the odds baseline
        # without re-joining raw data) ---
        if pd.notna(feat["odds_home"]) and pd.notna(feat["odds_draw"]) and pd.notna(feat["odds_away"]):
            feat["implied_prob_home"], feat["implied_prob_draw"], feat["implied_prob_away"] = _implied_probabilities(row)
        else:
            feat["implied_prob_home"] = feat["implied_prob_draw"] = feat["implied_prob_away"] = np.nan

        records.append(feat)

        # --- now fold this match's actual result into each team's history ---
        actual_home = 1.0 if row["result"] == "H" else (0.5 if row["result"] == "D" else 0.0)
        elo[home] = elo_home_pre + ELO_K_FACTOR * (actual_home - expected_home)
        elo[away] = elo_away_pre + ELO_K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        overall_history[home].append(_match_stats(row, is_home=True))
        overall_history[away].append(_match_stats(row, is_home=False))
        home_history[home].append(_match_stats(row, is_home=True))
        away_history[away].append(_match_stats(row, is_home=False))
        last_played[home] = date
        last_played[away] = date

    return pd.DataFrame.from_records(records)


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
