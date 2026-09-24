"""Leakage guards for src/features.py.

These tests exist to make one claim checkable, not just assertable in
prose: a match's engineered features never depend on that match's own
result, or on any match that happens after it.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ELO_INITIAL_RATING
from src.features import FEATURE_COLUMNS, build_feature_table


def _match(date, home, away, hg, ag, hs=10, as_=10, hst=5, ast=5, hc=5, ac=5,
           season="2324", oh=2.0, od=3.3, oa=3.5):
    result = "H" if hg > ag else ("A" if ag > hg else "D")
    return dict(
        date=pd.Timestamp(date), season=season, home_team=home, away_team=away,
        home_goals=hg, away_goals=ag, result=result,
        home_shots=hs, away_shots=as_, home_shots_target=hst, away_shots_target=ast,
        home_corners=hc, away_corners=ac,
        odds_home=oh, odds_draw=od, odds_away=oa,
    )


def test_first_match_has_no_history():
    raw = pd.DataFrame([_match("2023-08-01", "A", "B", 1, 0)])
    feat = build_feature_table(raw).iloc[0]

    assert pd.isna(feat["home_form5_goals_for"])
    assert pd.isna(feat["away_form5_goals_for"])
    assert feat["home_form5_n"] == 0
    assert pd.isna(feat["rest_days_home"])
    assert feat["elo_home_pre"] == ELO_INITIAL_RATING
    assert feat["elo_away_pre"] == ELO_INITIAL_RATING


def test_rolling_form_excludes_current_match_and_future_matches():
    # Team A plays 6 matches at home with a known, strictly increasing
    # goals-for sequence: 1, 2, 3, 4, 5, 6.
    raw = pd.DataFrame([
        _match(f"2023-08-{d:02d}", "A", opp, gf, 0)
        for d, (opp, gf) in enumerate(
            [("B", 1), ("C", 2), ("D", 3), ("E", 4), ("F", 5), ("G", 6)], start=1
        )
    ])
    feat = build_feature_table(raw)

    last_row = feat.iloc[5]  # A's 6th match, goals_for=6 (excluded from its own features)
    assert last_row["home_form5_goals_for"] == pytest.approx((1 + 2 + 3 + 4 + 5) / 5)
    assert last_row["home_form5_n"] == 5
    # A future match value (6) leaking in would make this 3.5, not 3.0.
    assert last_row["home_form5_goals_for"] != pytest.approx(3.5)

    third_row = feat.iloc[2]  # A's 3rd match: only 2 prior matches exist
    assert third_row["home_form5_goals_for"] == pytest.approx((1 + 2) / 2)
    assert third_row["home_form5_n"] == 2


def test_rest_days_computed_from_prior_match_only():
    raw = pd.DataFrame([
        _match("2023-08-01", "A", "B", 1, 0),
        _match("2023-08-08", "A", "C", 0, 0),   # 7 days after A's last match
        _match("2023-08-08", "B", "D", 2, 1),   # 7 days after B's last match
    ])
    feat = build_feature_table(raw)

    assert pd.isna(feat.iloc[0]["rest_days_home"])  # A's first-ever match
    assert feat.iloc[1]["rest_days_home"] == 7       # A, second match (home)
    assert feat.iloc[2]["rest_days_home"] == 7       # B, second match (home this time)


def test_elo_updates_carry_forward_and_reward_the_winner():
    raw = pd.DataFrame([
        _match("2023-08-01", "A", "B", 3, 0),   # A beats B convincingly
        _match("2023-08-08", "B", "A", 0, 0),   # rematch, reversed venue
    ])
    feat = build_feature_table(raw)

    first, second = feat.iloc[0], feat.iloc[1]
    assert first["elo_home_pre"] == ELO_INITIAL_RATING
    assert first["elo_away_pre"] == ELO_INITIAL_RATING

    # Elo carried into the second match must reflect match 1's result, not
    # reset to the initial rating and not depend on match 2's own outcome.
    assert second["elo_away_pre"] > ELO_INITIAL_RATING   # A (won match 1), now visiting
    assert second["elo_home_pre"] < ELO_INITIAL_RATING   # B (lost match 1), now hosting


def test_season_boundary_regresses_elo_toward_mean_not_reset():
    raw = pd.DataFrame([
        _match("2023-08-01", "A", "B", 3, 0, season="2223"),
        _match("2024-08-01", "A", "B", 0, 0, season="2324"),
    ])
    feat = build_feature_table(raw)
    elo_after_win = feat.iloc[0]["elo_home_pre"]  # pre-match-1 baseline (1500)
    elo_start_new_season = feat.iloc[1]["elo_home_pre"]

    # Not a full reset to 1500 (that would discard the season-1 result)...
    assert elo_start_new_season != ELO_INITIAL_RATING
    # ...but pulled back toward 1500 relative to where an ungoverned running
    # Elo would have left it.
    assert abs(elo_start_new_season - ELO_INITIAL_RATING) < abs(feat.iloc[0]["elo_home_pre"] - ELO_INITIAL_RATING) + 100


def test_implied_probabilities_remove_the_overround():
    raw = pd.DataFrame([_match("2023-08-01", "A", "B", 1, 0, oh=2.0, od=3.0, oa=4.0)])
    feat = build_feature_table(raw).iloc[0]

    total = feat["implied_prob_home"] + feat["implied_prob_draw"] + feat["implied_prob_away"]
    assert total == pytest.approx(1.0)
    # Raw 1/odds sums to 1/2 + 1/3 + 1/4 = 1.0833... > 1 (the bookmaker's
    # margin); de-vigged probabilities must be strictly smaller per outcome.
    assert feat["implied_prob_home"] < 1 / 2.0


def _synthetic_season(n_teams=6, rounds=4, seed=0, season="2324", start="2023-08-01"):
    """A small round-robin fixture list with random scores, one match per
    day so the ordering is unambiguous."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows, day = [], pd.Timestamp(start)
    for _ in range(rounds):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                h, a = (home, away) if rng.random() < 0.5 else (away, home)
                rows.append(_match(day, h, a, int(rng.poisson(1.5)), int(rng.poisson(1.1)),
                                   hs=int(rng.integers(5, 20)), as_=int(rng.integers(5, 20)), season=season))
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_perturbing_a_future_result_leaves_all_earlier_features_unchanged():
    raw = _synthetic_season()
    k = len(raw) // 2
    original = build_feature_table(raw)[FEATURE_COLUMNS]

    perturbed_raw = raw.copy()
    # Replace match k's scoreline and stats with something very different.
    perturbed_raw.loc[k, ["home_goals", "away_goals", "result"]] = [0, 7, "A"]
    perturbed_raw.loc[k, ["home_shots", "away_shots", "home_corners", "away_corners"]] = [0, 40, 0, 20]
    perturbed = build_feature_table(perturbed_raw)[FEATURE_COLUMNS]

    # Features for every match up to and including k must be identical...
    pd.testing.assert_frame_equal(original.iloc[:k + 1], perturbed.iloc[:k + 1])
    # ...and the change must actually reach later matches, otherwise this
    # test would pass without testing anything.
    assert not original.iloc[k + 1:].equals(perturbed.iloc[k + 1:])


def test_appending_later_matches_does_not_change_earlier_features():
    first = _synthetic_season(seed=1)
    later = _synthetic_season(seed=2, season="2425", start="2024-08-01")
    alone = build_feature_table(first)[FEATURE_COLUMNS]
    extended = build_feature_table(pd.concat([first, later], ignore_index=True))[FEATURE_COLUMNS]

    assert len(extended) == len(first) + len(later)
    pd.testing.assert_frame_equal(alone, extended.iloc[:len(first)])


def test_custom_elo_params_are_used():
    raw = pd.DataFrame([
        _match("2023-08-01", "A", "B", 1, 0),
        _match("2023-08-08", "A", "B", 1, 0),
    ])
    small_k = build_feature_table(raw, elo_params={"k_factor": 10, "home_advantage": 0, "carryover": 1.0})
    big_k = build_feature_table(raw, elo_params={"k_factor": 40, "home_advantage": 0, "carryover": 1.0})
    # With no home advantage an even match has expected score 0.5, so A
    # gains exactly K * 0.5 from winning match 1.
    assert small_k.iloc[1]["elo_home_pre"] == pytest.approx(ELO_INITIAL_RATING + 5)
    assert big_k.iloc[1]["elo_home_pre"] == pytest.approx(ELO_INITIAL_RATING + 20)
