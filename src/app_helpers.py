"""Data helpers for the Streamlit app: recent form, head-to-head,
standings, actual-result lookup and chart labels. Moved verbatim out of
app.py so the app file is only layout; none of these are model features
and none of them touch prediction.
"""
import pandas as pd

from src.features import STAT_DESCRIPTIONS, form_scope_text, parse_feature_name


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
