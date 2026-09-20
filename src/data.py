"""Download and cache raw Premier League match CSVs from football-data.co.uk.

football-data.co.uk publishes one CSV per season per division. Column sets
drift slightly across seasons (extra bookmakers get added over time, date
formats change from dd/mm/yy to dd/mm/yyyy around 2019/20), so this module's
job is narrow: fetch, cache, and normalize to one stable schema. No feature
engineering happens here, that's features.py.
"""
import io
import logging
import re

import pandas as pd
import requests

from src.config import FOOTBALL_DATA_BASE_URL, RAW_DATA_DIR, SEASON_CODES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columns we actually use downstream, and the stable names we normalize them
# to. Bookmaker odds: Bet365 (B365*) is chosen because it's present across
# every season in our range (2010/11 onward); newer-only columns like Max*/
# Avg* (2019/20+) or Pinnacle are not used, to keep the odds baseline
# comparable across all seasons.
COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",       # H / D / A
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals", "result",
]


def _cache_path(season: str) -> "pathlib.Path":
    return RAW_DATA_DIR / f"E0_{season}.csv"


def download_season(season: str, force: bool = False) -> "pathlib.Path":
    """Download one season's CSV to the local cache, unless already cached."""
    path = _cache_path(season)
    if path.exists() and not force:
        return path

    url = FOOTBALL_DATA_BASE_URL.format(season=season)
    logger.info("Downloading %s -> %s", url, path)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _parse_dates(raw_dates: pd.Series) -> pd.Series:
    # football-data.co.uk uses dd/mm/yy before ~2019/20 and dd/mm/yyyy after.
    # dayfirst=True with format=None lets pandas infer per-row, which handles
    # both consistently.
    return pd.to_datetime(raw_dates, dayfirst=True, format="mixed")


def load_season(season: str, force_download: bool = False) -> pd.DataFrame:
    """Load one season's matches as a normalized DataFrame."""
    path = download_season(season, force=force_download)
    raw = pd.read_csv(path)

    available = {src: dst for src, dst in COLUMN_MAP.items() if src in raw.columns}
    missing = set(COLUMN_MAP) - set(available)
    if missing:
        logger.warning("Season %s missing columns: %s", season, sorted(missing))

    df = raw[list(available)].rename(columns=available)
    df["date"] = _parse_dates(df["date"])
    df["season"] = season

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Season {season} missing required columns: {missing_required}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["result"].isin(["H", "D", "A"])]
    return df.reset_index(drop=True)


def load_raw_matches(seasons: list[str] | None = None, force_download: bool = False) -> pd.DataFrame:
    """Load and concatenate all requested seasons into one sorted DataFrame."""
    seasons = seasons or SEASON_CODES
    frames = [load_season(s, force_download=force_download) for s in seasons]
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    if len(df) < before:
        logger.warning("Dropped %d duplicate matches", before - len(df))

    df = df.sort_values("date").reset_index(drop=True)
    return df


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

FIXTURE_COLUMN_MAP = {
    "Date": "date", "Time": "time", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
}


def load_upcoming_fixtures() -> pd.DataFrame:
    """Live upcoming-fixtures list (Premier League only), with real kickoff
    datetimes and Bet365 pre-match odds where already posted by bookmakers.

    Deliberately not disk-cached like load_season(): this data is genuinely
    live and changes day to day as fixtures are played and new ones get
    posted, unlike the historical per-season results, which are permanent.
    Returns an empty (but correctly-shaped) DataFrame if the request fails
    or no Premier League fixtures are currently listed, rather than raising,
    since callers treat "no upcoming fixtures right now" as a normal state.
    """
    columns = ["kickoff", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]
    try:
        response = requests.get(FIXTURES_URL, timeout=30, allow_redirects=True)
        response.raise_for_status()
        # Read from raw bytes with an explicit encoding, not response.text:
        # requests' auto-detected `.encoding` for this response comes back
        # ISO-8859-1 even though the file is actually UTF-8-with-BOM, which
        # mis-decodes the leading BOM bytes (EF BB BF) into three separate
        # mojibake characters instead of one clean U+FEFF. That silently
        # attaches itself to the first column's name ("Div" becomes
        # unrecognizable), breaking any lookup of "Div" by that name.
        # encoding="utf-8-sig" decodes correctly AND strips the BOM.
        raw = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
    except Exception:
        logger.warning("Could not fetch upcoming fixtures", exc_info=True)
        return pd.DataFrame(columns=columns)

    raw = raw[raw.get("Div") == "E0"]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    available = {src: dst for src, dst in FIXTURE_COLUMN_MAP.items() if src in raw.columns}
    df = raw[list(available)].rename(columns=available)
    df["kickoff"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True, format="mixed")
    df = df.drop(columns=["date", "time"])
    return df.sort_values("kickoff").reset_index(drop=True)[columns]


# football-data.co.uk's fixtures.csv only ever lists the nearest gameweek
# (confirmed empirically: it never mixes in far-future ones), which isn't
# useful for a "pick any upcoming match" selector once that gameweek has
# kicked off. openfootball's football.db project publishes the full season
# schedule up front, all ~380 matches, filled in with scores as they're
# played, as explicit public domain data, so it's the source used for
# genuinely-future fixtures instead.
FULL_SEASON_FIXTURES_URL = "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}

# A "  Wed Sep 17 2026" or "  Sat Sep 20" style line (year only repeats
# when it changes, so most lines within a season omit it).
_DATE_LINE = re.compile(r"^  \w{3}\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
# A match line either starts with "    HH:MM  " (a new kickoff time) or is
# a continuation at the same time as the previous line, indented to align
# with the team-name column instead (11-12 spaces, no time shown).
_MATCH_LINE = re.compile(
    r"^(?:\s{4}(\d{2}):(\d{2})\s{2}|\s{11,12})"
    r"(.+?)\s+v\s+(.+?)"
    r"(?:\s{2,}(\d+)-(\d+))?"
    r"\s*(?:\([^)]*\))?\s*$"
)

# openfootball spells out full club names ("Manchester United FC"); the
# rest of this project follows football-data.co.uk's shorter convention
# ("Man United"), so every source needs to agree on team names for the
# app's team-lookup logic (state.snapshot, standings, etc.) to work
# regardless of which source a given fixture came from. Hand-built from
# the current 20-team Premier League roster - needs updating whenever
# promotion/relegation changes league membership.
_FULL_TO_SHORT_TEAM_NAME = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
}


def load_full_season_fixtures(season: str = "2026-27") -> pd.DataFrame:
    """Full-season Premier League schedule (played and unplayed matches),
    parsed from openfootball's plain-text format. Scores are None for
    matches not yet played, which is what makes a fixture here usable as a
    genuinely-future matchup, unlike load_upcoming_fixtures()'s narrower
    nearest-gameweek-only list.
    """
    columns = ["kickoff", "home_team", "away_team", "home_goals", "away_goals"]
    url = FULL_SEASON_FIXTURES_URL.format(season=season)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
    except Exception:
        logger.warning("Could not fetch full-season fixtures", exc_info=True)
        return pd.DataFrame(columns=columns)

    current_year = None
    current_month_day = None
    current_time = None
    rows = []
    for raw_line in text.splitlines():
        date_match = _DATE_LINE.match(raw_line)
        if date_match:
            month_str, day_str, year_str = date_match.groups()
            if year_str:
                current_year = int(year_str)
            current_month_day = (_MONTHS[month_str], int(day_str))
            current_time = None
            continue

        match = _MATCH_LINE.match(raw_line)
        if not match or current_month_day is None or current_year is None:
            continue
        hour_str, minute_str, home, away, hg, ag = match.groups()
        if hour_str:
            current_time = (int(hour_str), int(minute_str))
        if current_time is None:
            continue

        month, day = current_month_day
        hour, minute = current_time
        rows.append({
            "kickoff": pd.Timestamp(year=current_year, month=month, day=day, hour=hour, minute=minute),
            "home_team": _FULL_TO_SHORT_TEAM_NAME.get(home.strip(), home.strip()),
            "away_team": _FULL_TO_SHORT_TEAM_NAME.get(away.strip(), away.strip()),
            "home_goals": int(hg) if hg else None,
            "away_goals": int(ag) if ag else None,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("kickoff").reset_index(drop=True)[columns]


if __name__ == "__main__":
    matches = load_raw_matches()
    logger.info(
        "Loaded %d matches across %d seasons (%s to %s)",
        len(matches), matches["season"].nunique(),
        matches["date"].min().date(), matches["date"].max().date(),
    )
    logger.info("Odds coverage: %.1f%%", matches["odds_home"].notna().mean() * 100)
