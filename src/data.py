"""Download and cache raw Premier League match CSVs from football-data.co.uk.

football-data.co.uk publishes one CSV per season per division. Column sets
drift slightly across seasons (extra bookmakers get added over time, date
formats change from dd/mm/yy to dd/mm/yyyy around 2019/20), so this module's
job is narrow: fetch, cache, and normalize to one stable schema. No feature
engineering happens here, that's features.py.
"""
import io
import logging

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


if __name__ == "__main__":
    matches = load_raw_matches()
    logger.info(
        "Loaded %d matches across %d seasons (%s to %s)",
        len(matches), matches["season"].nunique(),
        matches["date"].min().date(), matches["date"].max().date(),
    )
    logger.info("Odds coverage: %.1f%%", matches["odds_home"].notna().mean() * 100)
