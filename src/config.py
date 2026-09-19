"""Central paths and constants shared across the pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# football-data.co.uk season codes, e.g. "1516" = 2015/16.
# Earliest seasons with shot/corner columns populated consistently start ~2000/01;
# we use 2010/11 onward for a cleaner, more homogeneous feature set.
# All *complete* seasons, used for training/testing. 2025/26 finished in
# May 2026 (as of this writing, "today" is 2026-09-19) and is included.
SEASON_CODES = [
    "1011", "1112", "1213", "1314", "1415",
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
    "2526",
]

# Held out entirely for the final walk-forward test — never touched during
# feature/model iteration.
TEST_SEASONS = ["2324", "2425", "2526"]

# 2026/27 is currently in progress (a few gameweeks played). It's excluded
# from SEASON_CODES/train/test — too small a sample and still accumulating —
# but the Streamlit app fetches it separately to compute each team's
# up-to-date rolling form/Elo for live predictions.
CURRENT_SEASON = "2627"

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

ROLLING_WINDOWS = [5, 10]

ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 60.0
# Fraction of a team's Elo rating carried over between seasons; the rest
# regresses to ELO_INITIAL_RATING, approximating squad turnover in the
# transfer window (same idea FiveThirtyEight uses for its NFL Elo model).
ELO_SEASON_CARRYOVER = 0.75

# Window used for home-specific and away-specific form (separate from the
# overall ROLLING_WINDOWS, which mix home and away matches together).
HOME_AWAY_FORM_WINDOW = 5
