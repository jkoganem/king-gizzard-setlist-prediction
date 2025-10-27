"""Training constants and magic numbers.

This module centralizes all hardcoded values used in training pipelines.
"""

# Dataset filtering
RECENT_SHOWS_DATE = "2022-05-01"

# Temporal split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1  # The middle 10%
TEST_SPLIT = 0.2

# Calibration
CALIBRATION_SPLIT = 0.15

# Feature thresholds
CORE_SONG_THRESHOLD = 0.6  # Songs played in >60% of shows
NEVER_PLAYED_DAYS = 9999  # Sentinel value for "never played"
DEFAULT_PLAY_RATE = 0.1  # Fallback play rate when no data available

# Hyperparameter tuning
OPTUNA_TRIALS = 100  # Increased for comprehensive hyperparameter tuning
OPTUNA_TIMEOUT = None  # No timeout by default

# Feature engineering
TOP_COUNTRIES_COUNT = 5  # Number of top countries to track
TIME_DECAY_HALFLIFE_DAYS = 30  # Default halflife for time decay
TIME_DECAY_HALFLIFE_15 = 15  # Short-term time decay (days)
TIME_DECAY_HALFLIFE_30 = 30  # Medium-term time decay (days)
TIME_DECAY_HALFLIFE_60 = 60  # Long-term time decay (days)

# Residency detection
RESIDENCY_GAP_DAYS = 4  # Max gap between shows to consider residency
MIN_RESIDENCY_SHOWS = 2  # Minimum shows to qualify as residency

# Co-occurrence features
AFFINITY_SMOOTHING = 0.01  # Smoothing factor for affinity calculations
MIN_COOCCURRENCE_COUNT = 2  # Minimum co-occurrences to include

# ============================================================================
# SPECIALIZED SHOWS - Hardcoded date-based filtering
# ============================================================================
# Using dates instead of show IDs since IDs can change between API calls
# All dates verified to exist in database (as of Oct 2025)

# Orchestra shows - Phantom Island Tour 2025 (with Sarah Hicks)
# Source: User research + database verification
# All 8 shows confirmed in database
ORCHESTRA_SHOW_DATES = [
    "2025-07-28",  # TD Pavilion at the Mann, Philadelphia PA
    "2025-07-30",  # Westville Music Bowl, New Haven CT
    "2025-08-01",  # Forest Hills Stadium, Queens NY
    "2025-08-04",  # Merriweather Post Pavilion, Columbia MD
    "2025-08-06",  # The Pavilion (Ravinia), Highland Park IL
    "2025-08-08",  # Ford Amphitheater, Colorado Springs CO
    "2025-08-10",  # Hollywood Bowl, Los Angeles CA
    "2025-08-11",  # The Rady Shell at Jacobs Park, San Diego CA
]

# Rave show - Experimental/special format
# Confirmed in database
RAVE_SHOW_DATES = [
    "2024-11-06",  # The Regency Ballroom, San Francisco CA
]

# Festival shows - Hardcoded list of known major festivals
# Source: ChatGPT research + database verification
# 18 of 20 known festivals confirmed in database
FESTIVAL_SHOW_DATES = [
    # Coachella
    "2017-04-14",  # Coachella 2017 Weekend 1
    "2017-04-21",  # Coachella 2017 Weekend 2
    "2022-04-15",  # Coachella 2022 Weekend 1
    "2022-04-22",  # Coachella 2022 Weekend 2
    # Glastonbury
    "2017-06-24",  # Glastonbury 2017 (Secret set)
    "2017-06-25",  # Glastonbury 2017 (Main)
    # Primavera Sound 2022 (5 shows!)
    "2022-06-03",  # Primavera - Parc del Fòrum
    "2022-06-05",  # Primavera Club - Sala Apolo
    "2022-06-06",  # Primavera Club - Razzmatazz 2
    "2022-06-07",  # Primavera Club - Razzmatazz
    "2022-06-09",  # Primavera - Parc del Fòrum
    # Lollapalooza South America 2024
    "2024-03-15",  # Lolla Santiago, Chile
    "2024-03-17",  # Lolla Buenos Aires, Argentina
    "2024-03-23",  # Lolla São Paulo, Brazil
    # Other major festivals
    "2023-08-19",  # Lowlands Festival, Netherlands
    "2024-05-25",  # Wide Awake Festival, London UK
    "2018-10-13",  # Desert Daze 2018, Perris CA
    "2022-09-30",  # Desert Daze 2022, Perris CA
    # Note: Levitation 2014 and 2016 not in current database
]

# Combined specialized show lists
SPECIALIZED_SHOW_DATES = ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES
ALL_SPECIAL_DATES = ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES + FESTIVAL_SHOW_DATES

# Prediction/Inference constants
MAX_SONGS_PER_SHOW = 30  # Maximum songs in a standard show
RECENCY_DECAY_FACTOR = 0.5  # Decay factor for recency prior
LOOKBACK_WINDOW = 5  # Number of previous shows to consider
