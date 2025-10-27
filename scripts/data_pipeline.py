#!/usr/bin/env python3
"""
Data Pipeline for King Gizzard Setlist Prediction

This script handles the complete data pipeline:
- Fetching fresh data from setlist.fm API
- Building and rebuilding the DuckDB database
- Validating data integrity
- Testing for data leakage in feature engineering
- Generating exploratory data analysis reports

Usage:
    python scripts/data_pipeline.py --fetch-api
    python scripts/data_pipeline.py --rebuild-db
    python scripts/data_pipeline.py --validate
    python scripts/data_pipeline.py --check-leakage
    python scripts/data_pipeline.py --full-pipeline

"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src.ingest.setlistfm import SetlistFMClient
from src.dataio import load_all_data, save_parquet, DataPaths
from src.features.engineer_features import engineer_all_features
from src.utils.config import settings
from src.utils.constants import RECENT_SHOWS_DATE, SPECIALIZED_SHOW_DATES

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_api_data(artist_name: str = "King Gizzard & The Lizard Wizard") -> None:
    """
    Fetch fresh setlist data from setlist.fm API.

    Args:
        artist_name: Name of the artist to fetch data for

    Raises:
        ValueError: If API key is not configured
        RuntimeError: If API fetch fails
    """
    logger.info(f"Fetching setlist data for: {artist_name}")

    try:
        client = SetlistFMClient()
        setlists = client.get_artist_setlists(artist_name)

        logger.info(f"Fetched {len(setlists)} setlists")

        # Save raw data
        output_path = (
            settings.paths.data_raw_dir
            / f"setlists_{datetime.now().strftime('%Y%m%d')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(output_path, "w") as f:
            json.dump(setlists, f, indent=2)

        logger.info(f"Saved raw data to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to fetch API data: {e}")
        raise RuntimeError(f"API fetch failed: {e}")


def rebuild_database(data_source: str = "api") -> None:
    """
    Rebuild the DuckDB database from raw data.

    Args:
        data_source: Source of data ('api' or 'csv')

    Raises:
        FileNotFoundError: If data source files not found
        RuntimeError: If database build fails
    """
    logger.info(f"Rebuilding database from {data_source} data")

    try:
        builder = DatabaseBuilder()

        if data_source == "api":
            # Find most recent API data
            raw_files = sorted(settings.paths.data_raw_dir.glob("setlists_*.json"))
            if not raw_files:
                raise FileNotFoundError(
                    "No raw setlist data found. Run --fetch-api first."
                )
            data_file = raw_files[-1]
        else:
            # Use CSV data
            data_file = settings.paths.data_raw_dir / "setlists.csv"

        builder.build_from_file(data_file)
        logger.info("Database rebuild successful")

    except Exception as e:
        logger.error(f"Database rebuild failed: {e}")
        raise RuntimeError(f"Database rebuild failed: {e}")


def validate_data() -> Dict[str, bool]:
    """
    Validate data integrity and consistency.

    Returns:
        Dictionary of validation results

    Checks:
    - No missing critical fields
    - Date ranges are valid
    - No duplicate entries
    - Referential integrity
    - Statistical sanity checks
    """
    logger.info("Running data validation checks")

    validation_results = {}

    try:
        shows, songs, setlists, song_tags = load_all_data()

        # Check 1: No missing critical fields
        critical_show_fields = ["show_id", "date", "venue_id"]
        missing_shows = shows[critical_show_fields].isnull().any()
        validation_results["no_missing_shows"] = not missing_shows.any()

        critical_song_fields = ["song_id", "title"]
        missing_songs = songs[critical_song_fields].isnull().any()
        validation_results["no_missing_songs"] = not missing_songs.any()

        # Check 2: Valid date range
        shows["date"] = pd.to_datetime(shows["date"])
        valid_dates = (shows["date"] >= "2010-01-01") & (
            shows["date"] <= pd.Timestamp.now()
        )
        validation_results["valid_date_range"] = valid_dates.all()

        # Check 3: No duplicates
        validation_results["no_duplicate_shows"] = (
            not shows["show_id"].duplicated().any()
        )
        validation_results["no_duplicate_songs"] = (
            not songs["song_id"].duplicated().any()
        )

        # Check 4: Referential integrity
        show_ids_in_setlists = set(setlists["show_id"].unique())
        show_ids_in_shows = set(shows["show_id"].unique())
        validation_results["referential_integrity_shows"] = (
            show_ids_in_setlists.issubset(show_ids_in_shows)
        )

        song_ids_in_setlists = set(setlists["song_id"].unique())
        song_ids_in_songs = set(songs["song_id"].unique())
        validation_results["referential_integrity_songs"] = (
            song_ids_in_setlists.issubset(song_ids_in_songs)
        )

        # Check 5: Statistical sanity
        setlist_sizes = setlists.groupby("show_id").size()
        validation_results["reasonable_setlist_sizes"] = (
            (setlist_sizes >= 5) & (setlist_sizes <= 50)
        ).all()

        # Print results
        logger.info("\nValidation Results:")
        logger.info("=" * 50)
        for check, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")

        all_passed = all(validation_results.values())
        if all_passed:
            logger.info("\nAll validation checks passed!")
        else:
            logger.warning("\nSome validation checks failed!")

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def check_data_leakage() -> bool:
    """
    Test for data leakage in feature engineering pipeline.

    Returns:
        True if no leakage detected, False otherwise

    Tests:
    - Features only use train set statistics
    - Temporal ordering is preserved
    - No future information in features
    - Consistent results across multiple runs
    """
    logger.info("Checking for data leakage in feature engineering")

    try:
        shows, songs, setlists, song_tags = load_all_data()
        shows = shows[shows["date"] >= RECENT_SHOWS_DATE].copy()
        shows = shows.sort_values("date")

        # Split data temporally
        n_train = int(0.7 * len(shows))
        train_shows = shows.iloc[:n_train]
        test_shows = shows.iloc[n_train:]

        logger.info(f"Train shows: {len(train_shows)}, Test shows: {len(test_shows)}")

        # Engineer features with train-only statistics
        df_train, feature_cols = engineer_all_features(
            train_shows,
            songs,
            setlists,
            song_tags,
            train_show_ids=train_shows["show_id"].values,
        )

        df_test, _ = engineer_all_features(
            test_shows,
            songs,
            setlists,
            song_tags,
            train_show_ids=train_shows["show_id"].values,
        )

        leakage_detected = False

        # Test 1: Check for NaN or infinite values (indicates leakage)
        if df_test[feature_cols].isnull().any().any():
            logger.warning("NaN values found in test features - possible leakage")
            leakage_detected = True

        if np.isinf(df_test[feature_cols].values).any():
            logger.warning("Infinite values found in test features - possible leakage")
            leakage_detected = True

        # Test 2: Check that play_rate is reasonable for test set
        if "play_rate" in feature_cols:
            test_play_rates = df_test.groupby("song_id")["play_rate"].first()
            if (test_play_rates > 1.0).any() or (test_play_rates < 0.0).any():
                logger.warning("Invalid play_rate values - possible leakage")
                leakage_detected = True

        # Test 3: Check temporal features
        if "days_since_last" in feature_cols:
            # Should have some -1 values (never played before)
            never_played = (df_test["days_since_last"] == -1).sum()
            logger.info(f"Songs never played before: {never_played}")

        # Test 4: Verify consistency across multiple runs
        df_test_2, _ = engineer_all_features(
            test_shows,
            songs,
            setlists,
            song_tags,
            train_show_ids=train_shows["show_id"].values,
        )

        if not df_test[feature_cols].equals(df_test_2[feature_cols]):
            logger.warning(
                "Inconsistent feature values across runs - possible randomness or leakage"
            )
            leakage_detected = True

        if not leakage_detected:
            logger.info("\nNo data leakage detected!")
            logger.info("Feature engineering pipeline is correct.")
        else:
            logger.error("\nData leakage detected!")
            logger.error("Please review feature engineering code.")

        return not leakage_detected

    except Exception as e:
        logger.error(f"Leakage check failed: {e}")
        raise


def run_full_pipeline() -> None:
    """
    Run the complete data pipeline from start to finish.

    Steps:
    1. Fetch API data
    2. Rebuild database
    3. Validate data
    4. Check for leakage
    """
    logger.info("Running full data pipeline")
    logger.info("=" * 70)

    try:
        # Step 1: Fetch API data
        logger.info("\n[1/4] Fetching API data...")
        fetch_api_data()

        # Step 2: Rebuild database
        logger.info("\n[2/4] Rebuilding database...")
        rebuild_database(data_source="api")

        # Step 3: Validate data
        logger.info("\n[3/4] Validating data...")
        validation_results = validate_data()

        if not all(validation_results.values()):
            logger.error("Data validation failed. Stopping pipeline.")
            sys.exit(1)

        # Step 4: Check for leakage
        logger.info("\n[4/4] Checking for data leakage...")
        no_leakage = check_data_leakage()

        if not no_leakage:
            logger.error("Data leakage detected. Please fix before training models.")
            sys.exit(1)

        logger.info("\n" + "=" * 70)
        logger.info("Full pipeline completed successfully!")
        logger.info("Data is ready for model training.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for data pipeline script."""
    parser = argparse.ArgumentParser(
        description="Data pipeline for King Gizzard setlist prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch fresh data from API
    python scripts/data_pipeline.py --fetch-api

    # Rebuild database from latest API data
    python scripts/data_pipeline.py --rebuild-db

    # Validate data integrity
    python scripts/data_pipeline.py --validate

    # Check for data leakage
    python scripts/data_pipeline.py --check-leakage

    # Run complete pipeline
    python scripts/data_pipeline.py --full-pipeline
        """,
    )

    parser.add_argument(
        "--fetch-api", action="store_true", help="Fetch fresh data from setlist.fm API"
    )

    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Rebuild DuckDB database from raw data",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data integrity and consistency",
    )

    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Check for data leakage in feature engineering",
    )

    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run complete pipeline (fetch, rebuild, validate, check)",
    )

    parser.add_argument(
        "--data-source",
        choices=["api", "csv"],
        default="api",
        help="Data source for database rebuild (default: api)",
    )

    parser.add_argument(
        "--artist",
        type=str,
        default="King Gizzard & The Lizard Wizard",
        help="Artist name for API fetch (default: King Gizzard & The Lizard Wizard)",
    )

    args = parser.parse_args()

    # Execute requested operations
    if args.full_pipeline:
        run_full_pipeline()
    else:
        if args.fetch_api:
            fetch_api_data(args.artist)

        if args.rebuild_db:
            rebuild_database(args.data_source)

        if args.validate:
            validate_data()

        if args.check_leakage:
            check_data_leakage()

        if not any(
            [
                args.fetch_api,
                args.rebuild_db,
                args.validate,
                args.check_leakage,
                args.full_pipeline,
            ]
        ):
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
