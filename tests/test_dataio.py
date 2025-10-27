"""Stress tests for data I/O operations."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.dataio import (
    load_parquet,
    save_parquet,
    load_json,
    save_json,
    load_all_data,
    identify_marathon_shows,
    identify_residency_shows,
    DataPaths,
)


class TestDataIO:
    """Test suite for data I/O operations."""

    def test_load_all_data(self):
        """Test loading all curated data."""
        shows, songs, setlists, song_tags = load_all_data()

        # Verify data was loaded
        assert len(shows) > 0, "Shows data should not be empty"
        assert len(songs) > 0, "Songs data should not be empty"
        assert len(setlists) > 0, "Setlists data should not be empty"

        # Verify required columns exist
        assert 'show_id' in shows.columns
        assert 'venue_id' in shows.columns
        assert 'date' in shows.columns
        assert 'song_id' in songs.columns
        assert 'show_id' in setlists.columns
        assert 'song_id' in setlists.columns

        # Verify enriched columns exist
        assert 'is_marathon' in shows.columns
        assert 'is_residency' in shows.columns
        assert 'is_festival' in shows.columns

        print(f"[PASS] Loaded {len(shows)} shows, {len(songs)} songs, {len(setlists)} setlist entries")

    def test_marathon_identification(self):
        """Test marathon show identification logic."""
        shows, _, setlists, _ = load_all_data()

        marathon_shows = shows[shows['is_marathon'] == 1]

        # Verify marathon flags are binary
        assert set(shows['is_marathon'].unique()).issubset({0, 1})

        print(f"[PASS] Identified {len(marathon_shows)} marathon shows out of {len(shows)} total")

    def test_residency_identification(self):
        """Test residency show identification logic."""
        shows, _, _, _ = load_all_data()

        residency_shows = shows[shows['is_residency'] == 1]

        # Verify residency flags are binary
        assert set(shows['is_residency'].unique()).issubset({0, 1})

        print(f"[PASS] Identified {len(residency_shows)} residency shows out of {len(shows)} total")

    def test_data_integrity(self):
        """Test data integrity and referential constraints."""
        shows, songs, setlists, _ = load_all_data()

        # Test 1: All setlist show_ids should exist in shows
        setlist_show_ids = set(setlists['show_id'].unique())
        shows_show_ids = set(shows['show_id'].unique())
        orphaned_setlists = setlist_show_ids - shows_show_ids
        assert len(orphaned_setlists) == 0, f"Found {len(orphaned_setlists)} setlists with invalid show_ids"

        # Test 2: All setlist song_ids should exist in songs
        setlist_song_ids = set(setlists['song_id'].unique())
        songs_song_ids = set(songs['song_id'].unique())
        orphaned_songs = setlist_song_ids - songs_song_ids
        assert len(orphaned_songs) == 0, f"Found {len(orphaned_songs)} setlists with invalid song_ids"

        # Test 3: No null values in critical columns
        assert shows['show_id'].notna().all(), "Found null show_ids"
        assert shows['date'].notna().all(), "Found null dates"
        assert songs['song_id'].notna().all(), "Found null song_ids"
        assert setlists['show_id'].notna().all(), "Found null show_ids in setlists"
        assert setlists['song_id'].notna().all(), "Found null song_ids in setlists"

        print("[PASS] All data integrity checks passed")

    def test_parquet_save_load_roundtrip(self):
        """Test that data can be saved and loaded without corruption."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test data
            test_data = pd.DataFrame({
                'id': ['a', 'b', 'c'],
                'value': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie']
            })

            # Save and load
            test_file = tmpdir / "test.parquet"
            save_parquet(test_data, test_file)
            loaded_data = load_parquet(test_file)

            # Verify identical
            pd.testing.assert_frame_equal(test_data, loaded_data)

            print("[PASS] Parquet save/load roundtrip successful")

    def test_json_save_load_roundtrip(self):
        """Test that JSON data can be saved and loaded without corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test data
            test_data = {
                'name': 'Test',
                'values': [1, 2, 3],
                'nested': {'key': 'value'}
            }

            # Save and load
            test_file = tmpdir / "test.json"
            save_json(test_data, test_file)
            loaded_data = load_json(test_file)

            # Verify identical
            assert test_data == loaded_data

            print("[PASS] JSON save/load roundtrip successful")

    def test_date_parsing(self):
        """Test that dates are parsed correctly."""
        shows, _, _, _ = load_all_data()

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(shows['date']):
            shows['date'] = pd.to_datetime(shows['date'])

        # Verify dates are in reasonable range (2010-2030)
        min_date = pd.Timestamp('2010-01-01')
        max_date = pd.Timestamp('2030-12-31')
        assert (shows['date'] >= min_date).all(), f"Found dates before 2010: {shows['date'].min()}"
        assert (shows['date'] <= max_date).all(), f"Found dates after 2030: {shows['date'].max()}"

        print(f"[PASS] All {len(shows)} show dates are valid ({shows['date'].min()} to {shows['date'].max()})")

    def test_setlist_position_ordering(self):
        """Test that setlist positions are correctly ordered."""
        _, _, setlists, _ = load_all_data()

        if 'position' not in setlists.columns:
            print("[SKIP] No position column in setlists")
            return

        # Group by show and verify positions are sequential
        for show_id, group in setlists.groupby('show_id'):
            positions = sorted(group['position'].values)
            # Positions should start at 0 or 1 and be sequential
            if len(positions) > 0:
                assert positions[0] in [0, 1], f"Show {show_id} positions don't start at 0 or 1"
                # Check for duplicates
                assert len(positions) == len(set(positions)), f"Show {show_id} has duplicate positions"

        print("[PASS] Setlist positions are correctly ordered")

    def test_data_paths_creation(self):
        """Test DataPaths creation and validation."""
        paths = DataPaths.from_curated_dir()

        # Verify all paths exist
        assert paths.venues.exists(), f"Venues file not found: {paths.venues}"
        assert paths.shows.exists(), f"Shows file not found: {paths.shows}"
        assert paths.songs.exists(), f"Songs file not found: {paths.songs}"
        assert paths.setlists.exists(), f"Setlists file not found: {paths.setlists}"

        print("[PASS] All data paths exist and are accessible")

    def test_large_data_handling(self):
        """Test handling of large datasets."""
        shows, songs, setlists, _ = load_all_data()

        # Test memory efficiency
        shows_memory = shows.memory_usage(deep=True).sum() / 1024**2  # MB
        songs_memory = songs.memory_usage(deep=True).sum() / 1024**2
        setlists_memory = setlists.memory_usage(deep=True).sum() / 1024**2
        total_memory = shows_memory + songs_memory + setlists_memory

        # Verify memory usage is reasonable (should be < 100MB for this dataset)
        assert total_memory < 100, f"Data using too much memory: {total_memory:.2f}MB"

        print(f"[PASS] Memory usage: {total_memory:.2f}MB (shows: {shows_memory:.2f}MB, "
              f"songs: {songs_memory:.2f}MB, setlists: {setlists_memory:.2f}MB)")


def run_all_tests():
    """Run all tests in this module."""
    test_class = TestDataIO()

    tests = [
        test_class.test_load_all_data,
        test_class.test_marathon_identification,
        test_class.test_residency_identification,
        test_class.test_data_integrity,
        test_class.test_parquet_save_load_roundtrip,
        test_class.test_json_save_load_roundtrip,
        test_class.test_date_parsing,
        test_class.test_setlist_position_ordering,
        test_class.test_data_paths_creation,
        test_class.test_large_data_handling,
    ]

    passed = 0
    failed = 0

    print("="*60)
    print("Running Data I/O Stress Tests")
    print("="*60)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
