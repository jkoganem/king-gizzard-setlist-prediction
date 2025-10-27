"""Stress tests for inference and context retrieval."""

import pytest
import pandas as pd
from pathlib import Path

from src.inference.context_retrieval import ShowContextRetriever, prepare_inference_input
from src.dataio import load_all_data


class TestInference:
    """Test suite for inference operations."""

    def __init__(self):
        """Initialize with test data."""
        self.retriever = ShowContextRetriever(use_cached_data=True)
        self.shows, self.songs, self.setlists, _ = load_all_data()
        # Ensure dates are datetime objects
        self.shows["date"] = pd.to_datetime(self.shows["date"])

    def test_context_retriever_initialization(self):
        """Test that ShowContextRetriever initializes correctly."""
        assert self.retriever is not None
        assert hasattr(self.retriever, 'shows')
        assert hasattr(self.retriever, 'setlists')
        assert hasattr(self.retriever, 'venue_mapper')

        print(f"[PASS] ShowContextRetriever initialized with {len(self.retriever.shows)} shows")

    def test_get_previous_shows(self):
        """Test retrieving previous shows for a given date."""
        # Get a random show from the middle of the dataset
        sorted_shows = self.shows.sort_values('date')
        test_show = sorted_shows.iloc[len(sorted_shows) // 2]
        test_date = test_show['date'].strftime("%Y-%m-%d")

        # Get previous 5 shows
        previous_shows = self.retriever.get_previous_shows(test_date, n_shows=5)

        # Verify we got a list
        assert isinstance(previous_shows, list)

        # Verify length is at most 5
        assert len(previous_shows) <= 5

        # Verify all previous shows are before the test date
        for show in previous_shows:
            assert isinstance(show, dict), "Each show should be a dict"
            assert 'date' in show, "Show should have date field"
            assert 'setlist' in show, "Show should have setlist field"
            assert isinstance(show['setlist'], list), "Setlist should be a list of song IDs"
            show_date = pd.to_datetime(show['date'])
            assert show_date < pd.to_datetime(test_date), f"Show {show['date']} should be before {test_date}"

        print(f"[PASS] Retrieved {len(previous_shows)} previous shows for date {test_date}")

    def test_venue_country_mapping(self):
        """Test venue to country mapping."""
        # Get a venue from the data
        test_venue = self.shows.iloc[0]['venue_id']
        test_country = self.shows.iloc[0]['country']

        # Get country from mapper
        mapped_country = self.retriever.venue_mapper.get_country(test_venue)

        # Verify mapping is correct
        assert mapped_country == test_country, f"Venue {test_venue} should map to {test_country}, got {mapped_country}"

        print(f"[PASS] Venue {test_venue} correctly maps to {test_country}")

    def test_prepare_inference_input_with_country(self):
        """Test preparing inference input when country is provided."""
        # Get a real show from data
        test_show = self.shows.iloc[10]

        user_input = {
            'date': test_show['date'].strftime("%Y-%m-%d"),
            'venue_id': test_show['venue_id'],
            'tour_name': 'Test Tour',
            'country': test_show['country'],
            'is_festival': False,
            'is_marathon': False
        }

        # Prepare input
        complete_input = prepare_inference_input(user_input, self.retriever)

        # Verify required fields are present
        assert 'date' in complete_input
        assert 'venue_id' in complete_input
        assert 'country' in complete_input
        assert 'previous_shows' in complete_input

        # Verify country matches
        assert complete_input['country'] == test_show['country']

        print(f"[PASS] Prepared inference input with explicit country: {complete_input['country']}")

    def test_prepare_inference_input_without_country(self):
        """Test preparing inference input when country is auto-derived."""
        # Get a real show from data
        test_show = self.shows.iloc[15]

        user_input = {
            'date': test_show['date'].strftime("%Y-%m-%d"),
            'venue_id': test_show['venue_id'],
            'tour_name': 'Test Tour',
            'is_festival': False,
            'is_marathon': False
        }

        # Prepare input (should auto-derive country)
        complete_input = prepare_inference_input(user_input, self.retriever)

        # Verify country was auto-derived correctly
        assert 'country' in complete_input
        assert complete_input['country'] == test_show['country']

        print(f"[PASS] Auto-derived country for venue {test_show['venue_id']}: {complete_input['country']}")

    def test_prepare_inference_input_new_venue(self):
        """Test preparing inference input for a new venue."""
        user_input = {
            'date': '2025-12-01',
            'venue_id': 'new_venue_never_seen_before_12345',
            'tour_name': 'Test Tour',
            'is_festival': False,
            'is_marathon': False
        }

        # This should raise an error because venue is new and no country provided
        try:
            complete_input = prepare_inference_input(user_input, self.retriever)
            assert False, "Should have raised ValueError for unknown venue without country"
        except ValueError as e:
            assert 'Unknown venue' in str(e) or 'provide' in str(e).lower()
            print(f"[PASS] Correctly raised error for unknown venue: {str(e)[:80]}...")

    def test_prepare_inference_input_new_venue_with_country(self):
        """Test preparing inference input for a new venue with country provided."""
        user_input = {
            'date': '2025-12-01',
            'venue_id': 'new_venue_with_country_99999',
            'tour_name': 'Test Tour',
            'country': 'US',
            'is_festival': False,
            'is_marathon': False
        }

        # This should work because country is provided
        complete_input = prepare_inference_input(user_input, self.retriever)

        # Verify country was set
        assert complete_input['country'] == 'US'

        print("[PASS] Prepared inference input for new venue with explicit country")

    def test_optional_marathon_flag(self):
        """Test that is_marathon flag is handled correctly."""
        test_show = self.shows.iloc[0]

        # Test with marathon=True
        user_input_marathon = {
            'date': test_show['date'].strftime("%Y-%m-%d"),
            'venue_id': test_show['venue_id'],
            'tour_name': 'Test Tour',
            'is_marathon': True,
            'is_festival': False
        }

        complete_input = prepare_inference_input(user_input_marathon, self.retriever)
        assert 'is_marathon' in complete_input
        assert complete_input['is_marathon'] is True

        # Test with marathon=False
        user_input_normal = {
            'date': test_show['date'].strftime("%Y-%m-%d"),
            'venue_id': test_show['venue_id'],
            'tour_name': 'Test Tour',
            'is_marathon': False,
            'is_festival': False
        }

        complete_input = prepare_inference_input(user_input_normal, self.retriever)
        assert 'is_marathon' in complete_input
        assert complete_input['is_marathon'] is False

        print("[PASS] is_marathon flag handled correctly")

    def test_previous_shows_ordering(self):
        """Test that previous shows are in chronological order."""
        # Get a date in the middle of the dataset
        sorted_shows = self.shows.sort_values('date')
        test_show = sorted_shows.iloc[100]
        test_date = test_show['date'].strftime("%Y-%m-%d")

        # Get previous shows
        previous_shows = self.retriever.get_previous_shows(test_date, n_shows=5)

        if len(previous_shows) > 1:
            # Verify dates are in ascending order (oldest first)
            dates = [pd.to_datetime(show['date']) for show in previous_shows]
            for i in range(len(dates) - 1):
                assert dates[i] <= dates[i + 1], "Previous shows should be in chronological order"

            print(f"[PASS] Previous shows are correctly ordered chronologically")
        else:
            print("[SKIP] Not enough previous shows to test ordering")

    def test_edge_case_first_show(self):
        """Test handling of the very first show (no previous shows)."""
        # Get the first show
        first_show = self.shows.sort_values('date').iloc[0]
        first_date = first_show['date'].strftime("%Y-%m-%d")

        # Get previous shows (should be empty)
        previous_shows = self.retriever.get_previous_shows(first_date, n_shows=5)

        # Should handle gracefully
        assert isinstance(previous_shows, list)
        assert len(previous_shows) == 0

        print(f"[PASS] Handled first show correctly (no previous shows)")

    def test_stress_many_venues(self):
        """Test performance with many different venues."""
        import time

        unique_venues = self.shows['venue_id'].unique()[:50]  # Test first 50 venues

        start_time = time.time()
        for venue in unique_venues:
            country = self.retriever.venue_mapper.get_country(venue)
            assert country is not None, f"Venue {venue} should have a country mapping"

        elapsed_time = time.time() - start_time

        # Should be very fast (< 0.1 seconds for 50 lookups)
        assert elapsed_time < 0.1, f"Venue lookups too slow: {elapsed_time:.3f}s for {len(unique_venues)} venues"

        print(f"[PASS] Looked up {len(unique_venues)} venues in {elapsed_time:.4f}s "
              f"({elapsed_time/len(unique_venues)*1000:.2f}ms per venue)")


def run_all_tests():
    """Run all tests in this module."""
    test_class = TestInference()

    tests = [
        test_class.test_context_retriever_initialization,
        test_class.test_get_previous_shows,
        test_class.test_venue_country_mapping,
        test_class.test_prepare_inference_input_with_country,
        test_class.test_prepare_inference_input_without_country,
        test_class.test_prepare_inference_input_new_venue,
        test_class.test_prepare_inference_input_new_venue_with_country,
        test_class.test_optional_marathon_flag,
        test_class.test_previous_shows_ordering,
        test_class.test_edge_case_first_show,
        test_class.test_stress_many_venues,
    ]

    passed = 0
    failed = 0

    print("="*60)
    print("Running Inference & Context Retrieval Stress Tests")
    print("="*60)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
