"""
Comprehensive stress tests for venue-country mapping system.

Tests all edge cases to ensure robustness in production.
"""

import json
import tempfile
import pytest
from pathlib import Path
import pandas as pd

from src.utils.venue_mapping import VenueCountryMapper
from src.inference.context_retrieval import ShowContextRetriever, prepare_inference_input


class TestVenueCountryMapper:
    """Test suite for VenueCountryMapper class."""

    def test_initialization_creates_mapping_file(self):
        """Test that mapper creates mapping file if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            # Should initialize with empty mapping
            assert len(mapper) == 0
            assert not mapping_file.exists()  # Not saved yet

    def test_load_existing_mapping(self):
        """Test loading from existing mapping file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'

            # Create a mapping file
            test_mapping = {
                'venue_1': 'US',
                'venue_2': 'AU',
                'venue_3': 'GB'
            }
            with open(mapping_file, 'w') as f:
                json.dump(test_mapping, f)

            # Load it
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            assert len(mapper) == 3
            assert mapper.get_country('venue_1') == 'US'
            assert mapper.get_country('venue_2') == 'AU'
            assert mapper.get_country('venue_3') == 'GB'

    def test_get_country_unknown_venue(self):
        """Test that unknown venues return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            result = mapper.get_country('unknown_venue')
            assert result is None

    def test_add_single_venue(self):
        """Test adding a single venue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            mapper.add_venue('venue_1', 'US', save=True)

            assert mapper.get_country('venue_1') == 'US'
            assert mapping_file.exists()

            # Verify persistence
            mapper2 = VenueCountryMapper(mapping_file=mapping_file)
            assert mapper2.get_country('venue_1') == 'US'

    def test_bulk_add_venues(self):
        """Test adding multiple venues at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            new_venues = {
                'venue_1': 'US',
                'venue_2': 'AU',
                'venue_3': 'GB',
                'venue_4': 'DE'
            }
            mapper.bulk_add_venues(new_venues, save=True)

            assert len(mapper) == 4
            assert mapper.get_country('venue_3') == 'GB'

    def test_update_existing_venue(self):
        """Test that updating a venue's country works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            mapper.add_venue('venue_1', 'US', save=False)
            assert mapper.get_country('venue_1') == 'US'

            # Update country
            mapper.add_venue('venue_1', 'CA', save=False)
            assert mapper.get_country('venue_1') == 'CA'

    def test_contains_operator(self):
        """Test __contains__ operator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            mapper.add_venue('venue_1', 'US', save=False)

            assert 'venue_1' in mapper
            assert 'unknown_venue' not in mapper

    def test_build_from_shows_data(self):
        """Test building mapping from shows DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            # Create sample shows data
            shows_df = pd.DataFrame({
                'venue_id': ['v1', 'v2', 'v3', 'v1', 'v2'],
                'country': ['US', 'AU', 'GB', 'US', 'AU']
            })

            num_venues = mapper.build_from_shows_data(shows_df, save=True)

            assert num_venues == 3
            assert len(mapper) == 3
            assert mapper.get_country('v1') == 'US'
            assert mapper.get_country('v2') == 'AU'
            assert mapper.get_country('v3') == 'GB'

    def test_malformed_json_file(self):
        """Test handling of corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'

            # Create malformed JSON
            with open(mapping_file, 'w') as f:
                f.write('{ this is not valid json }')

            # Should handle gracefully
            mapper = VenueCountryMapper(mapping_file=mapping_file)
            assert len(mapper) == 0  # Empty mapping after load failure

    def test_concurrent_writes(self):
        """Test that save operations don't corrupt the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            # Simulate rapid writes
            for i in range(100):
                mapper.add_venue(f'venue_{i}', 'US', save=True)

            # Verify all were saved
            mapper2 = VenueCountryMapper(mapping_file=mapping_file)
            assert len(mapper2) == 100

    def test_special_characters_in_venue_id(self):
        """Test handling of special characters in venue IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            special_venues = {
                'venue-with-dashes': 'US',
                'venue_with_underscores': 'AU',
                'venue.with.dots': 'GB',
                'venue@with@symbols': 'DE'
            }

            mapper.bulk_add_venues(special_venues, save=True)

            assert mapper.get_country('venue-with-dashes') == 'US'
            assert mapper.get_country('venue.with.dots') == 'GB'


class TestShowContextRetrieverIntegration:
    """Test integration with ShowContextRetriever."""

    def test_retriever_initializes_mapper(self):
        """Test that ShowContextRetriever initializes venue mapper."""
        retriever = ShowContextRetriever(use_cached_data=True)

        assert hasattr(retriever, 'venue_mapper')
        assert isinstance(retriever.venue_mapper, VenueCountryMapper)
        # Should have loaded venues from actual database
        assert len(retriever.venue_mapper) > 0

    def test_prepare_input_auto_derives_country_for_known_venue(self):
        """Test automatic country derivation for known venues."""
        retriever = ShowContextRetriever(use_cached_data=True)

        # Get a known venue from the mapper
        known_venue_id = list(retriever.venue_mapper.mapping.keys())[0]
        expected_country = retriever.venue_mapper.get_country(known_venue_id)

        user_input = {
            'date': '2025-10-25',
            'venue_id': known_venue_id,
            'tour_name': 'Test Tour',
            'is_festival': False,
            'is_marathon': False
        }

        complete_input = prepare_inference_input(
            user_input=user_input,
            retriever=retriever,
            model_type='xgboost'
        )

        assert 'country' in complete_input
        assert complete_input['country'] == expected_country

    def test_prepare_input_uses_provided_country_for_known_venue(self):
        """Test that provided country is used even if venue is known."""
        retriever = ShowContextRetriever(use_cached_data=True)

        known_venue_id = list(retriever.venue_mapper.mapping.keys())[0]

        user_input = {
            'date': '2025-10-25',
            'venue_id': known_venue_id,
            'tour_name': 'Test Tour',
            'country': 'XX',  # Override with custom country
            'is_festival': False,
            'is_marathon': False
        }

        complete_input = prepare_inference_input(
            user_input=user_input,
            retriever=retriever,
            model_type='xgboost'
        )

        # Should respect user-provided country
        assert complete_input['country'] == 'XX'

    def test_prepare_input_raises_error_for_new_venue_without_country(self):
        """Test that new venues without country raise ValueError."""
        retriever = ShowContextRetriever(use_cached_data=True)

        user_input = {
            'date': '2025-10-25',
            'venue_id': 'brand_new_venue_that_does_not_exist',
            'tour_name': 'Test Tour',
            'is_festival': False,
            'is_marathon': False
        }

        with pytest.raises(ValueError) as excinfo:
            prepare_inference_input(
                user_input=user_input,
                retriever=retriever,
                model_type='xgboost'
            )

        assert "Unknown venue" in str(excinfo.value)
        assert "provide 'country' parameter" in str(excinfo.value)

    def test_prepare_input_accepts_new_venue_with_country(self):
        """Test that new venues work if country is provided."""
        retriever = ShowContextRetriever(use_cached_data=True)

        user_input = {
            'date': '2025-10-25',
            'venue_id': 'brand_new_venue_123',
            'tour_name': 'Test Tour',
            'country': 'JP',  # Provided
            'is_festival': False,
            'is_marathon': False
        }

        complete_input = prepare_inference_input(
            user_input=user_input,
            retriever=retriever,
            model_type='xgboost'
        )

        assert complete_input['country'] == 'JP'

    def test_prepare_input_without_venue_id(self):
        """Test behavior when venue_id is not provided."""
        retriever = ShowContextRetriever(use_cached_data=True)

        user_input = {
            'date': '2025-10-25',
            'tour_name': 'Test Tour',
            'country': 'US',  # Must provide country
            'is_festival': False,
            'is_marathon': False
        }

        # Should work but not auto-derive country
        complete_input = prepare_inference_input(
            user_input=user_input,
            retriever=retriever,
            model_type='xgboost'
        )

        assert complete_input['country'] == 'US'


class TestEdgeCases:
    """Stress tests for edge cases."""

    def test_empty_venue_id(self):
        """Test handling of empty venue_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            mapper.add_venue('', 'US', save=False)
            assert mapper.get_country('') == 'US'

    def test_none_country(self):
        """Test that None country values are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            # DataFrame with None values
            shows_df = pd.DataFrame({
                'venue_id': ['v1', 'v2', 'v3'],
                'country': ['US', None, 'GB']
            })

            mapper.build_from_shows_data(shows_df, save=False)

            assert mapper.get_country('v1') == 'US'
            assert mapper.get_country('v2') is None  # Should preserve None
            assert mapper.get_country('v3') == 'GB'

    def test_large_mapping_performance(self):
        """Test performance with large number of venues."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            # Add 10,000 venues
            large_mapping = {f'venue_{i}': 'US' for i in range(10000)}

            start = time.time()
            mapper.bulk_add_venues(large_mapping, save=True)
            save_time = time.time() - start

            # Should save in reasonable time (< 1 second)
            assert save_time < 1.0

            # Test lookup performance
            start = time.time()
            for i in range(1000):
                mapper.get_country(f'venue_{i}')
            lookup_time = time.time() - start

            # 1000 lookups should be fast (< 0.1 second)
            assert lookup_time < 0.1

    def test_unicode_venue_names(self):
        """Test handling of unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = Path(tmpdir) / 'test_mapping.json'
            mapper = VenueCountryMapper(mapping_file=mapping_file)

            unicode_venues = {
                '東京ドーム': 'JP',
                'Théâtre de Paris': 'FR',
                'Москва Арена': 'RU',
                '北京国家体育场': 'CN'
            }

            mapper.bulk_add_venues(unicode_venues, save=True)

            assert mapper.get_country('東京ドーム') == 'JP'
            assert mapper.get_country('Théâtre de Paris') == 'FR'

            # Verify persistence
            mapper2 = VenueCountryMapper(mapping_file=mapping_file)
            assert mapper2.get_country('Москва Арена') == 'RU'


def run_all_tests():
    """Run all tests manually (without pytest runner)."""
    import sys

    print("="*60)
    print("VENUE-COUNTRY MAPPING STRESS TESTS")
    print("="*60)

    test_classes = [
        TestVenueCountryMapper(),
        TestShowContextRetrieverIntegration(),
        TestEdgeCases()
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        for attr_name in dir(test_class):
            if attr_name.startswith('test_'):
                total_tests += 1
                test_method = getattr(test_class, attr_name)

                try:
                    test_method()
                    print(f"  [PASS] {attr_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {attr_name}: {e}")
                    failed_tests += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("="*60)

    return failed_tests == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
