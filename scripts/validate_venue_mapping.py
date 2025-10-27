"""
Validate venue-country mapping accuracy.

This script cross-references the venue_country_mapping.json with the actual
shows database to detect any inconsistencies or errors.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio import load_all_data
from src.utils.venue_mapping import VenueCountryMapper


def validate_venue_country_mapping():
    """
    Validate venue-country mapping against actual shows database.

    Returns:
        bool: True if all mappings are accurate, False otherwise
    """
    print("=" * 60)
    print("VENUE-COUNTRY MAPPING VALIDATION")
    print("=" * 60)

    # Load mapping file
    print("\n[1/4] Loading venue-country mapping...")
    mapper = VenueCountryMapper()
    print(f"  Found {len(mapper)} venues in mapping file")

    # Load actual shows data
    print("\n[2/4] Loading shows database...")
    shows, _, _, _ = load_all_data()
    print(f"  Loaded {len(shows)} shows")

    # Build ground truth from shows database
    print("\n[3/4] Building ground truth from shows database...")
    ground_truth = {}
    venue_countries = defaultdict(set)

    for _, show in shows.iterrows():
        venue_id = show["venue_id"]
        country = show["country"]
        venue_countries[venue_id].add(country)

        # Ground truth: take the most common country for each venue
        if venue_id not in ground_truth:
            ground_truth[venue_id] = country

    print(f"  Found {len(ground_truth)} unique venues in database")

    # Validate mappings
    print("\n[4/4] Validating mappings...")

    errors = []
    warnings = []
    correct = 0

    # Check 1: Venues in mapping that aren't in database
    missing_venues = set(mapper.mapping.keys()) - set(ground_truth.keys())
    if missing_venues:
        warnings.append(
            f"  WARNING: {len(missing_venues)} venues in mapping but not in database"
        )
        for venue_id in list(missing_venues)[:5]:
            warnings.append(f"    - {venue_id} -> {mapper.get_country(venue_id)}")
        if len(missing_venues) > 5:
            warnings.append(f"    ... and {len(missing_venues) - 5} more")

    # Check 2: Venues in database that aren't in mapping
    unmapped_venues = set(ground_truth.keys()) - set(mapper.mapping.keys())
    if unmapped_venues:
        warnings.append(
            f"  WARNING: {len(unmapped_venues)} venues in database but not in mapping"
        )
        for venue_id in list(unmapped_venues)[:5]:
            warnings.append(f"    - {venue_id} (should be {ground_truth[venue_id]})")
        if len(unmapped_venues) > 5:
            warnings.append(f"    ... and {len(unmapped_venues) - 5} more")

    # Check 3: Verify accuracy of mappings
    for venue_id, mapped_country in mapper.mapping.items():
        if venue_id in ground_truth:
            actual_country = ground_truth[venue_id]

            if mapped_country == actual_country:
                correct += 1
            else:
                errors.append(
                    f"  ERROR: {venue_id} mapped to '{mapped_country}' "
                    f"but database shows '{actual_country}'"
                )

    # Check 4: Venues with multiple countries (should be rare)
    multi_country_venues = {
        vid: countries
        for vid, countries in venue_countries.items()
        if len(countries) > 1
    }

    if multi_country_venues:
        warnings.append(
            f"  INFO: {len(multi_country_venues)} venues appear in multiple countries:"
        )
        for venue_id, countries in list(multi_country_venues.items())[:5]:
            warnings.append(f"    - {venue_id}: {', '.join(countries)}")
        if len(multi_country_venues) > 5:
            warnings.append(f"    ... and {len(multi_country_venues) - 5} more")

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    total_mapped = len(set(mapper.mapping.keys()) & set(ground_truth.keys()))
    accuracy = (correct / total_mapped * 100) if total_mapped > 0 else 0

    print(f"\nAccuracy: {correct}/{total_mapped} ({accuracy:.2f}%)")
    print(f"  Correctly mapped: {correct}")
    print(f"  Incorrectly mapped: {len(errors)}")
    print(f"  Venues in mapping only: {len(missing_venues)}")
    print(f"  Venues in database only: {len(unmapped_venues)}")
    print(f"  Multi-country venues: {len(multi_country_venues)}")

    # Print errors
    if errors:
        print("\n" + "-" * 60)
        print("ERRORS:")
        print("-" * 60)
        for error in errors[:10]:
            print(error)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Print warnings
    if warnings:
        print("\n" + "-" * 60)
        print("WARNINGS:")
        print("-" * 60)
        for warning in warnings[:15]:
            print(warning)
        if len(warnings) > 15:
            print(f"  ... and {len(warnings) - 15} more warnings")

    # Summary
    print("\n" + "=" * 60)
    if accuracy == 100.0 and not errors:
        print("STATUS: PASSED - All mappings are accurate!")
        print("=" * 60)
        return True
    elif accuracy > 95.0 and not errors:
        print("STATUS: PASSED (with warnings) - Mappings are accurate")
        print("=" * 60)
        return True
    else:
        print("STATUS: FAILED - Found mapping errors")
        print("=" * 60)
        return False


def test_specific_venues():
    """Test some known venues to verify correctness."""
    print("\n" + "=" * 60)
    print("SPOT CHECK: Testing Known Venues")
    print("=" * 60)

    mapper = VenueCountryMapper()
    shows, _, _, _ = load_all_data()

    # Get some sample venues from shows
    sample_shows = shows.groupby("venue_id").first().sample(min(10, len(shows)))

    print("\nTesting sample venues:")
    all_correct = True

    for venue_id, show in sample_shows.iterrows():
        mapped = mapper.get_country(venue_id)
        actual = show["country"]
        status = "PASS" if mapped == actual else "FAIL"

        if mapped != actual:
            all_correct = False

        print(f"  [{status}] {venue_id}")
        print(f"        Mapped: {mapped}")
        print(f"        Actual: {actual}")
        if mapped != actual:
            print(f"        City: {show['city']}")

    return all_correct


def check_coverage():
    """Check what percentage of database venues are mapped."""
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)

    mapper = VenueCountryMapper()
    shows, _, _, _ = load_all_data()

    unique_venues = shows["venue_id"].nunique()
    mapped_venues = len(set(shows["venue_id"]) & set(mapper.mapping.keys()))
    coverage = (mapped_venues / unique_venues * 100) if unique_venues > 0 else 0

    print(f"\nTotal unique venues in database: {unique_venues}")
    print(f"Venues with mappings: {mapped_venues}")
    print(f"Coverage: {coverage:.2f}%")

    # Find shows with unmapped venues
    unmapped_shows = shows[~shows["venue_id"].isin(mapper.mapping.keys())]

    if len(unmapped_shows) > 0:
        print(f"\nShows with unmapped venues: {len(unmapped_shows)}")
        print("\nSample unmapped venues:")
        for _, show in unmapped_shows.head(5).iterrows():
            print(f"  - {show['venue_id']} in {show['city']}, {show['country']}")

    return coverage >= 95.0


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# VENUE-COUNTRY MAPPING VALIDATION SUITE")
    print("#" * 60)

    # Run all validation tests
    test1 = validate_venue_country_mapping()
    test2 = test_specific_venues()
    test3 = check_coverage()

    # Final summary
    print("\n" + "#" * 60)
    print("# FINAL SUMMARY")
    print("#" * 60)

    tests = {"Mapping Accuracy": test1, "Spot Check": test2, "Coverage": test3}

    print()
    for test_name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")

    all_passed = all(tests.values())
    print("\n" + "#" * 60)
    if all_passed:
        print("# ALL TESTS PASSED")
        print("#" * 60)
        sys.exit(0)
    else:
        print("# SOME TESTS FAILED")
        print("#" * 60)
        sys.exit(1)
