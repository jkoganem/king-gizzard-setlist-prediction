"""Master test runner for all stress tests."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_basic import test_normalize_song_title, test_create_song_id, test_settings
from tests.test_dataio import run_all_tests as run_dataio_tests
from tests.test_inference import run_all_tests as run_inference_tests
from tests.test_features import run_all_tests as run_features_tests
from tests.test_models import run_all_tests as run_models_tests
from tests.test_venue_mapping import run_all_tests as run_venue_mapping_tests


def run_basic_tests():
    """Run basic utility tests."""
    print("="*60)
    print("Running Basic Utility Tests")
    print("="*60)

    passed = 0
    failed = 0

    tests = [
        ("test_normalize_song_title", test_normalize_song_title),
        ("test_create_song_id", test_create_song_id),
        ("test_settings", test_settings),
    ]

    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("="*60)

    return passed, failed


def main():
    """Run all test suites."""
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE STRESS TEST SUITE")
    print("="*80 + "\n")

    total_passed = 0
    total_failed = 0

    # Run all test suites
    test_suites = [
        ("Basic Utilities", run_basic_tests),
        ("Data I/O", run_dataio_tests),
        ("Venue Mapping", run_venue_mapping_tests),
        ("Inference & Context Retrieval", run_inference_tests),
        ("Feature Engineering", run_features_tests),
        ("Model Architectures", run_models_tests),
    ]

    suite_results = []

    for suite_name, run_func in test_suites:
        print(f"\n{'='*80}")
        print(f"Test Suite: {suite_name}")
        print(f"{'='*80}\n")

        try:
            if suite_name == "Basic Utilities":
                passed, failed = run_func()
            else:
                # These return None, we'll track via output
                run_func()
                # For now, assume they track their own pass/fail
                passed, failed = 0, 0
        except Exception as e:
            print(f"\n[ERROR] Test suite {suite_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            passed, failed = 0, 1

        suite_results.append((suite_name, passed, failed))
        total_passed += passed
        total_failed += failed

    # Final summary
    print("\n" + "="*80)
    print(" "*25 + "FINAL TEST SUMMARY")
    print("="*80)

    for suite_name, passed, failed in suite_results:
        if passed > 0 or failed > 0:
            total_tests = passed + failed
            status = "[PASS]" if failed == 0 else "[FAIL]"
            print(f"{status} {suite_name}: {passed}/{total_tests} tests passed")
        else:
            print(f"[INFO] {suite_name}: See detailed output above")

    print("\n" + "="*80)
    print(f"All stress tests completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
