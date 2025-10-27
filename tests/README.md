# Stress Test Suite

This directory contains comprehensive stress tests for the Setlist Predictor codebase.

## Test Files

### 1. test_basic.py
Basic utility function tests for core functions like song title normalization and song ID creation.

**Test Coverage:**
- Song title normalization (segue/transition removal, tease removal, cover removal)
- Song ID creation from titles
- Settings configuration validation

**Run:** `python3 tests/test_basic.py`

### 2. test_dataio.py
Comprehensive tests for data I/O operations including loading/saving parquet and JSON files.

**Test Coverage:**
- Loading all curated data (shows, songs, setlists, song_tags)
- Marathon show identification logic
- Residency show identification logic
- Data integrity and referential constraints
- Parquet save/load roundtrip
- JSON save/load roundtrip
- Date parsing validation
- Setlist position ordering
- DataPaths creation and validation
- Large dataset memory efficiency

**Run:** `PYTHONPATH=. python3 tests/test_dataio.py`

**Results:** 10/10 tests passing

### 3. test_venue_mapping.py
Extensive tests for venue-to-country mapping system with 362 known venues.

**Test Coverage:**
- VenueCountryMapper initialization
- Country lookups for known venues
- Graceful handling of unknown venues
- Adding new venue mappings
- Bulk mapping updates
- JSON persistence (save/load)
- Mapping accuracy validation
- Unicode venue name handling
- Performance stress test (10K venues)
- Edge case handling

**Run:** `PYTHONPATH=. python3 tests/test_venue_mapping.py`

**Results:** 10/10 tests passing (from previous validation)

### 4. test_inference.py
Tests for inference and context retrieval functionality.

**Test Coverage:**
- ShowContextRetriever initialization
- Previous setlist retrieval
- Venue-to-country mapping integration
- Inference input preparation (with/without country)
- New venue handling (with error checking)
- Optional is_marathon flag handling
- Previous setlist chronological ordering
- Edge case: first show (no previous setlists)
- Performance stress test for venue lookups

**Run:** `PYTHONPATH=. python3 tests/test_inference.py`

### 5. test_features.py
Tests for feature engineering operations.

**Test Coverage:**
- Basic feature engineering functionality
- Feature data types (numeric, binary labels)
- Data leakage detection
- Feature completeness (temporal, frequency, venue features)
- Missing value detection
- Feature scale reasonableness (< 1e10)
- Label distribution validation
- Song coverage across features
- Feature engineering determinism
- Performance stress test (< 60s for full dataset)

**Run:** `PYTHONPATH=. python3 tests/test_features.py`

### 6. test_models.py
Tests for model architectures (Note: needs updating to use actual model files).

**Intended Coverage:**
- Model initialization
- Forward pass validation
- Frequency/recency prior computation
- Gradient flow verification
- Dropout behavior (train vs eval mode)
- Device compatibility (CPU/GPU)
- Batch size flexibility
- Model determinism with fixed seeds

**Status:** Needs updating to use temporal_sets_with_priors.py instead of dnntsp.py

## Running All Tests

### Individual Test Files
```bash
# From project root
cd "/Users/junichikoganemaru/Desktop/Setlist predictor"

# Run individual test suites
PYTHONPATH=. python3 tests/test_basic.py
PYTHONPATH=. python3 tests/test_dataio.py
PYTHONPATH=. python3 tests/test_venue_mapping.py
PYTHONPATH=. python3 tests/test_inference.py
PYTHONPATH=. python3 tests/test_features.py
```

### Master Test Runner
```bash
# Run all tests at once (requires all dependencies)
PYTHONPATH=. python3 tests/run_all_tests.py
```

## Test Results Summary

| Test Suite | Tests | Status | Notes |
|------------|-------|--------|-------|
| test_basic.py | 3 | [PASS] | All basic utilities working |
| test_dataio.py | 10 | [PASS] | All data I/O operations validated |
| test_venue_mapping.py | 10 | [PASS] | 100% accuracy on 362 venues |
| test_features.py | 10 | [PASS] | Rewritten to work with current API |
| test_inference.py | 11 | [PASS] | Rewritten to work with current API |
| test_models.py | 10 | Not Running | References outdated model architecture |

## Dependencies

The test suite requires the following Python packages:
- pandas
- numpy
- torch
- pytest (optional)
- All project dependencies from pyproject.toml

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of creating a test class with methods
2. Include both positive and negative test cases
3. Add performance/stress tests for critical paths
4. Use descriptive test names that explain what's being tested
5. Print [PASS]/[FAIL] messages for clarity
6. Include a `run_all_tests()` function at the bottom
7. Update this README with test coverage information

## Design Principles

The test suite follows these principles:

1. **Comprehensive Coverage**: Test all major components and edge cases
2. **Performance Validation**: Ensure operations complete in reasonable time
3. **Data Integrity**: Verify referential constraints and data quality
4. **Determinism**: Ensure reproducible results
5. **Graceful Failure**: Test error handling and edge cases
6. **Production Readiness**: Validate the system works at scale

## Maintenance

Tests should be run:
- After any code changes to core modules
- Before committing to version control
- Before deploying to production
- As part of CI/CD pipeline (when implemented)

Last updated: 2025-10-25
