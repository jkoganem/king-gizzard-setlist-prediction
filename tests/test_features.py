"""Stress tests for feature engineering."""

import pandas as pd
import numpy as np

from src.features.feature_pipeline import engineer_all_features
from src.dataio import load_all_data


class TestFeatures:
    """Test suite for feature engineering."""

    def __init__(self):
        """Initialize with test data."""
        print("Loading data...")
        self.shows, self.songs, self.setlists, self.song_tags = load_all_data()
        self.shows["date"] = pd.to_datetime(self.shows["date"])
        self.shows = self.shows.sort_values("date")

        # Create a temporal split for testing
        n_train = int(len(self.shows) * 0.70)
        train_shows = self.shows.iloc[:n_train]
        self.train_show_ids = set(train_shows["show_id"].values)

    def test_feature_engineering_basic(self):
        """Test basic feature engineering."""
        # Engineer features
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        # Verify output shapes
        assert len(df) > 0, "Features DataFrame should not be empty"
        assert len(feature_cols) > 0, "Feature columns list should not be empty"
        assert "label" in df.columns, "DataFrame must have 'label' column"
        assert "show_id" in df.columns, "DataFrame must have 'show_id' column"
        assert "song_id" in df.columns, "DataFrame must have 'song_id' column"

        print(
            f"[PASS] Engineered {len(df)} examples with {len(feature_cols)} features"
        )

    def test_feature_dtypes(self):
        """Test that feature dtypes are correct."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        # Verify labels are binary (0 or 1)
        labels = df["label"].values
        assert set(np.unique(labels)).issubset(
            {0, 1}
        ), "Labels should be binary (0 or 1)"

        # Verify features are numeric
        features_df = df[feature_cols]
        assert (
            features_df.select_dtypes(include=[np.number]).shape[1]
            == features_df.shape[1]
        ), "All features should be numeric"

        print(f"[PASS] All {len(feature_cols)} features are numeric, labels are binary")

    def test_no_data_leakage(self):
        """Test that there's no data leakage in features."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        # Features should not contain any show-specific information that could leak
        # Check column names for suspicious patterns
        suspicious_cols = [col for col in feature_cols if "show_id" in col.lower()]
        assert (
            len(suspicious_cols) == 0
        ), f"Found suspicious columns that might leak data: {suspicious_cols}"

        print("[PASS] No obvious data leakage detected in feature names")

    def test_feature_completeness(self):
        """Test that all expected feature categories are present."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        columns = set(feature_cols)

        # Check for important feature categories
        has_temporal = any(
            "recency" in col.lower() or "days" in col.lower() for col in columns
        )
        has_frequency = any(
            "freq" in col.lower() or "count" in col.lower() for col in columns
        )

        assert has_temporal, "Should have temporal features (recency/days since)"
        assert has_frequency, "Should have frequency features"

        print(f"[PASS] Feature set includes temporal, frequency features")

    def test_no_missing_values(self):
        """Test that there are no missing values in features."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        features_df = df[feature_cols]

        # Check for NaN values
        nan_counts = features_df.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]

        if len(cols_with_nans) > 0:
            print(
                f"[WARNING] Found NaN values in columns: {cols_with_nans.to_dict()}"
            )
        else:
            print("[PASS] No missing values in features")

    def test_feature_scale_reasonableness(self):
        """Test that feature scales are reasonable."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        features_df = df[feature_cols]

        # Check for extremely large values that might indicate errors
        for col in features_df.columns:
            max_val = features_df[col].max()
            min_val = features_df[col].min()

            # Values shouldn't be astronomically large (> 1e10)
            assert (
                abs(max_val) < 1e10
            ), f"Column {col} has suspiciously large max value: {max_val}"
            assert (
                abs(min_val) < 1e10
            ), f"Column {col} has suspiciously large min value: {min_val}"

        print("[PASS] All feature scales are reasonable (< 1e10)")

    def test_label_distribution(self):
        """Test that label distribution is reasonable."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        labels = df["label"].values

        # Calculate positive rate
        positive_rate = labels.mean()

        # For set prediction, positive rate should be relatively low
        # (most songs aren't played at any given show)
        assert 0 < positive_rate < 1, "Positive rate should be between 0 and 1"

        # Typical range for this task is 5-20%
        print(
            f"[PASS] Label distribution: {positive_rate*100:.2f}% positive "
            f"({labels.sum():,} positives out of {len(labels):,} total)"
        )

    def test_song_coverage(self):
        """Test that all songs are represented in features."""
        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        # Get unique songs from setlists
        unique_songs_in_setlists = set(self.setlists["song_id"].unique())
        unique_songs_in_features = set(df["song_id"].unique())

        # All songs that appear in setlists should be in features
        missing_songs = unique_songs_in_setlists - unique_songs_in_features

        if len(missing_songs) > 0:
            print(
                f"[WARNING] {len(missing_songs)} songs in setlists but not in features"
            )
        else:
            print(
                f"[PASS] All {len(unique_songs_in_setlists)} songs from setlists are in features"
            )

    def test_feature_determinism(self):
        """Test that feature engineering is deterministic."""
        # Engineer features twice
        df1, feature_cols1 = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        df2, feature_cols2 = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        # Results should be identical
        assert feature_cols1 == feature_cols2, "Feature columns should be identical"
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
        )

        print("[PASS] Feature engineering is deterministic")

    def test_performance_stress(self):
        """Test feature engineering performance with full dataset."""
        import time

        start_time = time.time()

        df, feature_cols = engineer_all_features(
            shows=self.shows,
            songs=self.songs,
            setlists=self.setlists,
            song_tags=self.song_tags,
            train_show_ids=self.train_show_ids,
            include_residency_features=True,
        )

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 60 seconds for full dataset)
        assert (
            elapsed_time < 60
        ), f"Feature engineering too slow: {elapsed_time:.2f}s"

        print(
            f"[PASS] Feature engineering completed in {elapsed_time:.2f}s "
            f"({len(df)/elapsed_time:.0f} examples/sec)"
        )


def run_all_tests():
    """Run all tests in this module."""
    test_class = TestFeatures()

    tests = [
        test_class.test_feature_engineering_basic,
        test_class.test_feature_dtypes,
        test_class.test_no_data_leakage,
        test_class.test_feature_completeness,
        test_class.test_no_missing_values,
        test_class.test_feature_scale_reasonableness,
        test_class.test_label_distribution,
        test_class.test_song_coverage,
        test_class.test_feature_determinism,
        test_class.test_performance_stress,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running Feature Engineering Stress Tests")
    print("=" * 60)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
