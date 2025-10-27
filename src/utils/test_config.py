"""Test configuration constants.

This module provides centralized configuration for all tests, ensuring consistency
and making it easy to adjust test parameters globally.
"""

from dataclasses import dataclass

from src.utils.config import settings


@dataclass(frozen=True)
class TestConstants:
    """Constants used across test suites.

    These values are used for test data generation, model initialization,
    and validation thresholds. Centralizing them here makes tests more
    maintainable and consistent.
    """

    # Random seeds (use project default)
    RANDOM_SEED: int = settings.random_seed  # 42

    # Model architecture defaults for testing
    NUM_SONGS: int = 204  # Total songs in curated dataset
    EMBEDDING_DIM: int = 32  # Default embedding dimension for tests
    HIDDEN_DIMS: list = None  # Will be [64, 64] by default
    DROPOUT_RATE: float = 0.15  # Standard dropout for BERT-style masking

    # Training hyperparameters for tests
    BATCH_SIZE: int = 16  # Default batch size for test forward passes
    LEARNING_RATE: float = 0.001  # Standard learning rate

    # Performance thresholds
    MAX_FEATURE_ENGINEERING_TIME: int = 60  # seconds
    MAX_VENUE_LOOKUP_TIME: float = 0.1  # seconds for 50 lookups
    MAX_MEMORY_USAGE_MB: int = 100  # MB for full dataset

    # Data validation thresholds
    MAX_FEATURE_SCALE: float = 1e10  # Features shouldn't exceed this
    MIN_POSITIVE_RATE: float = 0.01  # Minimum % of positive labels
    MAX_POSITIVE_RATE: float = 0.5  # Maximum % of positive labels

    # Date range validation
    MIN_YEAR: int = 2000
    MAX_YEAR: int = 2030

    # Test data sizes for stress tests
    STRESS_TEST_VENUES: int = 50  # Number of venues for perf tests
    STRESS_TEST_LARGE_MAPPING: int = 10000  # Large venue mapping size

    # Numerical precision
    FLOAT_TOLERANCE: float = 1e-10  # For float comparisons
    FLOAT_RTOL: float = 1e-5  # Relative tolerance for model determinism

    def __post_init__(self):
        """Set mutable defaults after initialization."""
        if self.HIDDEN_DIMS is None:
            object.__setattr__(self, "HIDDEN_DIMS", [64, 64])


# Global test constants instance
TEST_CONSTANTS = TestConstants()


@dataclass(frozen=True)
class ModelTestConfig:
    """Configuration for model architecture tests.

    This allows testing different model configurations while keeping
    the test code clean and maintainable.
    """

    # DNNTSP / Temporal Sets with Priors
    dnntsp_embedding_dim: int = 32
    dnntsp_hidden_dims: list = None  # [64, 64]
    dnntsp_dropout_rate: float = 0.15

    # SFCN Large
    sfcn_large_hidden_dims: list = None  # [200, 200, 128]
    sfcn_large_freq_embedding_dim: int = 8

    # Graph construction
    graph_edge_count: int = 100  # For test graphs

    def __post_init__(self):
        """Set mutable defaults."""
        if self.dnntsp_hidden_dims is None:
            object.__setattr__(self, "dnntsp_hidden_dims", [64, 64])
        if self.sfcn_large_hidden_dims is None:
            object.__setattr__(self, "sfcn_large_hidden_dims", [200, 200, 128])


# Global model test config instance
MODEL_TEST_CONFIG = ModelTestConfig()


# Best practices for using this module:
#
# 1. Always import constants, don't hardcode values:
#    from src.utils.test_config import TEST_CONSTANTS
#    torch.manual_seed(TEST_CONSTANTS.RANDOM_SEED)
#
# 2. For model tests, use MODEL_TEST_CONFIG:
#    model = DNNTSP(
#        num_songs=TEST_CONSTANTS.NUM_SONGS,
#        embedding_dim=MODEL_TEST_CONFIG.dnntsp_embedding_dim,
#        hidden_dims=MODEL_TEST_CONFIG.dnntsp_hidden_dims,
#        dropout_rate=MODEL_TEST_CONFIG.dnntsp_dropout_rate
#    )
#
# 3. Override via environment variables when needed:
#    export RANDOM_SEED=123  # Will be picked up by settings
#
# 4. Document why you're overriding if you must hardcode in a test:
#    dropout_rate = 0.5  # Intentionally high to test dropout behavior
